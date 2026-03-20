#!/usr/bin/env python
"""
split_vllm.py - Universal vLLM Inference + Profiler

This script provides a generic, prompt-driven inference pipeline
using the vLLM engine. It supports any NLP task (translation,
summarisation, description, Q&A, etc.) by accepting a free-form
prompt template at runtime.

Key capabilities:
  - Load any HuggingFace-compatible model via vLLM.
  - Accept input from a file (--input), inline text (--text),
    or no external input at all (self-contained prompt).
  - Inject input content into a prompt via an {input_text}
    placeholder, or append it automatically when the placeholder
    is absent.
  - Fall back to a built-in translation prompt when no custom
    prompt is supplied (requires --source and --target).
  - Report per-phase (PREFILL / DECODE) token counts and wall
    times for performance profiling.

Usage examples:
  # Self-contained prompt, no input file needed
  python vllm_universal.py --output out.txt \\
      --prompt "Describe transformers in detail."

  # Summarise a file
  python vllm_universal.py --input doc.txt --output out.txt \\
      --prompt "Summarise the text below:\\n\\n{input_text}"

  # Default translation fallback
  python vllm_universal.py --input doc.txt --output out.txt \\
      --source en --target hi
"""

import gc
import logging
import os
import warnings
from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Tuple

import torch

# ------------------------------------------------------------------ #
# Suppress noisy vLLM and deprecation warnings before vLLM imports.  #
# ------------------------------------------------------------------ #
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
warnings.filterwarnings("ignore", category=DeprecationWarning)

# pylint: disable=wrong-import-position
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.outputs import RequestOutput

try:
    from vllm.tokenizers.get_tokenizer import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer

# ------------------------------------------------------------------ #
# Module-level logger.  Root vLLM loggers are silenced to WARNING.   #
# ------------------------------------------------------------------ #
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True, 
)
logger = logging.getLogger(__name__)


# ================================================================== #
#  Constants                                                          #
# ================================================================== #

# Map of ISO-639-1 codes to full language names used in prompts.
LANGUAGE_MAP = {
    "en": "English",
    "fr": "French",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
}


# ================================================================== #
#  Data classes                                                       #
# ================================================================== #

@dataclass(frozen=True)
class PhaseStats:
    """
    Immutable record of timing and token counts for one inference
    phase (PREFILL or DECODE).

    Attributes:
        phase_name (str): Label for the phase, e.g. 'PREFILL'.
        wall_s (float): Elapsed wall-clock time in seconds.
        device (str): Device used, e.g. 'GPU' or 'CPU+GPU'.
        prompt_tokens (int): Number of prompt tokens processed.
        gen_tokens (int): Number of tokens generated.
    """

    phase_name: str
    wall_s: float
    device: str
    prompt_tokens: int = 0
    gen_tokens: int = 0


# ================================================================== #
#  Engine helpers                                                     #
# ================================================================== #

class EngineManager:
    """
    Thin wrapper around :class:`vllm.LLMEngine` that encapsulates
    engine construction and per-phase inference stepping.

    Args:
        model (str): HuggingFace model identifier or local path.
        gpu_memory_utilization (float): Fraction of GPU memory for
            vLLM's KV cache (default 0.8).
        dtype (str): Weight dtype passed to vLLM (default 'auto').
        max_model_len (int): Maximum sequence length (default 32768).
    """

    def __init__(
        self,
        model: str,
        gpu_memory_utilization: float = 0.8,
        dtype: str = "auto",
        max_model_len: int = 32768,
    ) -> None:
        """Initialise and warm up the vLLM engine."""
        logger.debug(
            "Building engine: model=%s dtype=%s max_len=%d",
            model,
            dtype,
            max_model_len,
        )
        engine_args = EngineArgs(
            model=model,
            disable_log_stats=True,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            max_model_len=max_model_len,
        )
        self.engine: LLMEngine = LLMEngine.from_engine_args(engine_args)

    # -------------------------------------------------------------- #

    @staticmethod
    def _extract_data(
        request_output: RequestOutput,
    ) -> Tuple[int, int, str]:
        """
        Pull prompt token count, generated token count, and generated
        text out of a :class:`vllm.outputs.RequestOutput` object.

        Args:
            request_output (RequestOutput): Output from engine.step().

        Returns:
            Tuple[int, int, str]:
                (prompt_tokens, generated_tokens, generated_text)
        """
        prompt_tokens = (
            len(request_output.prompt_token_ids)
            if request_output.prompt_token_ids
            else 0
        )
        gen_tokens = 0
        text = ""
        if request_output.outputs:
            gen_tokens = len(request_output.outputs[0].token_ids)
            text = request_output.outputs[0].text
        return prompt_tokens, gen_tokens, text

    # -------------------------------------------------------------- #

    def run_phase(
        self,
        request_id: str,
        phase: str,
    ) -> Tuple[PhaseStats, str]:
        """
        Step the engine until the named phase completes.

        For PREFILL, returns as soon as the first generated token
        appears (one engine.step() producing >= 1 output token).
        For DECODE, runs until the request is marked finished.

        Args:
            request_id (str): Identifier passed to engine.add_request.
            phase (str): 'PREFILL' or 'DECODE'.

        Returns:
            Tuple[PhaseStats, str]:
                Timing/token statistics and the generated text so far.
        """
        start_time = perf_counter()
        final_text = ""
        final_ro: Optional[RequestOutput] = None

        print(f"\n>>> Phase: {phase}")
        torch.cuda.nvtx.range_push(phase)

        try:
            while self.engine.has_unfinished_requests():
                outputs: List[RequestOutput] = self.engine.step()
                for req_out in outputs:
                    if req_out.request_id != request_id:
                        continue
                    final_ro = req_out
                    p_tok, g_tok, text = self._extract_data(req_out)
                    final_text = text

                    # PREFILL ends on the first generated token.
                    if phase == "PREFILL" and g_tok >= 1:
                        torch.cuda.synchronize()
                        return (
                            PhaseStats(
                                phase,
                                perf_counter() - start_time,
                                "GPU",
                                p_tok,
                                g_tok,
                            ),
                            final_text,
                        )

                if final_ro and final_ro.finished:
                    break

            torch.cuda.synchronize()
            p, g, t = (
                self._extract_data(final_ro) if final_ro else (0, 0, "")
            )
            return (
                PhaseStats(phase, perf_counter() - start_time, "GPU", p, g),
                t,
            )

        finally:
            torch.cuda.nvtx.range_pop()

    # -------------------------------------------------------------- #

    def shutdown(self) -> None:
        """Delete the engine and free GPU memory."""
        del self.engine
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("Engine shut down and memory freed.")


# ================================================================== #
#  Text utilities                                                     #
# ================================================================== #

class TextProcessor:
    """
    Static helpers for cleaning raw input text and stripping
    garbage tokens from model output.
    """

    @staticmethod
    def normalize(text: str, aggressive: bool = False) -> str:
        """
        Strip dataset artefacts (brackets, stray quotes, comma-joined
        list elements) and collapse blank lines.

        Args:
            text (str): Raw text read from a file or CLI argument.
            aggressive (bool): When True, strip brackets and quotes
                in addition to normalising whitespace.
                (default: False)

        Returns:
            str: Cleaned, newline-separated text.
        """
        if aggressive:
          text = text.replace("[", "").replace("]", "")
          text = text.replace('",', "\n").replace("',", "\n")
          text = text.replace('"', "").replace("'", "")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)

    @staticmethod
    def clean_output(text: str, tokenizer) -> str:
        """
        Remove known garbage tokens and stray role markers that some
        models append to their output.

        Args:
            text (str): Raw generated text from the model.

        Returns:
            str: Cleaned generated text.
        """
      
        token_ids=tokenizer.encode(text, add_special_tokens=False)
        cleaned=tokenizer.decode(
          token_ids, skip_special_tokens=True
        )
        return cleaned.strip()


# ================================================================== #
#  Prompt builders                                                    #
# ================================================================== #

class PromptBuilder:
    """
    Constructs the final prompt string sent to the vLLM engine.

    Two modes:
      1. **Custom prompt** (``--prompt`` supplied): the caller's
         template is used verbatim.  If it contains ``{input_text}``
         the input content is injected there; otherwise the input is
         appended at the end.
      2. **Translation fallback** (no ``--prompt``): the built-in
         translation template is used; ``--source`` and ``--target``
         are then mandatory.
    """

    @staticmethod
    def _translation_prompt(
        input_text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Build the default literal-translation prompt.

        Args:
            input_text (str): Text to translate.
            source_lang (str): ISO-639-1 source language code.
            target_lang (str): ISO-639-1 target language code.

        Returns:
            str: Fully formed translation prompt.
        """
        src = LANGUAGE_MAP.get(source_lang.lower(), source_lang)
        tgt = LANGUAGE_MAP.get(target_lang.lower(), target_lang)

        return (
            f"\nYou are a professional translation engine performing"
            f" literal translation.\n\n"
            f"Translate the text from {src} to {tgt}.\n\n"
            f"Rules:\n"
            f"- Do NOT summarize\n"
            f"- Do NOT paraphrase\n"
            f"- Preserve every sentence\n"
            f"- Preserve all details and information\n"
            f"- Translate sentence-by-sentence\n"
            f"- If a sentence seems redundant, still translate it\n\n"
            f"Output ONLY the translation.\n\n"
            f"Text:\n{input_text}\n\nTranslation:\n"
        )

    # -------------------------------------------------------------- #

    @classmethod
    def build(cls, input_text: str, args) -> str:
        """
        Produce the final prompt for the model.

        Logic:
          - If ``args.prompt`` is given and ``input_text`` is empty,
            return the prompt as-is (self-contained).
          - If ``args.prompt`` contains ``{input_text}``, inject the
            content at that placeholder.
          - If ``args.prompt`` has no placeholder, append the content
            after the prompt.
          - If no ``args.prompt``, use the built-in translation
            template (``args.source`` and ``args.target`` required).

        Args:
            input_text (str): Normalised input content (may be empty).
            args (argparse.Namespace): Parsed CLI arguments.

        Returns:
            str: The complete prompt ready for the engine.

        Raises:
            ValueError: When no prompt and no source/target are given.
        """
        if args.prompt:
            if not input_text:
                # Self-contained prompt; nothing to inject.
                return args.prompt
            if "{input_text}" in args.prompt:
                return args.prompt.format(input_text=input_text)
            # Append mode: input goes after the prompt.
            return f"{args.prompt}\n\n{input_text}\n\nOutput:\n"

        # ---- Translation fallback --------------------------------- #
        if not args.source or not args.target:
            raise ValueError(
                "No --prompt provided.  For the default translation "
                "mode --source and --target are required."
            )
        return cls._translation_prompt(
            input_text, args.source, args.target
        )


# ================================================================== #
#  Profiling reporter                                                 #
# ================================================================== #

class ProfileReporter:
    """
    Formats and prints the inference profiling table to stdout.
    """

    @staticmethod
    def report(
        output_path: str,
        init_stats: PhaseStats,
        prefill_stats: PhaseStats,
        decode_stats: PhaseStats,
    ) -> None:
        """
        Print token counts, per-phase timings, and total latency.

        Args:
            output_path (str): Path where the result was written.
            init_stats (PhaseStats): Engine initialisation stats.
            prefill_stats (PhaseStats): PREFILL phase stats.
            decode_stats (PhaseStats): DECODE phase stats.
        """
        prompt_tokens = decode_stats.prompt_tokens
        generated_tokens = decode_stats.gen_tokens
        ratio = (
            generated_tokens / prompt_tokens if prompt_tokens else 0.0
        )

        print(f"\nSaved output to: {output_path}")
        print("\n================ INFERENCE PROFILING ================")
        print(f"Input tokens (prompt)     : {prompt_tokens}")
        print(f"Output tokens (generated) : {generated_tokens}")
        print(f"Output/Input ratio        : {ratio:.3f}")
        print(
            "\nPHASE           TIME(s)    DEVICE"
            "   INPUT TOKENS   OUTPUT TOKENS"
        )
        print(
            "-------------------------------------------------"
            "----------------"
        )
        print(
            f"INITIALIZATION  "
            f"{init_stats.wall_s:.4f}   CPU+GPU  "
            f"0              0"
        )
        print(
            f"PREFILL         "
            f"{prefill_stats.wall_s:.4f}   GPU      "
            f"{prefill_stats.prompt_tokens:<14}"
            f"{prefill_stats.gen_tokens}"
        )
        print(
            f"DECODE          "
            f"{decode_stats.wall_s:.4f}   GPU      "
            f"{decode_stats.prompt_tokens:<14}"
            f"{decode_stats.gen_tokens}"
        )
        print(
            "-------------------------------------------------"
            "----------------"
        )
        total = prefill_stats.wall_s + decode_stats.wall_s
        print(f"Total inference latency: {total:.3f} sec\n")


# ================================================================== #
#  CLI argument parser                                                #
# ================================================================== #

def build_arg_parser() -> "argparse.ArgumentParser":
    """
    Construct the argument parser for the CLI entry-point.

    Returns:
        argparse.ArgumentParser: Configured parser instance.
    """
    import argparse  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(
        description=(
            "Universal vLLM inference script.  Accepts any prompt "
            "template and runs inference on a given input, writing "
            "the result to an output file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Self-contained prompt, no input needed\n"
            "  %(prog)s --output out.txt \\\n"
            '      --prompt "Describe transformers in detail."\n\n'
            "  # Summarise a file\n"
            "  %(prog)s --input doc.txt --output out.txt \\\n"
            '      --prompt "Summarise:\\n\\n{input_text}"\n\n'
            "  # Default translation fallback\n"
            "  %(prog)s --input doc.txt --output out.txt \\\n"
            "      --source en --target hi\n"
        ),
    )

    parser.add_argument(
        "--input",
        default=None,
        metavar="FILE",
        help=(
            "Path to a plain-text input file whose content is "
            "injected into the prompt.  Mutually exclusive with "
            "--text.  (default: None)"
        ),
    )
    parser.add_argument(
        "--text",
        default=None,
        metavar="STRING",
        help=(
            "Inline input text injected into the prompt.  Use "
            "instead of --input when the content is short enough "
            "to supply on the command line.  (default: None)"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="FILE",
        help="Path to the file where generated output is written.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B",
        metavar="MODEL",
        help=(
            "HuggingFace model identifier or local path. "
            "(default: Qwen/Qwen2.5-1.5B)"
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        metavar="TEMPLATE",
        help=(
            "Full prompt template.  Use {input_text} as a "
            "placeholder for the input content.  If the placeholder "
            "is absent, the content is appended automatically.  "
            "When omitted the built-in translation prompt is used "
            "and --source / --target become required.  "
            "(default: None)"
        ),
    )
    parser.add_argument(
        "--source",
        default=None,
        metavar="LANG",
        help=(
            "Source language ISO-639-1 code, e.g. 'en'.  "
            "Required only when --prompt is omitted "
            "(translation fallback mode).  (default: None)"
        ),
    )
    parser.add_argument(
        "--target",
        default=None,
        metavar="LANG",
        help=(
            "Target language ISO-639-1 code, e.g. 'hi'.  "
            "Required only when --prompt is omitted "
            "(translation fallback mode).  (default: None)"
        ),
    )
    parser.add_argument(
        "--aggressive-normalize",
        action="store_true",
        default=False,
        help=(
            "Strip brackets and quotes from input text in addition "
            "to normalising whitespace.  Useful for dataset "
            "artefacts; disable when input is structured data such "
            "as JSON.  (default: False)"
        ),
    )
    return parser


# ================================================================== #
#  Orchestrator                                                       #
# ================================================================== #

class InferencePipeline:
    """
    High-level orchestrator that wires together argument parsing,
    input loading, prompt construction, engine inference, output
    saving, and profiling.

    Designed to be importable so other scripts can call
    ``InferencePipeline().run(args)`` directly.
    """

    # -------------------------------------------------------------- #

    @staticmethod
    def _load_input(args) -> str:
        """
        Read and normalise input text from a file, inline string, or
        return an empty string when no input source is specified.

        Args:
            args (argparse.Namespace): Parsed CLI arguments.

        Returns:
            str: Normalised input text (may be empty).
        """
        aggressive = getattr(args, "aggressive_normalize", False)
        if args.text:
            logger.debug("Using inline --text as input.")
            return TextProcessor.normalize(args.text, aggressive)
        if args.input:
            logger.debug("Reading input file: %s", args.input)
            with open(args.input, "r", encoding="utf-8") as fh:
                return TextProcessor.normalize(fh.read(), aggressive)
        logger.debug(
            "No --input or --text supplied; "
            "prompt is treated as self-contained."
        )
        return ""

    # -------------------------------------------------------------- #

    def run(self, args) -> None:
        """
        Execute the full inference pipeline.

        Steps:
          1. Initialise the vLLM engine.
          2. Load and normalise input text.
          3. Build the final prompt.
          4. Tokenise and set sampling parameters.
          5. Run PREFILL then DECODE phases.
          6. Clean model output via tokenizer.
          7. Write output to disk.
          8. Print profiling table.

        Args:
            args (argparse.Namespace): Parsed CLI arguments.
        """
        # ---- Engine initialisation -------------------------------- #
        print("\nLoading model...")
        init_start = perf_counter()
        manager = EngineManager(model=args.model)
        init_stats = PhaseStats(
            "INITIALIZATION", perf_counter() - init_start, "CPU+GPU"
        )

        # ---- Input loading ---------------------------------------- #
        input_text = self._load_input(args)

        # ---- Tokeniser -------------------------------------------- #
        tokenizer = get_tokenizer(args.model)
        eos_token = tokenizer.eos_token

        # ---- Prompt construction ---------------------------------- #
        prompt = PromptBuilder.build(input_text, args)
        logger.debug("Prompt length: %d chars", len(prompt))

        input_tokens = len(tokenizer.encode(prompt))

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=0.9,
            repetition_penalty=1.15,
            max_tokens=int(input_tokens * 1.3),
            stop=[eos_token],
        )

        # ---- Inference -------------------------------------------- #
        request_id = "inference"
        manager.engine.add_request(request_id, prompt, sampling_params)

        prefill_stats, _ = manager.run_phase(request_id, "PREFILL")
        decode_stats, output_text = manager.run_phase(
            request_id, "DECODE"
        )

        # ---- Post-processing -------------------------------------- #
        clean_text = TextProcessor.clean_output(output_text, tokenizer)

        # ---- Save output ------------------------------------------ #
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(clean_text)

        # ---- Profiling -------------------------------------------- #
        ProfileReporter.report(
            args.output, init_stats, prefill_stats, decode_stats
        )

        # ---- Cleanup ---------------------------------------------- #
        manager.shutdown()


# ================================================================== #
#  CLI entry-point                                                    #
# ================================================================== #

def main() -> None:
    """
    Parse CLI arguments and run the inference pipeline.

    This function is the entry-point when the script is executed
    directly or installed as a console script.
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    InferencePipeline().run(args)


if __name__ == "__main__":
    main()
