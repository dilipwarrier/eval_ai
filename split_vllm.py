#!/usr/bin/env python
"""
vLLM Phase Profiler and Performance Benchmarker

This script measures and analyzes the three primary phases of Large Language
Model inference on GPU hardware: INITIALIZATION, PREFILL, and DECODE.
It supports both local GPU execution and remote execution via an OpenAI
compatible API. It can be used as a standalone CLI tool or as a Promptfoo
Python provider.
"""

import os
import sys
import warnings
import argparse
import gc
import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import List, Tuple, Dict, Any

import requests
from openai import OpenAI

# Standard vLLM environment controls
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["VLLM_REPORT_STATS"] = "0"

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from vllm import EngineArgs, LLMEngine, SamplingParams
    from vllm.outputs import RequestOutput
    try:
        from vllm.tokenizers.get_tokenizer import get_tokenizer
    except ImportError:
        from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    # Graceful fallback for type hinting if vLLM is missing locally
    LLMEngine = Any
    RequestOutput = Any
    EngineArgs = Any
    SamplingParams = Any
    get_tokenizer = None

# Basic logging configuration
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
    force=True
)
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass(frozen=True)
class PhaseStats:
    """
    Data class representing performance statistics for a specific phase.
    """
    phase_name: str
    wall_s: float
    device: str
    prompt_tokens: int = 0
    gen_tokens: int = 0


class RemoteVLLMClient:
    """
    Client to interact with and profile a remote vLLM OpenAI server.
    """

    def __init__(self, ip: str, port: int, model: str):
        """
        Initialize the remote client configuration.

        Args:
            ip: The IP address of the remote server.
            port: The port number of the remote server.
            model: The model identifier to use for requests.
        """
        self.ip = ip
        self.port = port
        self.model = model
        self.base_url = f"http://{self.ip}:{self.port}/v1"
        self.api_key = "llm-d-local"

    def test_connection(self) -> None:
        """
        Test the connection to the remote server.
        Exits the script with an error message if the connection fails.
        """
        url = f"{self.base_url}/models"
        try:
            logging.info("Testing connection to server %s...", self.ip)
            response = requests.get(url, timeout=2.0)
            response.raise_for_status()
            logging.info("Connection successful.")
        except requests.exceptions.ConnectionError:
            logging.error("Connection refused at %s.", url)
            logging.error(
                "Ensure server %s is running and port %s is open.",
                self.ip, self.port
            )
            sys.exit(1)
        except requests.exceptions.Timeout:
            logging.error("Connection to %s timed out.", url)
            sys.exit(1)
        except Exception as err:  # pylint: disable=broad-exception-caught
            logging.error("Unexpected error connecting to server: %s", err)
            sys.exit(1)

    def profile(self, prompt: str) -> dict:
        """
        Profile the remote API by measuring TTFT (Prefill) and Decode.

        Args:
            prompt: The user prompt to send to the model.

        Returns:
            A dictionary formatted for Promptfoo containing metrics.
        """
        init_start = perf_counter()
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        init_duration = perf_counter() - init_start

        start_time = perf_counter()
        first_token_time = None
        full_text = ""
        prompt_tok = 0
        gen_tok = 0

        logging.info("Sending streaming request to remote server...")
        try:
            stream = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True}
            )

            for chunk in stream:
                delta = ""
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content

                if first_token_time is None and delta:
                    first_token_time = perf_counter()

                if delta:
                    full_text += delta

                if chunk.usage:
                    prompt_tok = chunk.usage.prompt_tokens
                    gen_tok = chunk.usage.completion_tokens

        except Exception as err:  # pylint: disable=broad-exception-caught
            logging.error("Error during remote inference: %s", err)
            sys.exit(1)

        end_time = perf_counter()
        first_token_time = first_token_time or end_time

        prefill_s = first_token_time - start_time
        decode_s = end_time - first_token_time
        total_s = end_time - start_time

        return {
            "output": full_text,
            "metadata": {
                "formatted_prompt": prompt,
                "init_s": round(init_duration, 4),
                "init_input_tokens": 0,
                "init_output_tokens": 0,
                "prefill_s": round(prefill_s, 4),
                "prefill_input_tokens": prompt_tok,
                "prefill_output_tokens": 1,
                "decode_s": round(decode_s, 4),
                "decode_input_tokens": 0,
                "decode_output_tokens": gen_tok,
                "total_s": round(total_s, 4),
                "total_input_tokens": prompt_tok,
                "total_output_tokens": gen_tok,
            },
            "latencyMs": round(total_s * 1000, 2)
        }


def build_engine(model: str, **engine_kwargs) -> LLMEngine:
    """
    Initializes the LLMEngine with background stats disabled.

    Args:
        model: The identifier for the model.
        engine_kwargs: Additional arguments for the engine.

    Returns:
        The initialized LLMEngine.
    """
    engine_args = EngineArgs(
        model=model, disable_log_stats=True, **engine_kwargs
    )
    return LLMEngine.from_engine_args(engine_args)


def _extract_data(ro: RequestOutput) -> Tuple[int, int, str]:
    """
    Safely extracts token counts and text from RequestOutput.

    Args:
        ro: The RequestOutput object from vLLM.

    Returns:
        A tuple of (prompt_tokens, generated_tokens, generated_text).
    """
    p_tokens = len(ro.prompt_token_ids) if ro.prompt_token_ids else 0
    g_tokens = 0
    text = ""
    if ro.outputs:
        g_tokens = len(ro.outputs[0].token_ids)
        text = ro.outputs[0].text
    return p_tokens, g_tokens, text


def run_phase(
    engine: LLMEngine, request_id: str, phase: str
) -> Tuple[PhaseStats, str]:
    """
    Manually steps the engine through a specific inference phase.

    Args:
        engine: The local vLLM engine instance.
        request_id: The unique request identifier.
        phase: The phase name (e.g., 'PREFILL', 'DECODE').

    Returns:
        A tuple containing PhaseStats and the final text.
    """
    start_time = perf_counter()
    final_text = ""
    final_ro = None

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for req_out in request_outputs:
            if req_out.request_id == request_id:
                final_ro = req_out
                p_count, g_count, current_text = _extract_data(req_out)
                final_text = current_text

                if phase == "PREFILL" and g_count >= 1:
                    stats = PhaseStats(
                        phase, perf_counter() - start_time, "GPU",
                        p_count, g_count
                    )
                    return stats, final_text

        if final_ro and final_ro.finished:
            break

    if final_ro:
        p_count, g_count, current_text = _extract_data(final_ro)
    else:
        p_count, g_count, current_text = 0, 0, ""

    stats = PhaseStats(
        phase, perf_counter() - start_time, "GPU", p_count, g_count
    )
    return stats, current_text


def call_api(
    prompt: str, options: dict, context: dict # pylint: disable=unused-argument
) -> dict:
    """
    Interface for Promptfoo Python Provider. Non-persistent version.

    Args:
        prompt: The user prompt to send.
        options: Dictionary containing config arguments.
        context: Promptfoo context data.

    Returns:
        A dictionary containing the response output and metadata.
    """
    config = options.get('config', {})
    model_name = config.get('model', "Qwen/Qwen2.5-1.5B-Instruct")
    save_path = config.get('save_output')
    is_remote = config.get('remote', False)

    if is_remote:
        ip_addr = config.get('ip', 'localhost')
        port_num = config.get('port', 8080)
        client = RemoteVLLMClient(ip_addr, port_num, model_name)
        client.test_connection()
        result = client.profile(prompt)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as file_obj:
                file_obj.write(result["output"])
        return result

    enforce_eager = config.get('enforce_eager', False)
    gpu_util = config.get('gpu_util', 0.9)

    # --- PHASE 1: INITIALIZATION ---
    init_start = perf_counter()
    vllm_engine = build_engine(
        model=model_name,
        gpu_memory_utilization=gpu_util,
        enforce_eager=enforce_eager
    )
    init_duration = perf_counter() - init_start

    try:
        tokenizer = get_tokenizer(model_name)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(max_tokens=2048, temperature=0.7)
        request_id = "bench_request"

        vllm_engine.add_request(
            request_id, formatted_prompt, sampling_params
        )

        # --- PHASE 2: PREFILL ---
        stat_p, _ = run_phase(vllm_engine, request_id, "PREFILL")

        # --- PHASE 3: DECODE ---
        stat_d, full_text = run_phase(vllm_engine, request_id, "DECODE")

        total_s = init_duration + stat_p.wall_s + stat_d.wall_s
        total_in = stat_p.prompt_tokens
        total_out = stat_d.gen_tokens

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as file_obj:
                file_obj.write(full_text)

        return {
            "output": full_text,
            "metadata": {
                "formatted_prompt": formatted_prompt,
                "init_s": round(init_duration, 4),
                "init_input_tokens": 0,
                "init_output_tokens": 0,
                "prefill_s": round(stat_p.wall_s, 4),
                "prefill_input_tokens": stat_p.prompt_tokens,
                "prefill_output_tokens": 1,
                "decode_s": round(stat_d.wall_s, 4),
                "decode_input_tokens": 0,
                "decode_output_tokens": stat_d.gen_tokens,
                "total_s": round(total_s, 4),
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
            },
            "latencyMs": round(total_s * 1000, 2)
        }
    finally:
        del vllm_engine
        gc.collect()
        try:
            import torch  # pylint: disable=import-outside-toplevel
            torch.cuda.empty_cache()
        except ImportError:
            pass


class ProfilerCLI:
    """
    Command Line Interface wrapper for the vLLM Profiler.
    """

    @staticmethod
    def print_table(meta: Dict[str, Any]) -> None:
        """
        Prints a formatted table of metrics.

        Args:
            meta: The metadata dictionary containing phase timings.
        """
        header = (
            f"{'PHASE':<15} | {'TIME (s)':<10} | "
            f"{'INPUT TOKENS':<12} | {'OUTPUT TOKENS':<13}"
        )
        print(header)
        print("-" * 58)

        rows = [
            (
                "Initialization", meta["init_s"],
                meta["init_input_tokens"], meta["init_output_tokens"]
            ),
            (
                "Prefill", meta["prefill_s"],
                meta["prefill_input_tokens"], meta["prefill_output_tokens"]
            ),
            (
                "Decode", meta["decode_s"],
                meta["decode_input_tokens"], meta["decode_output_tokens"]
            ),
            (
                "Total", meta["total_s"],
                meta["total_input_tokens"], meta["total_output_tokens"]
            ),
        ]

        for name, time_s, in_tok, out_tok in rows:
            if name == "Total":
                print("-" * 58)
            print(
                f"{name:<15} | {time_s:<10.4f} | "
                f"{in_tok:<12} | {out_tok:<13}"
            )
        print("=" * 58)

    @classmethod
    def run(cls) -> None:
        """
        Parses arguments and executes the profile run.
        """
        desc = ("vLLM profiler: Respond to an AI prompt, run inference to "
                "generate a response, and measure execution times for the "
                "prefill and decode steps.")
        parser = argparse.ArgumentParser(
            description=desc,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        group_eng = parser.add_argument_group("Engine Configuration")
        group_eng.add_argument(
            "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
            help="Path or HF repo ID of the model to load"
        )
        group_eng.add_argument(
            "--gpu-util", type=float, default=0.9,
            help="Fraction of GPU memory to reserve"
        )
        group_eng.add_argument(
            "--enforce-eager", action="store_true",
            help="Disable CUDA graph capturing"
        )

        group_inf = parser.add_argument_group("Inference Options")
        group_inf.add_argument(
            "--prompt", type=str,
            default="Write a 1000 word essay on the Enlightenment movement",
            help="Input text to process"
        )
        group_inf.add_argument(
            "--prompt-file", type=str, metavar="PATH",
            help="Path to a file whose content will be appended to the prompt"
        )
        group_inf.add_argument(
            "--save-output", type=str, metavar="PATH",
            help="Save the generated text to a specific file"
        )

        group_rem = parser.add_argument_group("Remote Execution")
        group_rem.add_argument(
            "--remote", action="store_true",
            help="Route inference to a remote server"
        )
        group_rem.add_argument(
            "--ip", type=str, default="localhost",
            help="IP address of the remote server"
        )
        group_rem.add_argument(
            "--port", type=int, default=8080,
            help="Port of the remote server"
        )

        group_out = parser.add_argument_group("Output Control")
        group_out.add_argument(
            "--json", action="store_true",
            help="Output result as JSON for Promptfoo compatibility"
        )

        args = parser.parse_args()

        # Update: Load content from prompt-file if provided
        if args.prompt_file:
            if not os.path.exists(args.prompt_file):
                logging.error("Prompt file not found: %s", args.prompt_file)
                sys.exit(1)
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
                # Append file content to the command-line prompt
                args.prompt = f"{args.prompt}\n\n{file_content}"

        res = call_api(args.prompt, {"config": vars(args)}, {})

        if args.json:
            if len(res['output']) > 200:
                res['output'] = res['output'][:200] + "..."
            sys.stdout.write(json.dumps(res, indent=4) + "\n")
        else:
            print("\n" + "=" * 58)
            print("FORMATTED PROMPT")
            print("-" * 58)
            print(f"Prompt:\n{res['metadata']['formatted_prompt']}")

            print("\n" + "=" * 58)
            print("PHASE PERFORMANCE METRICS")
            print("-" * 58)

            cls.print_table(res["metadata"])

            if args.save_output:
                print(f"\nFull output saved to: {args.save_output}")
            else:
                print("\n" + "=" * 58)
                print("FULL LLM OUTPUT")
                print("-" * 58)
                print(res['output'])
                print("=" * 58)


if __name__ == "__main__":
    ProfilerCLI.run()
