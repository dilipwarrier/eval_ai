#!/usr/bin/env python
"""
vLLM Phase Profiler and Performance Benchmarker

This script provides a framework to measure and analyze the two primary phases of
Large Language Model inference:
1. Prefill (Time to First Token - TTFT): The initial processing of the prompt.
2. Decode (Time Per Output Token - TPOT): The sequential generation of subsequent tokens.

By manually stepping through the vLLM LLMEngine, the script captures granular
timing data for each phase while streaming generated text to the terminal in
real-time. It uses the model's chat template to ensure command-style execution
rather than text completion.
"""

import os
# Set logging level before importing vLLM to silence internal INFO logs
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import argparse
import gc
import logging
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Tuple

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.outputs import RequestOutput

# Robust import for get_tokenizer to handle different vLLM versions
try:
    from vllm.tokenizers.get_tokenizer import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer

# Configure local script logging to INFO to see model parameters
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhaseStats:
    """Stores performance metrics for a specific execution phase."""

    wall_s: float
    prompt_tokens: int
    gen_tokens: int
    text: str


def inspect_and_log_config(engine: LLMEngine):
    """Extracts and logs the resolved configuration parameters of the engine."""
    config = engine.vllm_config.model_config
    cache_config = engine.vllm_config.cache_config
    parallel_config = engine.vllm_config.parallel_config
    scheduler_config = engine.vllm_config.scheduler_config

    logger.info("--- Resolved Model Parameters ---")
    logger.info(f"Model: {config.model}")
    logger.info(f"DataType: {config.dtype}")
    logger.info(f"Max Model Len: {config.max_model_len}")
    logger.info(f"Enforce Eager: {config.enforce_eager}")

    logger.info("--- Cache Configuration ---")
    logger.info(f"Block Size: {cache_config.block_size}")
    logger.info(f"GPU Memory Utilization: {cache_config.gpu_memory_utilization}")

    logger.info("--- Parallel Configuration ---")
    logger.info(f"Pipeline Parallel Size: {parallel_config.pipeline_parallel_size}")
    logger.info(f"Tensor Parallel Size: {parallel_config.tensor_parallel_size}")

    logger.info("--- Scheduler Configuration ---")
    logger.info(f"Max Num Batched Tokens: {scheduler_config.max_num_batched_tokens}")
    logger.info(f"Max Num Seq: {scheduler_config.max_num_seqs}")
    logger.info("----------------------------------")


def build_engine(model: str, **engine_kwargs) -> LLMEngine:
    """Initializes the LLMEngine with the specified model and hardware args."""
    engine_args = EngineArgs(model=model, **engine_kwargs)
    return LLMEngine.from_engine_args(engine_args)


def _extract_data(ro: RequestOutput) -> Tuple[int, int, str]:
    """Safely extracts token counts and text from the RequestOutput object."""
    p_tokens = len(ro.prompt_token_ids) if ro.prompt_token_ids else 0
    g_tokens = 0
    text = ""
    if ro.outputs:
        out = ro.outputs[0]
        g_tokens = len(out.token_ids)
        text = out.text
    return p_tokens, g_tokens, text


def run_phase(engine: LLMEngine, request_id: str, phase: str) -> PhaseStats:
    """
    Executes an engine loop for a specific phase (PREFILL or DECODE).
    Streams new tokens to the terminal and captures timing data.
    """
    start_time = perf_counter()
    last_len = 0
    final_ro = None

    print(f"\n>>> Phase: {phase}")

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()

        for ro in request_outputs:
            if ro.request_id == request_id:
                final_ro = ro
                _, gen_count, current_text = _extract_data(ro)

                # Stream incremental text using standard print
                new_text = current_text[last_len:]
                if new_text:
                    print(new_text, end="")
                    sys.stdout.flush()
                    last_len = len(current_text)

                # Prefill stops immediately after the first token is produced
                if phase == "PREFILL" and gen_count >= 1:
                    return _create_stats(start_time, ro)

        if final_ro and final_ro.finished:
            break

    return _create_stats(start_time, final_ro)


def _create_stats(start_time: float, ro: Optional[RequestOutput]) -> PhaseStats:
    """Helper to package metrics into PhaseStats."""
    duration = perf_counter() - start_time
    if not ro:
        return PhaseStats(duration, 0, 0, "")
    p, g, t = _extract_data(ro)
    return PhaseStats(duration, p, g, t)


def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(description="vLLM Phase Profiler CLI")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the Pythagoras theorem.",
        help="The text prompt to send to the model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="The model identifier to use for benchmarking."
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum context length to reduce VRAM and KV cache overhead."
    )
    args = parser.parse_args()

    # Build engine with optimized sequence length
    vllm_engine = build_engine(model=args.model, max_model_len=args.max_model_len)

    # Log resolved parameters
    inspect_and_log_config(vllm_engine)

    # Apply Chat Template to ensure instruct-style response
    tokenizer = get_tokenizer(args.model)
    messages = [{"role": "user", "content": args.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(max_tokens=1024, temperature=0.7)
    request_id = "demo0"

    try:
        vllm_engine.add_request(request_id, formatted_prompt, sampling_params)

        # Execute Phase 1: Prefill (TTFT)
        prefill_results = run_phase(vllm_engine, request_id, "PREFILL")

        # Execute Phase 2: Decode (TPOT)
        decode_results = run_phase(vllm_engine, request_id, "DECODE")

        # Final Summary Print
        print("\n\n" + "=" * 40)
        print("BENCHMARKING REPORT")
        print("=" * 40)
        prompt_preview = (
            f"{args.prompt[:50]}..." if len(args.prompt) > 50 else args.prompt
        )
        print(f"Prompt:         {prompt_preview}")
        print(
            f"Prefill (TTFT): {prefill_results.wall_s:.4f}s | "
            f"Tokens: {prefill_results.prompt_tokens}"
        )
        print(
            f"Decode (TPOT):  {decode_results.wall_s:.4f}s | "
            f"Tokens: {decode_results.gen_tokens}"
        )
        total_wall = prefill_results.wall_s + decode_results.wall_s
        print(f"Total Wall:     {total_wall:.4f}s")
        print("=" * 40)

    finally:
        # Explicitly delete engine and collect garbage to release GPU VRAM
        del vllm_engine
        gc.collect()
        print("\nEngine successfully deallocated.")


if __name__ == "__main__":
    main()
