#!/usr/bin/env python
"""
vLLM Phase Profiler and Performance Benchmarker

This script measures and analyzes the three primary phases of Large Language
Model inference on Lyptus hardware:
1. Initialization: Model loading, VRAM allocation, and CUDA graph capture.
2. Prefill (TTFT): Initial processing of the prompt and first token generation.
3. Decode (TPOT): Sequential generation of subsequent tokens.

Supports structured JSON for automated analysis and file-based output saving.
"""

import os
import sys
import warnings
import argparse
import gc
import json
import logging
from dataclasses import dataclass, asdict
from time import perf_counter
from typing import List, Optional, Tuple

# Standard vLLM environment controls
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["VLLM_REPORT_STATS"] = "0"

warnings.filterwarnings("ignore", category=DeprecationWarning)

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.outputs import RequestOutput

try:
    from vllm.tokenizers.get_tokenizer import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer

# Basic logging configuration
logging.basicConfig(level=logging.ERROR)
logging.getLogger("vllm").setLevel(logging.ERROR)

@dataclass(frozen=True)
class PhaseStats:
    phase_name: str
    wall_s: float
    device: str
    prompt_tokens: int = 0
    gen_tokens: int = 0

def build_engine(model: str, **engine_kwargs) -> LLMEngine:
    """Initializes the LLMEngine with background stats disabled."""
    engine_args = EngineArgs(model=model, disable_log_stats=True, **engine_kwargs)
    return LLMEngine.from_engine_args(engine_args)

def _extract_data(ro: RequestOutput) -> Tuple[int, int, str]:
    """Safely extracts token counts and text from the RequestOutput object."""
    p_tokens = len(ro.prompt_token_ids) if ro.prompt_token_ids else 0
    g_tokens = 0
    text = ""
    if ro.outputs:
        g_tokens = len(ro.outputs[0].token_ids)
        text = ro.outputs[0].text
    return p_tokens, g_tokens, text

def run_phase(engine: LLMEngine, request_id: str, phase: str) -> Tuple[PhaseStats, str]:
    """Manually steps the engine. Streaming to screen is disabled."""
    start_time = perf_counter()
    final_text = ""
    final_ro = None

    while engine.has_unfinished_requests():
        request_outputs: List[RequestOutput] = engine.step()
        for ro in request_outputs:
            if ro.request_id == request_id:
                final_ro = ro
                p_count, g_count, current_text = _extract_data(ro)
                final_text = current_text

                # Prefill logic: returns as soon as the first token exists
                if phase == "PREFILL" and g_count >= 1:
                    return PhaseStats(phase, perf_counter() - start_time, "GPU", p_count, g_count), final_text

        if final_ro and final_ro.finished:
            break

    p, g, t = _extract_data(final_ro) if final_ro else (0, 0, "")
    return PhaseStats(phase, perf_counter() - start_time, "GPU", p, g), t

def main():
    parser = argparse.ArgumentParser(description="vLLM Phase Profiler CLI")
    parser.add_argument("--prompt", type=str, default="Write a 1000 word essay on the Enlightenment movement")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--gpu-util", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Skip CUDA graph capture. Faster startup, but slower inference throughput.")
    parser.add_argument("--json", action="store_true", help="Output results in clean JSON format")
    parser.add_argument("--save-text", type=str, help="Path to save the generated LLM output text locally.")
    args = parser.parse_args()

    # --- 1. INITIALIZATION (CPU + GPU) ---
    init_start = perf_counter()
    vllm_engine = build_engine(
        model=args.model,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_util,
        enforce_eager=args.enforce_eager
    )
    init_stats = PhaseStats("INITIALIZATION", perf_counter() - init_start, "CPU+GPU")

    tokenizer = get_tokenizer(args.model)
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True
    )

    sampling_params = SamplingParams(max_tokens=2048, temperature=0.7)
    request_id = "bench_0"
    results = [init_stats]

    try:
        vllm_engine.add_request(request_id, formatted_prompt, sampling_params)

        # Inference runs silently
        stat_p, _ = run_phase(vllm_engine, request_id, "PREFILL")
        results.append(stat_p)

        stat_d, full_output_text = run_phase(vllm_engine, request_id, "DECODE")
        results.append(stat_d)

        if args.save_text:
            with open(args.save_text, 'w') as f:
                f.write(full_output_text)
        else:
            print("\n" + "="*45)
            print("LLM OUTPUT:")
            print("-" * 45)
            print(full_output_text)
            print("="*45)

        if args.json:
            sys.stdout.write(json.dumps([asdict(r) for r in results]))
            sys.stdout.flush()
        else:
            # Only print the summary table, no streaming text
            print("\n" + "="*45)
            for r in results:
                print(f"{r.phase_name:15} | {r.wall_s:8.4f}s | {r.device}")
            print("="*45)

    finally:
        del vllm_engine
        gc.collect()

if __name__ == "__main__":
    main()
