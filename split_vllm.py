#!/usr/bin/env python
"""
vLLM Phase Profiler and Performance Benchmarker

This script measures and analyzes the three primary phases of Large Language
Model inference on GPU hardware: INITIALIZATION, PREFILL, and DECODE.
It can be used as a standalone CLI tool or as a Promptfoo Python provider.
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
    engine_args = EngineArgs(
        model=model, disable_log_stats=True, **engine_kwargs
    )
    return LLMEngine.from_engine_args(engine_args)

def _extract_data(ro: RequestOutput) -> Tuple[int, int, str]:
    """Safely extracts token counts and text from RequestOutput."""
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
    """Manually steps the engine through a specific inference phase."""
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

                if phase == "PREFILL" and g_count >= 1:
                    return (
                        PhaseStats(
                            phase, perf_counter() - start_time, "GPU",
                            p_count, g_count
                        ),
                        final_text,
                    )

        if final_ro and final_ro.finished:
            break

    p, g, t = _extract_data(final_ro) if final_ro else (0, 0, "")
    return (
        PhaseStats(phase, perf_counter() - start_time, "GPU", p, g),
        t,
    )

def call_api(prompt: str, options: dict, context: dict) -> dict:
    """Interface for Promptfoo Python Provider. Non-persistent version."""
    config = options.get('config', {})
    model_name = config.get('model', "Qwen/Qwen2.5-1.5B-Instruct")
    enforce_eager = config.get('enforce_eager', False)
    gpu_util = config.get('gpu_util', 0.9)
    save_path = config.get('save_output')

    vllm_engine = build_engine(
        model=model_name,
        gpu_memory_utilization=gpu_util,
        enforce_eager=enforce_eager
    )

    try:
        tokenizer = get_tokenizer(model_name)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(max_tokens=2048, temperature=0.7)
        request_id = "bench_request"

        vllm_engine.add_request(request_id, formatted_prompt, sampling_params)
        inference_start = perf_counter()

        stat_p, _ = run_phase(vllm_engine, request_id, "PREFILL")
        stat_d, full_output_text = run_phase(vllm_engine, request_id, "DECODE")

        total_latency_ms = (perf_counter() - inference_start) * 1000

        # Save FULL output to file if requested
        if save_path:
            with open(save_path, 'w') as f:
                f.write(full_output_text)

        return {
            "output": full_output_text,
            "metadata": {
                "formatted_prompt": formatted_prompt,
                "prefill_s": round(stat_p.wall_s, 4),
                "decode_s": round(stat_d.wall_s, 4),
                "prompt_tokens": stat_p.prompt_tokens,
                "gen_tokens": stat_d.gen_tokens
            },
            "latencyMs": round(total_latency_ms, 2)
        }
    finally:
        del vllm_engine
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

def main():
    parser = argparse.ArgumentParser(
        description="vLLM Phase Profiler: Measure Prefill and Decode phases separately.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group_engine = parser.add_argument_group("Engine Configuration")
    group_engine.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                              help="Path or HF repo ID of the model to load")
    group_engine.add_argument("--gpu-util", type=float, default=0.9,
                              help="Fraction of GPU memory to reserve")
    group_engine.add_argument("--enforce-eager", action="store_true",
                              help="Disable CUDA graph capturing")

    group_inference = parser.add_argument_group("Inference Options")
    group_inference.add_argument("--prompt", type=str,
                                 default="Write a 1000 word essay on the Enlightenment movement",
                                 help="Input text to process")
    group_inference.add_argument("--save-output", type=str, metavar="PATH",
                                 help="Save the generated text to a specific file")

    group_output = parser.add_argument_group("Output Control")
    group_output.add_argument("--json", action="store_true",
                              help="Output result as JSON for Promptfoo compatibility")

    args = parser.parse_args()

    res = call_api(args.prompt, {"config": vars(args)}, {})

    if args.json:
        # Truncate 'output' field with ellipses if long for JSON readability
        if len(res['output']) > 200:
            res['output'] = res['output'][:200] + "..."
        sys.stdout.write(json.dumps(res, indent=4) + "\n")
    else:
        print("\n" + "="*45)
        print("INFERENCE DETAILS")
        print("-"*45)
        print(f"Prompt:\n{res['metadata']['formatted_prompt']}")

        print("\n" + "="*45)
        print("PHASE PERFORMANCE METRICS")
        print("-"*45)
        for key in ["prefill_s", "decode_s", "prompt_tokens", "gen_tokens"]:
            print(f"{key:15}: {res['metadata'][key]}")
        print(f"{'total_latency':15}: {res['latencyMs']} ms")
        print("="*45)

        if args.save_output:
            print(f"\nFull output saved to: {args.save_output}")
        else:
            print("\n" + "="*45)
            print("FULL LLM OUTPUT")
            print("-"*45)
            print(res['output'])
            print("="*45)

if __name__ == "__main__":
    main()
