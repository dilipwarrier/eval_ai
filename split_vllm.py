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
from typing import List, Optional, Tuple, Dict, Any

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

        vllm_engine.add_request(request_id, formatted_prompt, sampling_params)

        # --- PHASE 2: PREFILL ---
        # Note: We don't include prefill setup in the timing, just the execution
        stat_p, _ = run_phase(vllm_engine, request_id, "PREFILL")

        # --- PHASE 3: DECODE ---
        stat_d, full_output_text = run_phase(vllm_engine, request_id, "DECODE")

        # Calculate Totals
        # Note: stat_p and stat_d have their own wall_s.
        # Total latency is Init + Prefill + Decode (approximate)
        total_time_s = init_duration + stat_p.wall_s + stat_d.wall_s
        total_input_tokens = stat_p.prompt_tokens
        total_output_tokens = stat_d.gen_tokens

        if save_path:
            with open(save_path, 'w') as f:
                f.write(full_output_text)

        # Construct Promptfoo response
        return {
            "output": full_output_text,
            "metadata": {
                "formatted_prompt": formatted_prompt,

                # Detailed Breakdown for Table
                "init_s": round(init_duration, 4),
                "init_input_tokens": 0,
                "init_output_tokens": 0,

                "prefill_s": round(stat_p.wall_s, 4),
                "prefill_input_tokens": stat_p.prompt_tokens,
                "prefill_output_tokens": 1, # By definition, prefill stops after 1 token

                "decode_s": round(stat_d.wall_s, 4),
                "decode_input_tokens": 0,
                "decode_output_tokens": stat_d.gen_tokens,

                "total_s": round(total_time_s, 4),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
            },
            # Promptfoo uses this field for its main latency metric
            "latencyMs": round(total_time_s * 1000, 2)
        }
    finally:
        del vllm_engine
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

def print_metrics_table(meta: Dict[str, Any]):
    """Prints a formatted table of metrics."""
    # Header
    print(f"{'PHASE':<15} | {'TIME (s)':<10} | {'INPUT TOKENS':<12} | {'OUTPUT TOKENS':<13}")
    print("-" * 58)

    # Rows
    rows = [
        ("Initialization", meta["init_s"], meta["init_input_tokens"], meta["init_output_tokens"]),
        ("Prefill", meta["prefill_s"], meta["prefill_input_tokens"], meta["prefill_output_tokens"]),
        ("Decode", meta["decode_s"], meta["decode_input_tokens"], meta["decode_output_tokens"]),
        ("Total", meta["total_s"], meta["total_input_tokens"], meta["total_output_tokens"]),
    ]

    for name, time_s, in_tok, out_tok in rows:
        print(f"{name:<15} | {time_s:<10.4f} | {in_tok:<12} | {out_tok:<13}")
    print("=" * 58)

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
        if len(res['output']) > 200:
            res['output'] = res['output'][:200] + "..."
        sys.stdout.write(json.dumps(res, indent=4) + "\n")
    else:
        print("\n" + "="*58)
        print("FORMATTED PROMPT")
        print("-"*58)
        print(f"Prompt:\n{res['metadata']['formatted_prompt']}")

        print("\n" + "="*58)
        print("PHASE PERFORMANCE METRICS")
        print("-"*58)

        print_metrics_table(res["metadata"])

        if args.save_output:
            print(f"\nFull output saved to: {args.save_output}")
        else:
            print("\n" + "="*58)
            print("FULL LLM OUTPUT")
            print("-"*58)
            print(res['output'])
            print("="*58)

if __name__ == "__main__":
    main()
