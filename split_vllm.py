"""
vLLM Performance Profiler: Independent Benchmarking of Prefill and Decode Phases.

This script provides a framework to measure the 'Time to First Token' (TTFT) 
and 'Time Per Output Token' (TPOT) by explicitly stepping through the vLLM 
LLMEngine. It allows for granular performance analysis of the prefill 
(prompt processing) and decode (token generation) stages.

Functionality includes:
- Manual engine stepping for phase-level timing.
- Real-time terminal streaming of generated text using standard stdout.
- Clean process termination to avoid vLLM core engine crashes.
- Structured reporting of latency and token throughput metrics.
"""

import gc
import logging
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import List, Tuple, Optional

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.outputs import RequestOutput

# Configure Logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class PhaseStats:
    """Stores performance metrics for a specific execution phase."""
    wall_s: float
    prompt_tokens: int
    gen_tokens: int
    text: str

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
                    sys.stdout.flush()  # Ensure text appears immediately
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

if __name__ == "__main__":
    # Qwen-1.5B provides much better reasoning for theorem descriptions
    vllm_engine = build_engine(model="Qwen/Qwen2.5-1.5B-Instruct")
    
    PROMPT = "Describe the Pythagoras theorem."
    SAMPLING = SamplingParams(max_tokens=256, temperature=0.7)
    REQ_ID = "demo0"
    
    try:
        vllm_engine.add_request(REQ_ID, PROMPT, SAMPLING)
        
        # Execute Phase 1: Prefill (TTFT)
        prefill_results = run_phase(vllm_engine, REQ_ID, "PREFILL")
        
        # Execute Phase 2: Decode (TPOT)
        decode_results = run_phase(vllm_engine, REQ_ID, "DECODE")
        
        # Final Summary Print
        print("\n\n" + "="*40)
        print("BENCHMARKING REPORT")
        print("="*40)
        print(f"Prefill (TTFT): {prefill_results.wall_s:.4f}s | Tokens: {prefill_results.prompt_tokens}")
        print(f"Decode (TPOT):  {decode_results.wall_s:.4f}s | Tokens: {decode_results.gen_tokens}")
        print(f"Total Wall:     {prefill_results.wall_s + decode_results.wall_s:.4f}s")
        print("="*40)
        
    finally:
        # Standard object deletion to avoid engine core proc death
        del vllm_engine
        gc.collect()
        print("\nEngine successfully deallocated.")
        
