#!/usr/bin/env python
"""
Ollama Performance Benchmark Utility.

This script measures the arithmetic efficiency of LLMs on local hardware.
It calculates statistics (mean, sigma) over multiple attempts to identify
performance consistency across different systems.
"""

import argparse
import random
import statistics
import string
import ollama

# A sample prompt of roughly 500 tokens (repeating a text block)
BASE_TEXT = "The quick brown fox jumps over the lazy dog. " * 55


def run_benchmark(model: str):
    """
    Perform a single inference pass and return timing metrics.

    Args:
        model (str): The name of the Ollama model to evaluate.

    Returns:
        dict: A dictionary containing load, prefill, decode, and total times.
    """
    # Generate a random 8-character string to break the KV cache
    nonce = ''.join(random.choices(string.ascii_letters, k=8))
    prompt = f"[{nonce}] Summarize the following text: {BASE_TEXT}"

    # Execute inference with keep_alive=0 to prevent background residency
    response = ollama.generate(
        model=model,
        prompt=prompt,
        stream=False,
        keep_alive=0,
    )

    # Convert nanoseconds to seconds for all durations
    l_dur = response.get('load_duration', 0)
    p_dur = response['prompt_eval_duration']
    d_dur = response['eval_duration']
    t_dur = response['total_duration']

    return {
        "load": l_dur / 1e9,
        "prefill": p_dur / 1e9,
        "decode": d_dur / 1e9,
        "total": t_dur / 1e9,
        "prefill_tps": (response['prompt_eval_count'] / p_dur) * 1e9,
        "decode_tps": (response['eval_count'] / d_dur) * 1e9
    }


def main():
    """
    Parse arguments, execute the benchmark loop, and print statistics.
    """
    parser = argparse.ArgumentParser(description="Benchmark Ollama models.")
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="llama3:latest",
        help="The name of the Ollama model to benchmark."
    )
    parser.add_argument(
        "-n", "--num_attempts",
        type=int,
        default=1,
        help="Number of times to run the benchmark."
    )
    args = parser.parse_args()

    results = {
        "Load time": [],
        "Prefill time": [],
        "Decode time": [],
        "Total time": []
    }

    print(f"Benchmarking {args.model} for {args.num_attempts} attempts...\n")

    for i in range(args.num_attempts):
        metrics = run_benchmark(args.model)
        results["Load time"].append(metrics['load'])
        results["Prefill time"].append(metrics['prefill'])
        results["Decode time"].append(metrics['decode'])
        results["Total time"].append(metrics['total'])

        print(
            f"Attempt {i+1}: "
            f"Prefill {metrics['prefill_tps']:.1f} tokens/s | "
            f"Decode {metrics['decode_tps']:.1f} tokens/s"
        )

    def get_stats(data):
        mean_val = statistics.mean(data)
        sigma_val = statistics.stdev(data) if len(data) > 1 else 0.0
        return mean_val, sigma_val

    # Table formatting
    header = f"{'Times':<15} | {'Mean (s)':>10} | {'Sigma (s)':>10}"
    sep = "-" * len(header)

    print(f"\nSummary statistics for {args.model}")
    print(sep)
    print(header)
    print(sep)

    # Print the specific components
    for label in ["Load time", "Prefill time", "Decode time"]:
        mean, sigma = get_stats(results[label])
        print(f"{label:<15} | {mean:>10.1f} | {sigma:>10.1f}")

    # Separator line above the total
    print(sep)

    # Print the total
    t_mean, t_sigma = get_stats(results["Total time"])
    print(f"{'Total time':<15} | {t_mean:>10.1f} | {t_sigma:>10.1f}")
    print(sep)


if __name__ == "__main__":
    main()
