# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

# Set environment variables for CPU operation BEFORE importing vllm
os.environ["VLLM_TARGET_DEVICE"] = "cpu"  # Required for vLLM 0.15.0+
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "4"  # Allocates 4GB for KV cache
os.environ["VLLM_CPU_NUM_OF_RESERVED_CPU"] = "1" # Reserves a core for the engine
os.environ["VLLM_USE_TRITON"] = "0" # Disables GPU-focused Triton
os.environ["VLLM_USE_V1"] = "0" # Force V0 usage for CPU

# Optional: Preload TCMalloc if installed for better performance
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    # Force execution on CPU and use recommended data type for it
    llm = LLM(model="facebook/opt-125m",
              dtype="bfloat16")
    
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
