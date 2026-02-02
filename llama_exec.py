from io import BytesIO
from pathlib import Path
from typing import Optional

import fire
from termcolor import cprint

from models.llama3.generation import Llama3

import os
import torch


def get_device():
    if "DEVICE" in os.environ:
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    return "cpu"


def run_main(
    ckpt_dir: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    world_size: Optional[int] = None,
    quantization_mode: Optional[str] = None,
):
    generator = Llama3.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        world_size=world_size,
        quantization_mode=quantization_mode,
        device=get_device(),
    )

    # Run the model
    results = generator.completion([prompt], total_len=100)
    
    # Print ONLY the output so promptfoo can capture it
    for token_results in results:
        print(token_results[0].text, end="")
        if token_results[0].finished:
            break

if __name__ == "__main__":
    fire.Fire(run_main)

