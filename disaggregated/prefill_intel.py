import os, sys, json, argparse, warnings, time

os.environ["VLLM_TARGET_DEVICE"] = "xpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ZE_AFFINITY_MASK"] = "0"
os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "5600"
os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"

warnings.filterwarnings("ignore", category=DeprecationWarning)

from vllm import LLM, SamplingParams
try:
    from vllm.tokenizers.get_tokenizer import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer

def run_prefill():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=str, required=True)
    args = parser.parse_args()
    config = json.loads(args.payload)

    print(f"--- Intel B60 Prefiller Initializing ({config['model']}) ---")
    
    engine_kwargs = {
        "model": config['model'],
        "gpu_memory_utilization": config['gpu_util'],
        "enforce_eager": config.get('enforce_eager', False),
        "kv_transfer_config": {
            "kv_connector": "NixlConnector", "kv_role": "kv_producer",
            "kv_buffer_device": "cpu", "kv_connector_extra_config": {"backends": ["UCX"], "num_threads": 4}
        }
    }
    if config.get('max_model_len'): engine_kwargs["max_model_len"] = config['max_model_len']
    
    llm = LLM(**engine_kwargs)

    tokenizer = get_tokenizer(config['model'])
    formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": config['prompt']}], tokenize=False, add_generation_prompt=True)
    
    print("Executing Prefill on Intel B60...")
    prefill_start = time.time()
    llm.generate(formatted_prompt, SamplingParams(max_tokens=1, temperature=0))
    prefill_end = time.time()
    
    with open("prefill_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"prefill_start": prefill_start, "prefill_end": prefill_end}, f)
        
    print("Prefill complete. KV Cache shipped.")

if __name__ == "__main__":
    run_prefill()
