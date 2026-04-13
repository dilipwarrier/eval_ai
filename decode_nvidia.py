import os, sys, json, time, argparse, warnings, asyncio

os.environ["VLLM_TARGET_DEVICE"] = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "5600"
os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"

warnings.filterwarnings("ignore", category=DeprecationWarning)

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams

try:
    from vllm.tokenizers.get_tokenizer import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer

async def run_decode():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    args = parser.parse_args()
    config = json.loads(args.payload)

    print(f"--- NVIDIA RTX 5060 Ti Decoder Initializing ({config['model']}) ---")
    
    init_start = time.time()
    
    # Configure the Async Engine (Removed invalid logging parameter)
    engine_kwargs = {
        "model": config['model'],
        "gpu_memory_utilization": config['gpu_util'],
        "enforce_eager": config.get('enforce_eager', False),
        "kv_transfer_config": {
            "kv_connector": "NixlConnector", "kv_role": "kv_consumer",
            "kv_buffer_device": "cpu", "kv_connector_extra_config": {"backends": ["UCX"], "num_threads": 4}
        }
    }
    if config.get('max_model_len'): engine_kwargs["max_model_len"] = config['max_model_len']
    
    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    init_time = time.time() - init_start

    tokenizer = get_tokenizer(config['model'])
    formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": config['prompt']}], tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.7)
    
    print("Waiting for KV Cache from Intel Node...")
    
    # Stream the generation to catch the first token timestamp
    request_id = "disagg_req_1"
    results_generator = engine.generate(formatted_prompt, sampling_params, request_id)
    
    first_token_time = None
    final_output = None
    
    async for request_output in results_generator:
        # The exact moment the first token arrives
        if first_token_time is None and request_output.outputs[0].token_ids:
            first_token_time = time.time()
        final_output = request_output
        
    decode_end = time.time()

    # Read Intel Metrics to calculate the gap
    prefill_s, transfer_s, decode_s = 0.0, 0.0, 0.0
    try:
        with open("prefill_metrics.json", "r") as f:
            p_metrics = json.load(f)
            prefill_s = p_metrics["prefill_end"] - p_metrics["prefill_start"]
            
            # Transfer Time = First Token - Intel Finish Time
            if first_token_time:
                transfer_s = first_token_time - p_metrics["prefill_end"]
                decode_s = decode_end - first_token_time
            else:
                decode_s = decode_end - p_metrics["prefill_end"]
    except Exception as e:
        print(f"Could not load precise metrics: {e}")
        decode_s = decode_end - init_start
    
    gen_tokens = len(final_output.outputs[0].token_ids)
    prompt_tokens = len(final_output.prompt_token_ids)
    if transfer_s < 0: transfer_s = 0.0001 # Sync guard

    result = {
        "output": final_output.outputs[0].text,
        "metadata": {
            "formatted_prompt": formatted_prompt,
            "init_s": round(init_time, 4), "init_input_tokens": 0, "init_output_tokens": 0,
            "prefill_s": round(prefill_s, 4), "prefill_input_tokens": prompt_tokens, "prefill_output_tokens": 1,
            "transfer_s": round(transfer_s, 4), "transfer_input_tokens": 0, "transfer_output_tokens": 0,
            "decode_s": round(decode_s, 4), "decode_input_tokens": 0, "decode_output_tokens": gen_tokens,
            "total_s": round(init_time + prefill_s + transfer_s + decode_s, 4), 
            "total_input_tokens": prompt_tokens, "total_output_tokens": gen_tokens,
        },
        "latencyMs": round((init_time + prefill_s + transfer_s + decode_s) * 1000, 2)
    }

    with open(args.outpath, "w", encoding="utf-8") as f:
        json.dump(result, f)

if __name__ == "__main__":
    asyncio.run(run_decode())
