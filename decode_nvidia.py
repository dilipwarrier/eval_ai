import os, sys, json, time, argparse, warnings, asyncio

os.environ["VLLM_TARGET_DEVICE"] = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = "5600"
os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"

# --- KV transfer verification: must be set BEFORE importing kv_verify ---
os.environ["KV_VERIFY_SIDE"] = "consumer"
os.environ["KV_VERIFY_DIR"] = os.path.dirname(os.path.abspath(__file__))
# Optional knobs (must MATCH the producer side or comparison will be partial):
# os.environ["KV_VERIFY_SAMPLE"] = "8"
# os.environ["KV_VERIFY_LAYERS"] = "0,15,31"
# --- end KV verify env ---

warnings.filterwarnings("ignore", category=DeprecationWarning)

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams

try:
    from vllm.tokenizers.get_tokenizer import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer

# --- KV transfer verification: install hooks after vLLM is importable ---
import kv_verify
kv_verify.reset_sidecar()
kv_verify.install_nixl_hooks()
# --- end KV verify hooks ---


# ============================================================
# Engine introspection helpers (robust across vLLM versions)
# ============================================================
def _resolve_cache_config(engine):
    """Walk known attribute paths to find vLLM's CacheConfig."""
    candidates = [
        ("vllm_config", "cache_config"),                  # AsyncLLM v0.15.0+
        ("engine_core", "vllm_config", "cache_config"),
        ("llm_engine", "cache_config"),                   # older vLLM
        ("llm_engine", "vllm_config", "cache_config"),
        ("engine_core", "cache_config"),
        ("engine_core", "engine_core", "cache_config"),
    ]
    for path in candidates:
        obj = engine
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if hasattr(obj, "num_gpu_blocks") and hasattr(obj, "block_size"):
                return obj, "engine." + ".".join(path)
        except AttributeError:
            continue
    return None, None


def _resolve_model_config(engine):
    """Find vLLM's ModelConfig."""
    candidates = [
        ("model_config",),                                # AsyncLLM v0.15.0+
        ("vllm_config", "model_config"),
        ("engine_core", "vllm_config", "model_config"),
        ("llm_engine", "model_config"),
    ]
    for path in candidates:
        obj = engine
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def _resolve_parallel_config(engine):
    """Find vLLM's ParallelConfig."""
    candidates = [
        ("vllm_config", "parallel_config"),
        ("engine_core", "vllm_config", "parallel_config"),
        ("llm_engine", "parallel_config"),
    ]
    for path in candidates:
        obj = engine
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def _compute_kv_cache_gib(num_gpu_blocks, block_size, model_cfg, parallel_cfg):
    """
    Compute KV cache size in GiB from model geometry.
    Returns (gib, geometry_dict) or (-1, None) on failure.
    """
    if num_gpu_blocks <= 0 or model_cfg is None or parallel_cfg is None:
        return -1, None
    try:
        num_kv_heads = model_cfg.get_num_kv_heads(parallel_cfg)
        head_dim = model_cfg.get_head_size()
        num_layers = model_cfg.get_num_layers(parallel_cfg)
        dtype_size = model_cfg.dtype.itemsize
        kv_cache_bytes = (
            num_gpu_blocks * block_size * num_layers * 2  # 2 = K + V
            * num_kv_heads * head_dim * dtype_size
        )
        gib = round(kv_cache_bytes / (1024 ** 3), 2)
        return gib, {
            "layers": num_layers,
            "kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype_size_bytes": dtype_size,
        }
    except Exception as e:
        print(f"Could not compute precise KV cache GiB: {e}")
        return -1, None


async def run_decode():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    args = parser.parse_args()
    config = json.loads(args.payload)

    print(f"--- NVIDIA RTX 5060 Ti Decoder Initializing ({config['model']}) ---")

    init_start = time.time()

    # Configure the Async Engine
    engine_kwargs = {
        "model": config['model'],
        "gpu_memory_utilization": config['gpu_util'],
        "enforce_eager": config.get('enforce_eager', False),
        "kv_transfer_config": {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_consumer",
            "kv_buffer_device": "cpu",
            "kv_connector_extra_config": {"backends": ["UCX"], "num_threads": 4},
        },
    }
    if config.get('max_model_len'):
        engine_kwargs["max_model_len"] = config['max_model_len']

    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    init_time = time.time() - init_start

    # ============================================================
    # Resolve KV cache size from the engine
    # ============================================================
    cache_cfg, cache_path = _resolve_cache_config(engine)
    model_cfg = _resolve_model_config(engine)
    parallel_cfg = _resolve_parallel_config(engine)

    if cache_cfg is not None:
        num_gpu_blocks = cache_cfg.num_gpu_blocks or -1
        block_size = cache_cfg.block_size or -1
        print(f"KV Cache: {num_gpu_blocks} blocks "
              f"(block_size={block_size}, source={cache_path})")
    else:
        num_gpu_blocks = -1
        block_size = -1
        print("KV Cache: could not locate cache_config on engine")

    kv_cache_gib, geometry = _compute_kv_cache_gib(
        num_gpu_blocks, block_size, model_cfg, parallel_cfg
    )
    if geometry is not None:
        print(f"KV Cache geometry: layers={geometry['layers']} "
              f"kv_heads={geometry['kv_heads']} "
              f"head_dim={geometry['head_dim']} "
              f"dtype_size={geometry['dtype_size_bytes']}B "
              f"-> {kv_cache_gib} GiB")
    else:
        print(f"KV Cache GiB: {kv_cache_gib} (geometry unavailable)")

    # ============================================================
    # Run the decode
    # ============================================================
    tokenizer = get_tokenizer(config['model'])
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": config['prompt']}],
        tokenize=False,
        add_generation_prompt=True,
    )
    sampling_params = SamplingParams(max_tokens=2048, temperature=0.7)

    print("Waiting for KV Cache from Intel Node...")

    request_id = "disagg_req_1"
    results_generator = engine.generate(formatted_prompt, sampling_params, request_id)

    first_token_time = None
    final_output = None

    async for request_output in results_generator:
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
    if transfer_s < 0:
        transfer_s = 0.0001  # Sync guard

    result = {
        "output": final_output.outputs[0].text,
        "metadata": {
            "formatted_prompt": formatted_prompt,
            "init_s": round(init_time, 4),
            "init_input_tokens": 0, "init_output_tokens": 0,
            "prefill_s": round(prefill_s, 4),
            "prefill_input_tokens": prompt_tokens, "prefill_output_tokens": 1,
            "transfer_s": round(transfer_s, 4),
            "transfer_input_tokens": 0, "transfer_output_tokens": 0,
            "decode_s": round(decode_s, 4),
            "decode_input_tokens": 0, "decode_output_tokens": gen_tokens,
            "total_s": round(init_time + prefill_s + transfer_s + decode_s, 4),
            "total_input_tokens": prompt_tokens, "total_output_tokens": gen_tokens,
            "kv_cache_gib": kv_cache_gib,
            "kv_cache_blocks": num_gpu_blocks,
            "kv_cache_block_size": block_size,
            "kv_cache_geometry": geometry,
            "kv_cache_source": cache_path,
        },
        "latencyMs": round((init_time + prefill_s + transfer_s + decode_s) * 1000, 2),
    }

    with open(args.outpath, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    asyncio.run(run_decode())
