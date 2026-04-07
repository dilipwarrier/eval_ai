import os, sys, warnings, argparse, json, logging, subprocess, time
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Any
import requests
from openai import OpenAI

INTEL_PYTHON = "/home/lyptusadmin/vllm-xpu-env/bin/python" 
NVIDIA_PYTHON = "/home/user4/gemini_pro_agent/nvidia-agent-env/bin/python"
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
XPU_ISOLATED_SOURCE = os.path.join(WORKING_DIR, "vllm_xpu_isolated")
SHARED_RESULT_FILE = os.path.join(WORKING_DIR, "disagg_results.json")

os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["VLLM_REPORT_STATS"] = "0"

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s", force=True)

class RemoteVLLMClient:
    def __init__(self, ip: str, port: int, model: str):
        self.ip, self.port, self.model = ip, port, model
        self.base_url = f"http://{self.ip}:{self.port}/v1"
        self.api_key = "llm-d-local"

    def test_connection(self) -> None:
        try:
            requests.get(f"{self.base_url}/models", timeout=2.0).raise_for_status()
        except Exception as err:
            logging.error("Connection failed: %s", err)
            sys.exit(1)

    def profile(self, prompt: str) -> dict:
        init_start = perf_counter()
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        init_duration = perf_counter() - init_start
        start_time = perf_counter()
        first_token_time, full_text, prompt_tok, gen_tok = None, "", 0, 0

        try:
            stream = client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                stream=True, stream_options={"include_usage": True}
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta.content else ""
                if first_token_time is None and delta: first_token_time = perf_counter()
                if delta: full_text += delta
                if chunk.usage:
                    prompt_tok, gen_tok = chunk.usage.prompt_tokens, chunk.usage.completion_tokens
        except Exception as err:
            logging.error("Remote inference error: %s", err)
            sys.exit(1)

        end_time = perf_counter()
        first_token_time = first_token_time or end_time

        return {
            "output": full_text,
            "metadata": {
                "formatted_prompt": prompt, "init_s": round(init_duration, 4),
                "init_input_tokens": 0, "init_output_tokens": 0,
                "prefill_s": round(first_token_time - start_time, 4),
                "prefill_input_tokens": prompt_tok, "prefill_output_tokens": 1,
                "transfer_s": 0.0, "transfer_input_tokens": 0, "transfer_output_tokens": 0,
                "decode_s": round(end_time - first_token_time, 4),
                "decode_input_tokens": 0, "decode_output_tokens": gen_tok,
                "total_s": round(end_time - start_time, 4),
                "total_input_tokens": prompt_tok, "total_output_tokens": gen_tok,
            },
            "latencyMs": round((end_time - start_time) * 1000, 2)
        }

def call_disaggregated(prompt: str, options: dict) -> dict:
    config = options.get('config', {})
    model_name = config.get('model', "Qwen/Qwen2.5-1.5B-Instruct")
    
    if config.get('remote', False):
        client = RemoteVLLMClient(config.get('ip', 'localhost'), config.get('port', 8080), model_name)
        client.test_connection()
        return client.profile(prompt)

    payload = config.copy()
    payload["prompt"] = prompt
    payload_str = json.dumps(payload)

    if os.path.exists(SHARED_RESULT_FILE): os.remove(SHARED_RESULT_FILE)

    print("--- Starting Disaggregated Serving Cluster ---")
    print(f"Working Directory locked to: {WORKING_DIR}")
    
    env_nv = os.environ.copy()
    env_nv["VLLM_TARGET_DEVICE"] = "cuda"
    env_nv["CUDA_VISIBLE_DEVICES"] = "0"
    
    env_intel = os.environ.copy()
    env_intel["VLLM_TARGET_DEVICE"] = "xpu"
    env_intel["ZE_AFFINITY_MASK"] = "0"
    env_intel["CUDA_VISIBLE_DEVICES"] = "" 
    env_intel["PYTHONPATH"] = f"{XPU_ISOLATED_SOURCE}:{env_intel.get('PYTHONPATH', '')}"

    print("[1/2] Launching NVIDIA Decode Node...")
    decode_proc = subprocess.Popen(
        [NVIDIA_PYTHON, "decode_nvidia.py", "--payload", payload_str, "--outpath", SHARED_RESULT_FILE], 
        env=env_nv, cwd=WORKING_DIR
    )
    
    time.sleep(100) 
    
    print("[2/2] Launching Intel B60 Prefill Node...")
    prefill_proc = subprocess.Popen(
        [INTEL_PYTHON, "prefill_intel.py", "--payload", payload_str], 
        env=env_intel, cwd=WORKING_DIR
    )

    try:
        prefill_proc.wait()
        decode_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping cluster...")
        prefill_proc.terminate()
        decode_proc.terminate()
        sys.exit(1)

    if not os.path.exists(SHARED_RESULT_FILE):
        logging.error("Disaggregated pipeline failed. No result file generated.")
        sys.exit(1)

    with open(SHARED_RESULT_FILE, "r", encoding="utf-8") as f:
        res = json.load(f)

    if config.get('save_output'):
        with open(config['save_output'], 'w', encoding='utf-8') as f:
            f.write(res["output"])

    return res

class ProfilerCLI:
    @staticmethod
    def print_table(meta: Dict[str, Any]) -> None:
        print(f"{'PHASE':<15} | {'TIME (s)':<10} | {'INPUT TOKENS':<12} | {'OUTPUT TOKENS':<13}\n" + "-" * 58)
        for name, time_s, in_tok, out_tok in [
            ("Initialization", meta["init_s"], meta["init_input_tokens"], meta["init_output_tokens"]),
            ("Prefill", meta["prefill_s"], meta["prefill_input_tokens"], meta["prefill_output_tokens"]),
            ("KV Transfer", meta.get("transfer_s", 0.0), meta.get("transfer_input_tokens", 0), meta.get("transfer_output_tokens", 0)),
            ("Decode", meta["decode_s"], meta["decode_input_tokens"], meta["decode_output_tokens"]),
            ("Total", meta["total_s"], meta["total_input_tokens"], meta["total_output_tokens"]),
        ]:
            if name == "Total": print("-" * 58)
            print(f"{name:<15} | {time_s:<10.4f} | {in_tok:<12} | {out_tok:<13}")
        print("=" * 58)

    @classmethod
    def run(cls) -> None:
        parser = argparse.ArgumentParser(description="vLLM NIXL Profiler", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        group_eng = parser.add_argument_group("Engine Configuration")
        group_eng.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
        group_eng.add_argument("--gpu-util", type=float, default=0.9)
        group_eng.add_argument("--enforce-eager", action="store_true")
        group_eng.add_argument("--max-model-len", type=int, default=None)

        group_inf = parser.add_argument_group("Inference Options")
        group_inf.add_argument("--prompt", type=str, default="Write a 1000 word essay on the Enlightenment movement")
        group_inf.add_argument("--prompt-file", type=str, metavar="PATH")
        group_inf.add_argument("--save-output", type=str, metavar="PATH")

        group_rem = parser.add_argument_group("Remote")
        group_rem.add_argument("--remote", action="store_true")
        group_rem.add_argument("--ip", type=str, default="localhost")
        group_rem.add_argument("--port", type=int, default=8080)
        parser.add_argument("--json", action="store_true")

        args = parser.parse_args()

        if args.prompt_file:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                args.prompt = f"{args.prompt}\n\n{f.read()}"

        res = call_disaggregated(args.prompt, {"config": vars(args)})

        if args.json:
            sys.stdout.write(json.dumps(res, indent=4) + "\n")
        else:
            print(f"\n{'=' * 58}\nFORMATTED PROMPT\n{'-' * 58}\n{res['metadata']['formatted_prompt']}\n{'=' * 58}")
            print(f"PHASE PERFORMANCE METRICS\n{'-' * 58}")
            cls.print_table(res["metadata"])
            if not args.save_output: print(f"\n{'=' * 58}\nFULL LLM OUTPUT\n{'-' * 58}\n{res['output']}\n{'=' * 58}")

if __name__ == "__main__":
    ProfilerCLI.run()
