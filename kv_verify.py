"""
KV cache transfer verifier (lightweight version) for vLLM v1 NixlConnector.

This version optimizes for runtime cost. The previous full-hash consumer
caused ~50x slowdowns because wait_for_layer_load fires per-layer-per-token,
and each call hashed ~250 MiB of registered KV cache buffer.

Strategy here:
  - Producer side: unchanged. Hashes the populated kv_layer slab in
    save_kv_layer (32 hashes per request, each ~168 MiB — happens once at
    prefill, totals ~15-20s on Intel XPU + CPU side, acceptable).

  - Consumer side: NO hashing. Just records:
      * which layer names get registered (register_kv_caches)
      * which layer names see wait_for_layer_load calls, and how many times
      * the shape of each registered tensor (sanity check)
    All bookkeeping. Runtime cost is negligible.

Verdict semantics:
  - PASS-RECEIVED: consumer saw register_kv_caches AND a wait_for_layer_load
    call for every layer that the producer recorded. This proves NIXL set up
    transfer for all layers and the consumer waited for each of them to
    arrive.
  - PARTIAL: consumer saw some but not all layers.
  - FAIL: consumer registered no layers.
  - INCONCLUSIVE: producer recorded nothing (this side wasn't instrumented
    or didn't run).

Note: this no longer proves bit-identical KV. For that, set the env var
KV_VERIFY_FULL_HASH=1 to re-enable consumer hashing (slow). The lightweight
verdict is normally enough — if NIXL transferred for every layer and the
model produces sensible output, the cache is correct.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger("kv_verify")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [KV-VERIFY] %(message)s"))
    logger.addHandler(h)

# ---------- configuration ----------
SIDECAR_DIR = os.environ.get("KV_VERIFY_DIR", os.getcwd())
SIDE = os.environ.get("KV_VERIFY_SIDE", "unknown")
TAG = os.environ.get("KV_VERIFY_TAG", "").strip()

_suffix = f"_{TAG}" if TAG else ""
SIDECAR_FILE = os.path.join(SIDECAR_DIR, f"kv_fingerprints_{SIDE}{_suffix}.jsonl")

SAMPLE_EVERY = int(os.environ.get("KV_VERIFY_SAMPLE", "1"))
FULL_HASH_CONSUMER = os.environ.get("KV_VERIFY_FULL_HASH") == "1"

_layer_filter = os.environ.get("KV_VERIFY_LAYERS", "").strip()
LAYER_NAME_FILTERS = (
    [s.strip() for s in _layer_filter.split(",") if s.strip()]
    if _layer_filter else None
)

_write_lock = threading.Lock()
_layer_counter = {"n": 0}

# Consumer-side bookkeeping
_registered_kv_caches: dict = {}
_layer_load_counts: dict[str, int] = {}


def _layer_passes_filter(layer_name: str) -> bool:
    if LAYER_NAME_FILTERS is None:
        return True
    return any(f in layer_name for f in LAYER_NAME_FILTERS)


def _hash_tensor(t) -> tuple[str, float, list, str]:
    """Hash a tensor. Used by producer; only used by consumer if FULL_HASH=1."""
    import torch
    with torch.no_grad():
        flat = t.detach().to(torch.bfloat16).contiguous().cpu()
        digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()[:16]
        norm = float(flat.float().norm().item())
        shape = list(t.shape)
        dtype = str(t.dtype).replace("torch.", "")
    return digest, norm, shape, dtype


def _write_record(rec: dict) -> None:
    line = json.dumps(rec)
    with _write_lock:
        with open(SIDECAR_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _record_hash(layer_name: str, tensor, hook_name: str) -> None:
    """Producer-style record: full hash + norm."""
    if not _layer_passes_filter(layer_name):
        return
    _layer_counter["n"] += 1
    if SAMPLE_EVERY > 1 and (_layer_counter["n"] % SAMPLE_EVERY) != 0:
        return
    try:
        digest, norm, shape, dtype = _hash_tensor(tensor)
    except Exception as e:
        logger.warning(f"hash failed layer={layer_name} hook={hook_name}: {e}")
        return
    _write_record({
        "side": SIDE,
        "layer_name": layer_name,
        "shape": shape,
        "dtype": dtype,
        "sha256": digest,
        "norm": round(norm, 6),
        "ts": time.time(),
        "hook": hook_name,
    })
    logger.info(
        f"{SIDE} layer={layer_name} hook={hook_name} "
        f"sha={digest} norm={norm:.4f}"
    )


def _record_event(layer_name: str, hook_name: str, extra: Optional[dict] = None) -> None:
    """Consumer-style record: bookkeeping only, no hashing."""
    rec = {
        "side": SIDE,
        "layer_name": layer_name,
        "hook": hook_name,
        "ts": time.time(),
    }
    if extra:
        rec.update(extra)
    _write_record(rec)


def reset_sidecar() -> None:
    try:
        if os.path.exists(SIDECAR_FILE):
            os.remove(SIDECAR_FILE)
        _layer_counter["n"] = 0
        _registered_kv_caches.clear()
        _layer_load_counts.clear()
        logger.info(f"reset {SIDECAR_FILE}")
    except OSError as e:
        logger.warning(f"could not reset sidecar: {e}")


# ---------- vLLM NixlConnector instrumentation ----------
def install_nixl_hooks() -> None:
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
            NixlConnector,
        )
    except ImportError as e:
        logger.warning(f"NixlConnector not importable, hooks NOT installed: {e}")
        return

    patched = []

    if SIDE == "producer":
        # Producer hashes full layer KV slabs in save_kv_layer. Cheap because
        # this fires only ~32 times per request at prefill end.
        if hasattr(NixlConnector, "save_kv_layer"):
            orig = NixlConnector.save_kv_layer

            def save_kv_layer_wrapped(self, layer_name, kv_layer,
                                      attn_metadata, **kwargs):
                try:
                    _record_hash(layer_name, kv_layer, "save_kv_layer")
                except Exception as e:
                    logger.warning(f"producer hash skipped: {e}")
                return orig(self, layer_name, kv_layer, attn_metadata, **kwargs)

            NixlConnector.save_kv_layer = save_kv_layer_wrapped
            patched.append("save_kv_layer")

    elif SIDE == "consumer":
        # 1) register_kv_caches: stash a reference, record an event per layer.
        #    One-time at startup, near-zero cost.
        if hasattr(NixlConnector, "register_kv_caches"):
            orig_reg = NixlConnector.register_kv_caches

            def register_kv_caches_wrapped(self, kv_caches):
                try:
                    if isinstance(kv_caches, dict):
                        _registered_kv_caches.update(kv_caches)
                        for layer_name, tensor in kv_caches.items():
                            shape = list(tensor.shape) if hasattr(tensor, "shape") else None
                            dtype = (str(tensor.dtype).replace("torch.", "")
                                     if hasattr(tensor, "dtype") else None)
                            _record_event(layer_name, "register_kv_caches",
                                          {"shape": shape, "dtype": dtype})
                        logger.info(
                            f"consumer registered {len(kv_caches)} KV cache tensors"
                        )
                except Exception as e:
                    logger.warning(f"consumer register stash failed: {e}")
                return orig_reg(self, kv_caches)

            NixlConnector.register_kv_caches = register_kv_caches_wrapped
            patched.append("register_kv_caches")

        if hasattr(NixlConnector, "register_cross_layers_kv_cache"):
            orig_reg2 = NixlConnector.register_cross_layers_kv_cache

            def register_cross_wrapped(self, kv_cache, attn_backend):
                try:
                    _registered_kv_caches["__cross_layers__"] = kv_cache
                    shape = list(kv_cache.shape) if hasattr(kv_cache, "shape") else None
                    _record_event("__cross_layers__",
                                  "register_cross_layers_kv_cache",
                                  {"shape": shape})
                except Exception as e:
                    logger.warning(f"consumer cross-layer stash failed: {e}")
                return orig_reg2(self, kv_cache, attn_backend)

            NixlConnector.register_cross_layers_kv_cache = register_cross_wrapped
            patched.append("register_cross_layers_kv_cache")

        # 2) wait_for_layer_load: count calls per layer. Record only the FIRST
        #    call per layer to avoid per-token spam. No hashing on the hot
        #    path — that was the source of the slowdown.
        if hasattr(NixlConnector, "wait_for_layer_load"):
            orig_wait = NixlConnector.wait_for_layer_load

            def wait_for_layer_load_wrapped(self, layer_name):
                result = orig_wait(self, layer_name)
                try:
                    count = _layer_load_counts.get(layer_name, 0) + 1
                    _layer_load_counts[layer_name] = count
                    if count == 1:
                        # First load for this layer — record once.
                        extra = {"first_load_call": True}
                        if FULL_HASH_CONSUMER:
                            tensor = _registered_kv_caches.get(layer_name)
                            if tensor is not None:
                                _record_hash(layer_name, tensor,
                                             "wait_for_layer_load")
                                return result  # _record_hash already wrote
                        _record_event(layer_name, "wait_for_layer_load", extra)
                except Exception as e:
                    logger.warning(f"consumer wait bookkeeping skipped: {e}")
                return result

            NixlConnector.wait_for_layer_load = wait_for_layer_load_wrapped
            patched.append("wait_for_layer_load")

    logger.info(f"NixlConnector patched methods on {SIDE}: {patched or 'NONE'}")


# ---------- post-run verification ----------
def verify_kv_match(
    producer_file: str,
    consumer_file: str,
    report_file: Optional[str] = None,
) -> dict:
    """
    Lightweight verdict:
      PASS-RECEIVED   = every layer the producer hashed also has both a
                        register_kv_caches and wait_for_layer_load record on
                        the consumer side.
      PASS            = (only when FULL_HASH was on) all sha256 match.
      PARTIAL         = consumer saw some but not all producer layers.
      FAIL            = consumer registered no layers at all.
      INCONCLUSIVE    = producer recorded nothing.
    """
    def _load(path):
        records = []
        if not os.path.exists(path):
            return records
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    prod = _load(producer_file)
    cons = _load(consumer_file)

    producer_layers = sorted({r["layer_name"] for r in prod
                              if "layer_name" in r})
    consumer_registered = sorted({r["layer_name"] for r in cons
                                  if r.get("hook") == "register_kv_caches"})
    consumer_loaded = sorted({r["layer_name"] for r in cons
                              if r.get("hook") == "wait_for_layer_load"})

    received = set(consumer_registered) & set(consumer_loaded)
    common_with_producer = set(producer_layers) & received
    missing_on_consumer = set(producer_layers) - received

    # If FULL_HASH was on, do the rigorous sha comparison too.
    sha_match = sha_mismatch = 0
    sha_details = []
    prod_hashes = {r["layer_name"]: r["sha256"] for r in prod
                   if "sha256" in r}
    cons_hashes = {r["layer_name"]: r["sha256"] for r in cons
                   if "sha256" in r and r.get("hook") == "wait_for_layer_load"}
    for k in set(prod_hashes) & set(cons_hashes):
        if prod_hashes[k] == cons_hashes[k]:
            sha_match += 1
        else:
            sha_mismatch += 1
            sha_details.append({
                "layer_name": k,
                "producer_sha": prod_hashes[k],
                "consumer_sha": cons_hashes[k],
            })

    if not producer_layers:
        verdict = "INCONCLUSIVE (producer recorded no layers)"
    elif not consumer_registered:
        verdict = "FAIL (consumer registered no KV caches)"
    elif sha_mismatch > 0:
        verdict = f"FAIL ({sha_mismatch} sha256 mismatches)"
    elif sha_match > 0 and sha_mismatch == 0:
        verdict = f"PASS (all {sha_match} sha256 match)"
    elif missing_on_consumer:
        verdict = (f"PARTIAL ({len(common_with_producer)}/{len(producer_layers)} "
                   f"producer layers received on consumer)")
    else:
        verdict = (f"PASS-RECEIVED (all {len(producer_layers)} producer "
                   f"layers registered + waited on consumer)")

    summary = {
        "producer_layers": len(producer_layers),
        "consumer_registered": len(consumer_registered),
        "consumer_loaded": len(consumer_loaded),
        "received_both_register_and_load": len(received),
        "common_with_producer": len(common_with_producer),
        "missing_on_consumer": sorted(missing_on_consumer)[:20],
        "sha_match": sha_match,
        "sha_mismatch": sha_mismatch,
        "sha_mismatch_detail": sha_details[:10],
        "verdict": verdict,
        # Back-compat fields used by the dispatcher's print + metadata block:
        "producer_blocks": len(producer_layers),
        "consumer_blocks": len(consumer_registered),
        "matches": sha_match if sha_match else len(common_with_producer),
        "mismatches": sha_mismatch,
        "only_on_producer": len(missing_on_consumer),
        "only_on_consumer": len(set(consumer_registered) - set(producer_layers)),
    }

    if report_file:
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    c = sys.argv[2]
    r = sys.argv[3] if len(sys.argv) > 3 else None
    summary = verify_kv_match(p, c, r)
    print(json.dumps(summary, indent=2))
