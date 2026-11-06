"""Offline token-row traces and locality diagnostics, outside the decode fast path."""

from hashlib import sha256
from math import ceil
from statistics import mean

import torch

from .config import CacheConfig
from .hybrid import HotConfig, HybridKVCache
from .selection import Selection, selection_oracle


def token_topk(query, keys, count):
    """Exact per-head QK ranking; reads ALL keys, so this is an offline oracle."""
    if count < 1:
        raise ValueError("token count must be positive")
    groups = query.shape[0] // keys.shape[1]
    scores = torch.einsum("hd,thd->ht", query.float(), keys.repeat_interleave(groups, 1).float())
    return scores.topk(min(count, len(keys)), dim=-1).indices.tolist()


def synthetic_trace(pattern, steps=8, seed=42):
    """Controlled access patterns; independent CPU RNG makes input portable."""
    if pattern not in ("reuse", "alternating", "streaming", "packed"):
        raise ValueError("unknown access pattern")
    if not 2 <= steps <= 8:
        raise ValueError("synthetic traces support 2..8 steps")
    rng = torch.Generator().manual_seed(seed)

    def kv(count):
        return torch.randn(count, 1, 8, generator=rng)

    trace = {
        "schema_version": 1,
        "source": {"kind": "synthetic", "pattern": pattern, "seed": seed},
        "page_size": 4,
        "prefill": {"k": kv(64), "v": kv(64)},
        "events": [],
    }
    for step in range(steps):
        if pattern == "packed":
            rows = [0, 1, 2, 3]
        elif pattern == "streaming":
            rows = [step + base for base in (0, 16, 32, 48)]
        else:
            offset = step % 2 if pattern == "alternating" else 0
            rows = [offset + base for base in (0, 4, 8, 12)]
        # Two query heads overlap on one token; union contains exactly four rows.
        positions = [rows[:3], [rows[1], rows[3]]]
        trace["events"].append(
            {"q": torch.randn(2, 8, generator=rng), "k": kv(1), "v": kv(1), "positions": positions}
        )
    return trace


def validate_trace(trace):
    if trace.get("schema_version") != 1 or not isinstance(trace.get("source"), dict):
        raise ValueError("trace requires schema_version=1 and source metadata")
    if not trace.get("events"):
        raise ValueError("trace needs at least one event")
    k, v = trace["prefill"]["k"], trace["prefill"]["v"]
    if k.ndim != 3 or k.shape != v.shape or min(k.shape) < 1 or k.dtype != v.dtype:
        raise ValueError("invalid prefill KV")
    if k.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("unsupported KV dtype")
    length, maximum = len(k), 0
    for event in trace["events"]:
        q, new_k, new_v = (event[name] for name in ("q", "k", "v"))
        if (
            new_k.shape != (1, *k.shape[1:])
            or new_v.shape != new_k.shape
            or new_k.dtype != k.dtype
            or new_v.dtype != k.dtype
            or q.ndim != 2
            or q.shape[1] != k.shape[2]
            or q.shape[0] < 1
            or q.shape[0] % k.shape[1]
            or q.dtype != k.dtype
        ):
            raise ValueError("incompatible event Q/K/V")
        length += 1
        selection = Selection.from_tokens(
            event["positions"], length, trace["page_size"], q.shape[0]
        )
        maximum = max(maximum, len({t for row in selection.positions for t in row}))
    return maximum


def save_trace(trace, path):
    validate_trace(trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, path)


def load_trace(path):
    trace = torch.load(path, map_location="cpu", weights_only=True)
    validate_trace(trace)
    return trace


def _jaccard(current, previous):
    return len(current & previous) / len(current | previous) if previous is not None else None


@torch.inference_mode()
def replay_trace(trace, hot_rows=None, device="cpu"):
    """Replay actual gathers and attention; synchronize for diagnostics, not timing."""
    union_bound = validate_trace(trace)
    page_size = trace["page_size"]
    k, v = [trace["prefill"][name].to(device) for name in ("k", "v")]
    maximum = len(k) + len(trace["events"])
    hot = HotConfig(union_bound, union_bound if hot_rows is None else hot_rows)
    config = CacheConfig(
        gpu_pages=hot.admission_blocks(page_size),
        host_pages=ceil(maximum / page_size),
        page_size=page_size,
        kv_heads=k.shape[1],
        head_dim=k.shape[2],
        dtype=k.dtype,
        device=device,
    )
    cache = HybridKVCache(config, hot)
    records, previous_rows, previous_pages, last_seen, gaps = [], None, None, {}, []
    row_bytes = 2 * config.kv_heads * config.head_dim * k.element_size()
    try:
        cache.reserve("trace", maximum, 1)
        cache.write("trace", 0, k, v)
        cache.storage.synchronize()
        # Start with cold hot contents; retain only the two protected tail pages.
        cache.reclaim_history("trace", 0, evict=True)
        cache.reset_stats()
        for step, event in enumerate(trace["events"]):
            q, new_k, new_v = [event[name].to(device) for name in ("q", "k", "v")]
            cache.write("trace", 0, new_k, new_v)
            k, v = torch.cat((k, new_k)), torch.cat((v, new_v))
            selection = Selection.from_tokens(event["positions"], len(k), page_size, q.shape[0])
            rows = {t for head in selection.positions for t in head}
            pages = {t // page_size for t in rows}
            before = cache.row_stats.clone()
            actual = cache.attention(q, "trace", 0, selection)
            expected, _ = selection_oracle(q, k, v, selection)
            tolerance = 0.008 if k.dtype != torch.float32 else 3e-5
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            resident, hits, misses = (cache.row_stats - before).cpu().tolist()
            assert resident + hits + misses == len(rows)
            for token in rows:
                if token in last_seen:
                    gaps.append(step - last_seen[token])
                last_seen[token] = step
            # Convert actual hot tags back to logical token IDs for readable traces.
            reverse = {p.host_slot: p.key[2] for p in cache.storage.pages.values()}
            slots = [
                None if tag < 0 else reverse[tag // page_size] * page_size + tag % page_size
                for tag in cache.tables["trace", 0].tags.cpu().tolist()
            ]
            records.append(
                {
                    "step": step,
                    "length": len(k),
                    "positions": selection.positions,
                    "union_rows": sorted(rows),
                    "selected_pages": sorted(pages),
                    "row_jaccard": _jaccard(rows, previous_rows),
                    "page_jaccard": _jaccard(pages, previous_pages),
                    "page_density": len(rows)
                    / sum(min(page_size, len(k) - p * page_size) for p in pages),
                    "resident_hits": resident,
                    "hot_hits": hits,
                    "misses": misses,
                    "h2d_bytes": misses * row_bytes,
                    "hot_slots": slots,
                    "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
                }
            )
            previous_rows, previous_pages = rows, pages
            cache.assert_invariants()
        totals = cache.snapshot()["row_stats"]
        nonresident = totals["hot_hits"] + totals["misses"]
        summary = {
            **totals,
            "hot_hit_rate": totals["hot_hits"] / nonresident if nonresident else None,
            "unique_rows": len(last_seen),
            "mean_reuse_gap_steps": mean(gaps) if gaps else None,
            "mean_row_jaccard": mean(r["row_jaccard"] for r in records[1:])
            if len(records) > 1
            else None,
            "mean_page_jaccard": mean(r["page_jaccard"] for r in records[1:])
            if len(records) > 1
            else None,
            "mean_page_density": mean(r["page_density"] for r in records),
            "max_abs_error": max(r["max_abs_error"] for r in records),
        }
        return {
            "source": trace["source"],
            "device": str(cache.device),
            "dtype": str(k.dtype),
            "page_size": page_size,
            "row_bytes": row_bytes,
            "max_union_rows": union_bound,
            "hot_rows": hot.hot_rows,
            "gpu_blocks": config.gpu_pages,
            "initial_residency": "history evicted; two tail pages retained; hot empty",
            "summary": summary,
            "events": records,
        }
    finally:
        if "trace" in cache.storage.requests:
            cache.release_request("trace")
        cache.close()


@torch.inference_mode()
def capture_trace(model, tokens, steps=8, token_budget=8, layer=0, page_size=16, source=None):
    """Capture one layer of a dense teacher; token oracle never drives generation."""
    cfg = model.config
    if not tokens or steps < 1 or token_budget < 1 or not 0 <= layer < cfg.num_hidden_layers:
        raise ValueError("invalid capture prompt, steps, token budget or layer")
    trace = {
        "schema_version": 1,
        "page_size": page_size,
        "prefill": {},
        "events": [],
        "source": {
            **(source or {}),
            "kind": "dense_teacher_token_oracle",
            "layer": layer,
            "token_budget_per_head": token_budget,
            "prompt_tokens": len(tokens),
            "prompt_sha256": sha256(str(tokens).encode()).hexdigest(),
            "torch": str(torch.__version__),
            "capture_device": str(model.device),
            "capture_dtype": str(model.dtype),
        },
    }

    class RecordingCache(HybridKVCache):
        def write(self, request_id, layer_id, k, v):
            super().write(request_id, layer_id, k, v)
            if layer_id == layer:
                self.latest = {"k": k.detach().cpu().clone(), "v": v.detach().cpu().clone()}
                if not trace["prefill"]:
                    trace["prefill"] = self.latest

        def attention(self, query, request_id, layer_id, selection, backend="auto"):
            if layer_id == layer:
                keys, _ = self.storage.host_kv(request_id, layer_id)
                q = query.detach().cpu()
                trace["events"].append(
                    {**self.latest, "q": q.clone(), "positions": token_topk(q, keys, token_budget)}
                )
            return super().attention(query, request_id, layer_id, selection, backend)

    maximum = len(tokens) + steps
    hot = HotConfig(
        maximum,
        maximum,
        resident_window_pages=ceil(maximum / page_size) + 1,
        transition_watermark=0,
    )
    cache = RecordingCache(
        CacheConfig(
            gpu_pages=hot.admission_blocks(page_size, cfg.num_hidden_layers),
            host_pages=ceil(maximum / page_size) * cfg.num_hidden_layers,
            page_size=page_size,
            kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            device=str(model.device),
            dtype=model.dtype,
        ),
        hot,
    )
    try:
        cache.reserve("capture", maximum, cfg.num_hidden_layers)
        logits = model.prefill(tokens, cache, "capture")
        for _ in range(steps):
            token = int(logits.argmax())
            logits = model.decode(token, "capture", cache, top_k=None)
        validate_trace(trace)
        return trace
    finally:
        if "capture" in cache.storage.requests:
            cache.release_request("capture")
        cache.close()
