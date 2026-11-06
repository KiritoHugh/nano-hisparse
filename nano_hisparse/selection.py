"""Explicit token semantics: page transfer granularity must not widen attention."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import torch


@dataclass(frozen=True)
class Selection:
    positions: tuple[tuple[int, ...], ...]  # per query head, canonical logical tokens
    length: int
    page_size: int
    source: str

    @classmethod
    def from_tokens(cls, rows, length, page_size, heads, source="external"):
        if length < 1 or page_size < 1 or heads < 1:
            raise ValueError("selection dimensions must be positive")
        if isinstance(rows, torch.Tensor):
            rows = rows.cpu().tolist()
        if rows and isinstance(rows[0], int):
            rows = [rows]
        if len(rows) == 1:
            rows = rows * heads
        if len(rows) != heads:
            raise ValueError("selection must be shared or have one row per query head")
        canonical = []
        for row in rows:
            if not isinstance(row, (list, tuple)):
                raise TypeError("selection rows must be integer sequences")
            if any(not isinstance(t, int) or t < -1 or t >= length for t in row):
                raise ValueError("token selection contains noncausal or invalid positions")
            values = tuple(sorted({t for t in row if t != -1}))
            if not values:
                raise ValueError("each query head must select at least one token")
            canonical.append(values)
        return cls(tuple(canonical), length, page_size, source)

    @classmethod
    def from_pages(cls, rows, length, page_size, heads, source="pages"):
        if isinstance(rows, torch.Tensor):
            rows = rows.cpu().tolist()
        if rows and isinstance(rows[0], int):
            rows = [rows]
        count = ceil(length / page_size)
        if any(p < 0 or p >= count for row in rows for p in row):
            raise ValueError("invalid logical page")
        return cls.from_tokens(
            [
                [t for p in row for t in range(p * page_size, min((p + 1) * page_size, length))]
                for row in rows
            ],
            length,
            page_size,
            heads,
            source,
        )

    @property
    def pages(self):
        return sorted({t // self.page_size for row in self.positions for t in row})


def quest_selection(query, minimum, maximum, length, page_size, top_pages, keep_sink=True):
    """Quest-style per-head bounds; explicit GQA page union, not native DSA.

    max(q_d * min(K_d), q_d * max(K_d)) upper-bounds each page's qK.
    Each head selects independently; the cache transfers the UNION of pages.
    """
    count = ceil(length / page_size)
    if top_pages < 1 or query.shape[0] % minimum.shape[1]:
        raise ValueError("invalid page budget or GQA shape")
    heads = query.shape[0]
    if top_pages >= count:
        return Selection.from_pages(list(range(count)), length, page_size, heads, "quest")
    groups = heads // minimum.shape[1]
    low = minimum[:count].repeat_interleave(groups, 1).transpose(0, 1).float()
    high = maximum[:count].repeat_interleave(groups, 1).transpose(0, 1).float()
    scores = torch.maximum(query.float()[:, None] * low, query.float()[:, None] * high).sum(-1)
    required = [count - 1]
    if keep_sink and top_pages > 1 and count > 1:
        required.append(0)
    scores[:, required] = torch.inf
    chosen = scores.topk(top_pages, dim=-1).indices.tolist()
    return Selection.from_pages(chosen, length, page_size, heads, "quest")


def selection_oracle(query, k, v, selection):
    """Offline only: quantify approximation separately from storage error."""
    groups = query.shape[0] // k.shape[1]
    keys, values = k.repeat_interleave(groups, 1).float(), v.repeat_interleave(groups, 1).float()
    scores = torch.einsum("hd,thd->ht", query.float(), keys) / query.shape[-1] ** 0.5
    probability = scores.softmax(-1)
    full = torch.einsum("ht,thd->hd", probability, values)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for head, positions in enumerate(selection.positions):
        mask[head, list(positions)] = True
    sparse = torch.einsum("ht,thd->hd", scores.masked_fill(~mask, -torch.inf).softmax(-1), values)
    recalls = []
    for head, positions in enumerate(selection.positions):
        oracle = set(scores[head].topk(len(positions)).indices.tolist())
        recalls.append(len(oracle & set(positions)) / len(oracle))
    metrics = {
        "retained_attention_mass": (probability * mask).sum(-1).mean().item(),
        "oracle_token_recall": sum(recalls) / len(recalls),
        "relative_output_l2": ((sparse - full).norm() / full.norm().clamp_min(1e-12)).item(),
        "selected_token_fraction": mask.float().mean().item(),
        "union_pages": len(selection.pages),
    }
    return sparse.to(query.dtype), metrics


def select_pages(
    cache, request_id: str, layer: int, query: torch.Tensor, top_k: int | None
) -> list[int]:
    """A small page-mean indexer for demonstrations, NOT the learned DSA indexer.

    None selects the full causal history. Sparse mode keeps the first and last
    page (within its budget), then uses query/mean-key scores for the remainder.
    """
    request = cache.storage.requests[request_id]
    count = ceil(request.lengths[layer] / cache.config.page_size)
    if count == 0:
        raise ValueError("cannot attend to empty KV")
    if top_k is None or top_k >= count:
        return list(range(count))
    if top_k < 1:
        raise ValueError("top_k must be positive or None")
    required = [count - 1] if top_k == 1 else [0, count - 1]
    if top_k == len(required):
        return sorted(required)
    groups = query.shape[0] // cache.config.kv_heads
    grouped_query = query.float().reshape(cache.config.kv_heads, groups, -1).mean(1)
    scores = torch.einsum("hd,phd->p", grouped_query, request.summaries[layer, :count])
    scores[required] = -torch.inf
    # Deliberate reference-control-plane synchronization; a production fused
    # resolver keeps these indices and the residency table entirely on GPU.
    chosen = scores.topk(top_k - len(required)).indices.tolist()
    return sorted(required + chosen)


def dense_attention(query: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Independent full-KV oracle, also useful for correctness tests."""
    groups = query.shape[0] // k.shape[1]
    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    scores = torch.einsum("hd,thd->ht", query.float(), k.float()) / sqrt(query.shape[-1])
    return torch.einsum("ht,thd->hd", scores.softmax(-1), v.float()).to(query.dtype)
