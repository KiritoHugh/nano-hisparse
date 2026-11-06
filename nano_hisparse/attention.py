"""Read selected physical rows after resolving residency and gathering misses."""

import torch


@torch.inference_mode()
def row_attention(query, cache, request_id, layer, topk, backend="auto"):
    if backend not in ("auto", "torch"):
        raise ValueError("invalid attention backend")
    if query.ndim != 2 or query.shape[1] != cache.config.head_dim:
        raise ValueError("invalid query shape")
    if query.shape[0] % cache.config.kv_heads or query.device != cache.pool.device:
        raise ValueError("query GQA/device mismatch")
    if topk.head_columns is not None and topk.head_columns.shape[0] != query.shape[0]:
        raise ValueError("selection must have one row per query head")
    with cache.resolve(request_id, layer, topk) as lease:
        rows = lease.plan.physical_rows
        if rows.ndim == 1:
            rows = rows[None, :].expand(query.shape[0], -1)
        safe = rows.clamp(min=0)
        blocks, offsets = safe // cache.config.page_size, safe % cache.config.page_size
        head = torch.arange(query.shape[0], device=query.device)
        kv_head = head // (query.shape[0] // cache.config.kv_heads)
        k = cache.pool[blocks, 0, offsets, kv_head[:, None]].float()
        v = cache.pool[blocks, 1, offsets, kv_head[:, None]].float()
        v = torch.where((rows >= 0)[..., None], v, 0.0)
        scores = (k * query.float()[:, None]).sum(-1) * cache.config.head_dim**-0.5
        weights = scores.masked_fill(rows < 0, -torch.inf).softmax(-1).nan_to_num()
        return (weights[..., None] * v).sum(1).to(query.dtype)
