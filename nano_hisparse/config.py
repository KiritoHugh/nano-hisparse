"""Shared storage dimensions and capacity limits."""

from dataclasses import dataclass

import torch


class CapacityError(RuntimeError):
    pass


@dataclass(frozen=True)
class CacheConfig:
    gpu_pages: int = 16
    host_pages: int = 1024
    page_size: int = 16
    kv_heads: int = 2
    head_dim: int = 32
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    index_budget_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for name in ("gpu_pages", "host_pages", "page_size", "kv_heads", "head_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("KV dtype must be float32, float16 or bfloat16")
        if self.index_budget_bytes < 1:
            raise ValueError("index budget must be positive")
