import pytest
import torch

from nano_hisparse.config import CacheConfig


@pytest.mark.parametrize("name", ["gpu_pages", "host_pages", "page_size", "kv_heads", "head_dim"])
def test_nonpositive_dimensions_are_rejected(name):
    with pytest.raises(ValueError, match=name):
        CacheConfig(**{name: 0})


def test_unsupported_dtype_and_empty_index_budget_are_rejected():
    with pytest.raises(ValueError, match="dtype"):
        CacheConfig(dtype=torch.int8)
    with pytest.raises(ValueError, match="index budget"):
        CacheConfig(index_budget_bytes=0)
