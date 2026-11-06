import pytest
import torch

from nano_hisparse.config import CacheConfig
from nano_hisparse.hybrid import HotConfig, HybridKVCache
from nano_hisparse.model import Model, ModelConfig


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_incremental_logits_match_full_causal_forward_after_offload(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(31)
    torch.set_num_threads(1)
    model = (
        Model(
            ModelConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=64,
            )
        )
        .eval()
        .to(device)
    )
    hot = HotConfig(24, 24)
    cache = HybridKVCache(
        CacheConfig(
            gpu_pages=hot.admission_blocks(4, 2),
            host_pages=64,
            page_size=4,
            kv_heads=2,
            head_dim=8,
            device=device,
        ),
        hot,
    )
    history = list(range(1, 18))
    cache.reserve("a", 24, 2)
    torch.testing.assert_close(model.prefill(history, cache, "a"), model.prefill(history))
    cache.storage.synchronize()
    for layer in range(2):
        cache.reclaim_history("a", layer, evict=True)
    for token in (18, 19, 20):
        history.append(token)
        actual = model.decode(token, "a", cache)
        torch.testing.assert_close(actual, model.prefill(history), atol=3e-6, rtol=3e-5)
    assert cache.snapshot()["row_stats"]["misses"] > 0
    with pytest.raises(ValueError, match="exactly once"):
        model.prefill(history, cache, "a")
    cache.release_request("a")
    cache.assert_invariants()
