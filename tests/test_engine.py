import pytest
import torch

from nano_hisparse.config import CacheConfig, CapacityError
from nano_hisparse.engine import Engine
from nano_hisparse.hybrid import HotConfig, HybridKVCache
from nano_hisparse.model import Model, ModelConfig


def make_model(device="cpu"):
    return (
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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("top_k", [None, 2])
def test_partial_residency_matches_resident_generation_without_reprefill(device, top_k):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(31)
    torch.set_num_threads(1)
    model = make_model(device)
    union = 28 if top_k is None else 8
    hot = HotConfig(union, union)
    outputs = []
    for budget in (64, hot.admission_blocks(4, 2)):
        cache = HybridKVCache(
            CacheConfig(
                gpu_pages=budget, host_pages=64, page_size=4, kv_heads=2, head_dim=8, device=device
            ),
            hot,
        )
        engine = Engine(model, cache, top_k=top_k)
        engine.add_request("a", list(range(1, 18)), 8)
        engine.add_request("b", list(range(8, 30)), 6)
        engine.step()
        if budget != 64:
            assert len(engine.waiting) == 1
        outputs.append(engine.run())
        assert all(r.prefill_count == 1 for r in engine.requests.values())
        assert all(r.decode_count == r.max_new_tokens - 1 for r in engine.requests.values())
        assert not cache.storage.requests and not cache.tables and not any(cache.owners)
        if budget != 64:
            assert cache.snapshot()["row_stats"]["misses"] > 0
        cache.assert_invariants()
    assert outputs[0] == outputs[1]
    if top_k is None:
        for name, prompt in (("a", list(range(1, 18))), ("b", list(range(8, 30)))):
            history = prompt[:]
            for token in outputs[0][name]:
                assert model.prefill(history).argmax().item() == token
                history.append(token)


def test_host_backpressure_eventually_admits_waiters():
    torch.manual_seed(4)
    model = make_model()
    hot = HotConfig(16, 16)
    cache = HybridKVCache(
        CacheConfig(gpu_pages=24, host_pages=8, page_size=4, kv_heads=2, head_dim=8, device="cpu"),
        hot,
    )
    engine = Engine(model, cache)
    for i in range(3):
        engine.add_request(str(i), [1] * 10, 5)
    engine.step()
    assert len(engine.waiting) == 2
    engine.run()
    assert all(len(r.generated) == 5 and r.prefill_count == 1 for r in engine.requests.values())


def test_cancel_running_and_waiting_requests_releases_all_capacity():
    model = make_model()
    hot = HotConfig(8, 8)
    cache = HybridKVCache(
        CacheConfig(gpu_pages=8, host_pages=64, page_size=4, kv_heads=2, head_dim=8, device="cpu"),
        hot,
    )
    engine = Engine(model, cache, top_k=2)
    for name in ("a", "b", "c"):
        engine.add_request(name, list(range(1, 22)), 5)
    engine.step()
    engine.cancel("a")
    engine.cancel("b")
    outputs = engine.run()
    assert len(outputs["c"]) == 5
    assert engine.requests["a"].cancelled and engine.requests["b"].prefill_count == 0
    assert not any(cache.owners) and len(cache.storage.free_host) == 64
    cache.assert_invariants()
    with pytest.raises(CapacityError, match="host"):
        engine.add_request("oversized", [1] * 1000, 2)
