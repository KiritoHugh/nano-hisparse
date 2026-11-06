from dataclasses import replace

import pytest
import torch

from nano_hisparse.attention import row_attention
from nano_hisparse.config import CacheConfig, CapacityError
from nano_hisparse.hybrid import HotConfig, HybridKVCache
from nano_hisparse.row_index import LogicalTopK, resolve_rows
from nano_hisparse.selection import Selection, dense_attention, selection_oracle


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    return request.param


def make(device, **overrides):
    config = replace(
        CacheConfig(gpu_pages=3, host_pages=64, page_size=4, kv_heads=1, head_dim=8, device=device),
        **overrides,
    )
    return HybridKVCache(config, HotConfig(4, 4))


def write(cache, name="a", length=32, layer=0):
    k, v = [
        torch.randn(
            length,
            cache.config.kv_heads,
            cache.config.head_dim,
            device=cache.device,
            dtype=cache.config.dtype,
        )
        for _ in range(2)
    ]
    cache.write(name, layer, k, v)
    return k, v


def topk(cache, values):
    return LogicalTopK(torch.tensor(values, device=cache.device, dtype=torch.int32))


def test_row_misses_copy_rows_and_resident_hits_bypass_hot(device):
    cache = make(device)
    cache.reserve("a", 40, 1)
    k, v = write(cache)
    q = torch.randn(2, 8, device=device)
    selected = [0, 7, 17, 31]
    for _ in range(2):
        actual = row_attention(q, cache, "a", 0, topk(cache, selected))
        torch.testing.assert_close(
            actual, dense_attention(q, k[selected], v[selected]), atol=2e-6, rtol=2e-5
        )
    stats = cache.snapshot()["row_stats"]
    assert stats == {"resident_hits": 2, "hot_hits": 3, "misses": 3, "h2d_bytes": 192}
    assert cache.pool.data_ptr() == cache.allocator.storage.data_ptr()
    cache.assert_invariants()


def test_all_resident_needs_no_hot_block_lease(device):
    cache = make(device, gpu_pages=8)
    cache.reserve("a", 32, 1)
    k, v = write(cache, length=20)
    q = torch.randn(2, 8, device=device)
    actual = row_attention(q, cache, "a", 0, topk(cache, [1, 7, 17, 19]))
    torch.testing.assert_close(actual, dense_attention(q, k[[1, 7, 17, 19]], v[[1, 7, 17, 19]]))
    assert not cache.tables["a", 0].blocks
    assert cache.snapshot()["row_stats"]["misses"] == 0


def test_late_hits_are_protected_and_lru_replaces_only_unselected_rows(device):
    cache = make(device)
    cache.reserve("a", 40, 1)
    k, v = write(cache)
    q = torch.randn(2, 8, device=device)
    for selected in ([0, 4, 8, 12], [16, 4, 8, 12], [4, 4, -1, 12]):
        actual = row_attention(q, cache, "a", 0, topk(cache, selected))
        unique = sorted({t for t in selected if t >= 0})
        # Duplicate top-k is deduplicated by resolution; attention must also
        # mask duplicate entries, otherwise it changes the softmax measure.
        torch.testing.assert_close(
            actual, dense_attention(q, k[unique], v[unique]), atol=2e-6, rtol=2e-5
        )
    assert cache.snapshot()["row_stats"]["misses"] == 5
    tables = cache.tables["a", 0]
    tag0 = cache.storage.pages["a", 0, 0].host_slot * 4
    assert tag0 not in tables.tags.tolist()
    cache.assert_invariants()


def test_per_head_token_masks_and_noncontiguous_hot_blocks(device):
    config = CacheConfig(
        gpu_pages=8, host_pages=64, page_size=3, kv_heads=2, head_dim=8, device=device
    )
    cache = HybridKVCache(config, HotConfig(6, 6, transition_watermark=0))
    cache.reserve("a", 40, 1)
    cache.reserve("b", 40, 1)
    # Interleave resident allocations so reclaimed A blocks are fragmented.
    for _ in range(3):
        write(cache, "a", 3)
        write(cache, "b", 3)
    write(cache, "b", 3)
    cache.storage.synchronize()
    cache.transition_to_host("a", 0)
    assert cache.tables["a", 0].blocks == [7, 0]
    write(cache, "a", 18)
    k, v = [x.to(device) for x in cache.storage.host_kv("a", 0)]
    q = torch.randn(4, 8, device=device)
    selection = Selection.from_tokens([[0, 7], [2], [7, 11], [19, 0]], 27, 3, 4)
    actual = cache.attention(q, "a", 0, selection=selection)
    expected, _ = selection_oracle(q, k, v, selection)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    cache.assert_invariants()


def test_capacity_admission_and_topk_overflow_are_atomic(device):
    cache = make(device, gpu_pages=3)
    cache.reserve("a", 40, 1)
    before = cache.index_bytes
    with pytest.raises(CapacityError, match="credits"):
        cache.reserve("b", 40, 1)
    assert "b" not in cache.storage.requests and cache.index_bytes == before
    write(cache)
    old = cache.tables["a", 0].tags.clone()
    with pytest.raises(ValueError, match="max_union_rows"):
        cache.resolve("a", 0, topk(cache, [0, 1, 2, 3, 4]))
    torch.testing.assert_close(old, cache.tables["a", 0].tags)
    with pytest.raises(ValueError):
        HotConfig(8, 4)
    cache.assert_invariants()


def test_index_tables_are_charged_to_budget(device):
    cache = make(device)
    needed = cache.index_reservation_bytes(40, 1)
    tiny = make(device, index_budget_bytes=needed - 1)
    with pytest.raises(CapacityError, match="index budget"):
        tiny.reserve("a", 40, 1)
    assert not tiny.storage.requests and not tiny.tables
    exact = make(device, index_budget_bytes=needed)
    exact.reserve("a", 40, 1)
    assert exact.index_bytes == needed
    exact.release_request("a")
    assert exact.index_bytes == 0


def test_active_tail_and_row_lease_lifetime(device):
    cache = make(device, gpu_pages=3)
    cache.reserve("a", 40, 1)
    write(cache)
    lease = cache.resolve("a", 0, topk(cache, [0, 4, 8, 31]))
    lease.wait()
    with pytest.raises(RuntimeError, match="already leased"):
        cache.resolve("a", 0, topk(cache, [1]))
    with pytest.raises(RuntimeError, match="active row"):
        cache.release_request("a")
    with pytest.raises(RuntimeError, match="append"):
        write(cache, length=1)
    lease.release()
    write(cache, length=1)
    assert cache.storage.pages["a", 0, 8].slot is not None
    assert cache.storage.pages["a", 0, 7].slot is not None
    assert cache.storage.pages["a", 0, 6].slot is None
    cache.assert_invariants()
    cache.release_request("a")
    with pytest.raises(RuntimeError, match="released"):
        lease.wait()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_row_gather_attention_low_precision(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cache = make("cuda:0", dtype=dtype, page_size=7, head_dim=24, kv_heads=2)
    cache.reserve("a", 100, 1)
    k, v = write(cache, length=80)
    q = torch.randn(6, 24, device="cuda", dtype=dtype)
    selected = [0, 13, 39, 79]
    actual = row_attention(q, cache, "a", 0, topk(cache, selected))
    torch.testing.assert_close(
        actual, dense_attention(q, k[selected], v[selected]), atol=0.004, rtol=0.02
    )


def test_device_resolver_supports_graph_capture_without_cpu_readback():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cache = make("cuda:0")
    cache.reserve("a", 40, 1)
    write(cache)
    tables = cache.tables["a", 0]
    logical = topk(cache, [0, 4, 8, 31])

    def resolve():
        return resolve_rows(
            logical, tables.source, tables.resident, tables.hot, tables.tags, tables.lru, 32, 4
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            resolve()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = resolve()
    logical.positions.copy_(torch.tensor([1, 5, 9, 31], device="cuda", dtype=torch.int32))
    graph.replay()
    assert result.counts.tolist() == [1, 0, 3]


def test_cuda_deferred_reuse_keeps_host_and_hot_readers_alive():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cache = make("cuda:0", gpu_pages=3)
    cache.reserve("same", 40, 1)
    k, _ = write(cache, "same")
    with cache.resolve("same", 0, topk(cache, [0, 4, 8, 31])) as lease:
        torch.cuda._sleep(2_000_000)
        row = lease.plan.physical_rows[0]
        actual = cache.pool[row // 4, 0, row % 4].clone()
    old_hosts = {p.host_slot for p in cache.storage.pages.values()}
    cache.release_request("same", wait=False)
    cache.reserve("same", 40, 1)
    write(cache, "same")
    if cache.storage.retired:
        assert old_hosts.isdisjoint({p.host_slot for p in cache.storage.pages.values()})
    torch.testing.assert_close(actual, k[0])
    cache.release_request("same")
    cache.assert_invariants()


def test_randomized_requests_layers_append_and_row_replacement(device):
    torch.manual_seed(17)
    cache = make(device, gpu_pages=12, host_pages=256)
    histories = {}
    for name in ("a", "b"):
        cache.reserve(name, 64, 2)
        for layer in range(2):
            histories[name, layer] = write(cache, name, 19, layer)
    for step in range(32):
        name, layer = ("a", "b")[step % 2], (step // 2) % 2
        k, v = histories[name, layer]
        new_k, new_v = write(cache, name, 1, layer)
        k, v = torch.cat((k, new_k)), torch.cat((v, new_v))
        histories[name, layer] = k, v
        selected = [*torch.randperm(len(k) - 1)[:3].tolist(), len(k) - 1]
        q = torch.randn(2, 8, device=device)
        actual = row_attention(q, cache, name, layer, topk(cache, selected))
        torch.testing.assert_close(
            actual, dense_attention(q, k[selected], v[selected]), atol=3e-6, rtol=3e-5
        )
        cache.assert_invariants()
    for name in ("a", "b"):
        cache.release_request(name)
    cache.assert_invariants()


def test_invalid_and_padding_only_topk_do_not_read_uninitialized_rows(device):
    cache = make(device)
    cache.reserve("a", 40, 1)
    write(cache)
    q = torch.randn(2, 8, device=device)
    output = row_attention(q, cache, "a", 0, topk(cache, [-1, -2, 100, 32]))
    torch.testing.assert_close(output, torch.zeros_like(output))
    assert cache.snapshot()["row_stats"]["misses"] == 0
