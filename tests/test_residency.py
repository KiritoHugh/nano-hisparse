"""Readable free pages, completion publication and proactive hot admission."""

import pytest
import torch

from nano_hisparse.attention import row_attention
from nano_hisparse.config import CacheConfig, CapacityError
from nano_hisparse.hybrid import ACTIVE_TAIL_PAGES, HotConfig, HybridKVCache
from nano_hisparse.row_index import LogicalTopK
from nano_hisparse.selection import dense_attention


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    return request.param


def make(device, pages=7, watermark=0, window=64):
    return HybridKVCache(
        CacheConfig(
            gpu_pages=pages, host_pages=128, page_size=4, kv_heads=1, head_dim=8, device=device
        ),
        HotConfig(4, 4, resident_window_pages=window, transition_watermark=watermark),
    )


def write(cache, name, count):
    k, v = [torch.randn(count, 1, 8, device=cache.device) for _ in range(2)]
    cache.write(name, 0, k, v)
    return k, v


def select(cache, positions):
    return LogicalTopK(torch.tensor(positions, dtype=torch.int32, device=cache.device))


def test_unpinned_page_still_hits_until_actual_block_reuse(device):
    cache = make(device)
    for name in ("a", "b"):
        cache.reserve(name, 64, 1)
    k, v = write(cache, "a", 12)
    cache.storage.synchronize()
    page = cache.storage.pages["a", 0, 0]
    slot = page.slot
    assert cache.allocator.slots[slot].ref_cnt == 1
    host = page.host_slot
    cache.reclaim_history("a", 0)
    assert cache.page_state(page.key)["residency"] == "clean_unpinned"
    assert page.slot == slot and cache.tables["a", 0].resident[0].item() == slot
    assert cache.allocator.slots[slot].ref_cnt == 0
    assert cache.snapshot()["allocatable_device_blocks"] == 4  # 3 empty + 1 readable

    q = torch.randn(2, 8, device=device)
    actual = row_attention(q, cache, "a", 0, select(cache, [0]))
    torch.testing.assert_close(actual, dense_attention(q, k[:1], v[:1]))
    assert cache.snapshot()["row_stats"]["misses"] == 0
    write(cache, "b", 12)  # Empty blocks take priority over readable clean KV.
    assert page.slot == slot and cache.storage.stats["unpinned_block_reuses"] == 0
    write(cache, "b", 1)  # No empties remain; now reuse the old A block.
    assert page.slot is None and page.host_slot == host
    assert cache.tables["a", 0].resident[0].item() == -1
    assert cache.owners[slot] == ("b", 0, 3)
    assert cache.allocator.slots[slot].ref_cnt == 1
    actual = row_attention(q, cache, "a", 0, select(cache, [0]))
    torch.testing.assert_close(actual, dense_attention(q, k[:1], v[:1]))
    assert cache.snapshot()["row_stats"]["misses"] == 1
    cache.assert_invariants()


def test_consumer_lease_temporarily_protects_a_readable_free_block(device):
    cache = make(device)
    for name in ("a", "b"):
        cache.reserve(name, 64, 1)
    k, _ = write(cache, "a", 12)
    cache.storage.synchronize()
    cache.reclaim_history("a", 0)
    page = cache.storage.pages["a", 0, 0]
    slot = page.slot
    with cache.resolve("a", 0, select(cache, [0])) as lease:
        assert slot in cache.allocator.reclaimable
        assert cache.allocator.slots[slot].ref_cnt == 1
        assert slot not in cache._available_blocks()
        with pytest.raises(RuntimeError, match="active references"):
            cache.allocator.reclaim(slot)
        assert all(cache.allocator.slots[i].ref_cnt == 2 for i in cache.tables["a", 0].blocks)
        cache.assert_invariants()
        if cache.storage.cuda:
            torch.cuda._sleep(2_000_000)
        observed = cache.pool[slot, 0, 0].clone()
        write(cache, "b", 13)  # B must reclaim its own history instead.
        assert page.slot == slot
        assert cache.tables["b", 0].blocks
        lease._validate()
    torch.testing.assert_close(observed, k[0])
    assert slot in cache._available_blocks()
    assert cache.allocator.slots[slot].ref_cnt == 0
    assert all(cache.allocator.slots[i].ref_cnt == 1 for i in cache.tables["a", 0].blocks)
    with pytest.raises(RuntimeError, match="released"):
        lease.release()
    cache.assert_invariants()


def test_window_triggers_hot_without_memory_exhaustion(device):
    cache = make(device, pages=12, window=3)
    cache.reserve("a", 64, 1)
    write(cache, "a", 8)
    assert not cache.tables["a", 0].hot_required
    write(cache, "a", 4)
    tables = cache.tables["a", 0]
    assert tables.hot_required and tables.trigger == "window" and tables.blocks
    assert cache.storage.stats["evictions"] == 0
    assert len(cache.allocator.free) == 8
    cache.storage.synchronize()
    cache.collect()
    assert cache.page_state(("a", 0, 0))["residency"] == "clean_unpinned"
    assert cache.storage.pages["a", 0, 0].slot is not None
    cache.assert_invariants()


def test_watermark_uses_projected_capacity_before_the_last_block(device):
    cache = make(device, pages=10, watermark=4)
    cache.reserve("a", 64, 1)
    write(cache, "a", 24)
    assert len(cache.allocator.free) == 4 and not cache.tables["a", 0].blocks
    write(cache, "a", 4)
    assert cache.tables["a", 0].trigger == "watermark"
    assert cache.tables["a", 0].blocks
    assert len(cache.allocator.free) == 2 and cache.storage.stats["evictions"] == 0
    cache.assert_invariants()


def test_hot_demand_survives_a_busy_layer_without_revoking_residents(device):
    cache = make(device, pages=6)
    cache.reserve("a", 64, 1)
    write(cache, "a", 12)
    before = cache.tables["a", 0].resident.clone()
    with cache.resolve("a", 0, select(cache, [0, 4, 8])):
        with pytest.raises(CapacityError, match="active row lease"):
            cache.transition_to_host("a", 0)
        assert cache.tables["a", 0].hot_required
        assert not cache.tables["a", 0].blocks
        torch.testing.assert_close(before, cache.tables["a", 0].resident)
        cache.collect()
        assert not cache.tables["a", 0].blocks
    cache.collect()
    assert cache.tables["a", 0].blocks
    cache.assert_invariants()


def test_two_active_tails_survive_full_and_partial_page_boundaries(device):
    assert ACTIVE_TAIL_PAGES == 2
    cache = make(device, pages=3, watermark=None)
    cache.reserve("a", 64, 1)
    write(cache, "a", 9)
    assert cache.storage.pages["a", 0, 0].slot is None
    assert cache.storage.pages["a", 0, 1].slot is not None
    assert cache.storage.pages["a", 0, 2].slot is not None
    write(cache, "a", 3)  # Last page seals; the preceding full page is still a tail.
    assert cache.storage.pages["a", 0, 1].slot is not None
    write(cache, "a", 1)
    assert cache.storage.pages["a", 0, 1].slot is None
    assert cache.storage.pages["a", 0, 2].slot is not None
    assert cache.storage.pages["a", 0, 3].slot is not None
    cache.assert_invariants()
    small = make(device, pages=2)
    with pytest.raises(CapacityError, match="credits"):
        small.reserve("a", 64, 1)
    assert not small.storage.requests and not small.tables


def test_minimum_hot_capacity_progresses_when_every_selected_row_misses(device):
    """H=U needs no spare row: protect zero hits, replace all slots, then cross tails."""
    torch.manual_seed(29)
    cache = make(device, pages=3, watermark=None)  # one hot block + two tails
    cache.reserve("a", 64, 1)
    k, v = write(cache, "a", 32)
    cache.reset_stats()
    choices = ([0, 4, 8, 12], [1, 5, 9, 13])
    for step in range(16):
        new_k, new_v = write(cache, "a", 1)
        k, v = torch.cat((k, new_k)), torch.cat((v, new_v))
        positions = choices[step % 2]
        assert all(cache.storage.pages["a", 0, t // 4].slot is None for t in positions)
        q = torch.randn(2, 8, device=device)
        actual = row_attention(q, cache, "a", 0, select(cache, positions))
        torch.testing.assert_close(
            actual, dense_attention(q, k[positions], v[positions]), atol=3e-6, rtol=3e-5
        )
        tail = (len(k) - 1) // 4
        assert all(cache.storage.pages["a", 0, p].slot is not None for p in (tail - 1, tail))
        cache.assert_invariants()
    assert cache.row_stats.tolist() == [0, 0, 64]
    assert cache.storage.requests["a"].lengths == [48]
    assert cache.storage.stats["peak_device_pages"] <= 3
    cache.release_request("a")
    cache.assert_invariants()


def test_pending_mirror_is_not_clean_and_policy_tick_does_not_wait():
    class Completion:
        done = False

        def query(self):
            return self.done

        def synchronize(self):
            pytest.fail("a normal policy tick must not wait for pending mirrors")

    cache = make("cpu", pages=6)
    cache.reserve("a", 64, 1)
    write(cache, "a", 12)
    page = cache.storage.pages["a", 0, 0]
    completion = Completion()
    page.host_ready = completion
    page.host_committed_version = 0
    cache.pending_clean.add(page.key)
    cache.reclaim_history("a", 0)
    assert cache.page_state(page.key)["residency"] == "dirty"
    assert page.slot not in cache.allocator.reclaimable
    cache.collect()
    assert page.slot not in cache.allocator.reclaimable
    completion.done = True
    cache.collect()
    assert cache.page_state(page.key)["residency"] == "clean_unpinned"
    page.host_ready = None
    cache.release_request("a")
    cache.assert_invariants()


def test_cuda_page_is_not_unpinned_before_its_d2h_event_completes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cache = make("cuda:0", pages=8)
    cache.reserve("a", 64, 1)
    k, v = [torch.randn(12, 1, 8, device=cache.device) for _ in range(2)]
    cache.storage.synchronize()
    with torch.cuda.stream(cache.storage.transfer):
        torch.cuda._sleep(100_000_000)
    cache.write("a", 0, k, v)
    page = cache.storage.pages["a", 0, 0]
    cache.transition_to_host("a", 0)
    assert not page.host_ready.query()
    assert page.slot not in cache.allocator.reclaimable
    assert cache.storage.stats["pressure_mirror_waits"] == 0
    cache.storage.synchronize()
    cache.collect()
    assert cache.page_state(page.key)["residency"] == "clean_unpinned"
    cache.assert_invariants()


def test_pressure_waits_only_for_an_eligible_mirror_event():
    class Completion:
        done = False
        waits = 0

        def query(self):
            return self.done

        def synchronize(self):
            self.done = True
            self.waits += 1

    cache = make("cpu", pages=3, watermark=0)
    cache.reserve("a", 64, 1)
    write(cache, "a", 12)
    # All blocks resident; a hot region and a new tail require completed history.
    completions = []
    for index in (0, 1):
        page = cache.storage.pages["a", 0, index]
        event = Completion()
        page.host_ready, page.host_committed_version = event, 0
        cache.pending_clean.add(page.key)
        completions.append(event)
    original_wait = cache.storage.wait
    cache.storage.wait = lambda stream, event: None  # CPU completion test double only
    write(cache, "a", 1)
    assert sum(e.waits for e in completions) >= 1
    assert cache.storage.stats["pressure_mirror_waits"] >= 1
    for page in cache.storage.pages.values():
        page.host_ready = None
    cache.storage.wait = original_wait
    cache.release_request("a")
    cache.assert_invariants()


def test_allocation_revisits_a_group_whose_mirror_was_published_late(monkeypatch):
    cache = make("cpu", pages=4)
    cache.reserve("a", 64, 1)
    write(cache, "a", 12)
    cache.transition_to_host("a", 0)
    page = cache.storage.pages["a", 0, 0]
    slot = page.slot
    cache.allocator.reclaimable.pop(slot)
    cache.allocator.incref(slot)  # Restore allocation ownership for this policy test.
    page.host_committed_version = 0
    assert not cache.allocator.free

    # A later group's transition may publish this page after the policy loop
    # has already visited A. Allocation must revisit the newly clean page.
    def publish_after_policy_visit(**kwargs):
        page.host_committed_version = page.content_version

    monkeypatch.setattr(cache, "update_residency", publish_after_policy_visit)
    assert cache._slot() == slot
    assert page.slot is None
    assert cache.tables["a", 0].resident[0].item() == -1
    cache.assert_invariants()
