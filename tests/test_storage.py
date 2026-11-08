import pytest
import torch

from nano_hisparse.allocator import DeviceAllocator
from nano_hisparse.config import CacheConfig, CapacityError
from nano_hisparse.storage import HostStore, Location, PageLease


def make_store():
    config = CacheConfig(
        gpu_pages=3, host_pages=6, page_size=4, kv_heads=1, head_dim=4, device="cpu"
    )
    return HostStore(config, DeviceAllocator(config))


def test_append_mirrors_only_new_rows_and_retirement_releases_backing():
    store = make_store()
    store.reserve("a", 12, 1)

    def prepare(page):
        if page.slot is None:
            store.bind(page, min(store.allocator.free), Location.RESIDENT)
        else:
            return PageLease(store, [page])

    k, v = torch.randn(9, 1, 4), torch.randn(9, 1, 4)
    store.append("a", 0, k[:3], v[:3], prepare)
    store.append("a", 0, k[3:], v[3:], prepare)
    actual_k, actual_v = store.host_kv("a", 0)
    torch.testing.assert_close(actual_k, k)
    torch.testing.assert_close(actual_v, v)
    assert store.stats["d2h_bytes"] == 9 * 2 * 4 * 4
    assert store.pages["a", 0, 0].content_version == 2
    assert store.pages["a", 0, 0].sealed
    old_request = store.requests["a"]
    old_page = store.pages["a", 0, 0]
    for page in list(store.pages.values()):
        store.detach(page)
    store.retire("a")
    store.collect()
    store.assert_invariants()
    assert len(store.free_host) == 6
    store.reserve("a", 12, 1)
    assert store.requests["a"] is not old_request
    new_page = store.materialize("a", 0, 0)
    store.bind(new_page, min(store.allocator.free), Location.RESIDENT)
    with pytest.raises(RuntimeError, match="retired page"):
        PageLease(store, [old_page])
    assert store.allocator.slots[new_page.slot].ref_cnt == 1


def test_host_credits_account_for_unmaterialized_pages():
    store = make_store()
    store.reserve("a", 24, 1)
    assert not store.pages and len(store.free_host) == 6
    with pytest.raises(CapacityError, match="host credits"):
        store.reserve("b", 1, 1)
    assert set(store.requests) == {"a"}
    store.assert_invariants()


def test_overlapping_page_leases_balance_references_and_reject_released_handles():
    store = make_store()
    store.reserve("a", 4, 1)
    page = store.materialize("a", 0, 0)
    store.bind(page, 0, Location.RESIDENT)
    store.allocator.unpin(0, lambda slot: store.detach(page))
    first, second = PageLease(store, [page]), PageLease(store, [page])
    assert store.allocator.slots[0].ref_cnt == page.pins == 2
    first.release()
    assert store.allocator.slots[0].ref_cnt == page.pins == 1
    with pytest.raises(RuntimeError, match="active references"):
        store.allocator.reclaim(0)
    second.release()
    assert store.allocator.slots[0].ref_cnt == page.pins == 0
    store.allocator.reclaim(0)
    store.bind(page, 0, Location.RESIDENT)
    for lease in (first, second):
        with pytest.raises(RuntimeError, match="released"):
            lease.wait()
        with pytest.raises(RuntimeError, match="released"):
            lease.release()
    assert store.allocator.slots[0].ref_cnt == 1
    store.assert_invariants()
