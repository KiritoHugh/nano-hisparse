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
            page.pins += 1
            store.pinned.add(page.key)
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
    generation = store.requests["a"].generation
    for page in list(store.pages.values()):
        store.detach(page)
    store.retire("a")
    store.collect()
    store.assert_invariants()
    assert len(store.free_host) == 6
    store.reserve("a", 12, 1)
    assert store.requests["a"].generation > generation


def test_host_credits_account_for_unmaterialized_pages():
    store = make_store()
    store.reserve("a", 24, 1)
    assert not store.pages and len(store.free_host) == 6
    with pytest.raises(CapacityError, match="host credits"):
        store.reserve("b", 1, 1)
    assert set(store.requests) == {"a"}
    store.assert_invariants()
