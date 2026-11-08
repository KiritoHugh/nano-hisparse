import pytest

from nano_hisparse.allocator import DeviceAllocator
from nano_hisparse.config import CacheConfig


def test_reclaimable_blocks_keep_their_owner_until_reuse():
    allocator = DeviceAllocator(CacheConfig(gpu_pages=2, device="cpu"))
    old = ("a", 0, 0)
    allocator.bind(0, old)
    assert allocator.slots[0].ref_cnt == 1
    callbacks = []

    def release(slot):
        callbacks.append(slot)
        allocator.release(slot, fence="old-reader")

    allocator.unpin(0, release)
    allocator.unpin(0, release)
    assert allocator.slots[0].owner == old and 0 not in allocator.free
    assert allocator.slots[0].ref_cnt == 0
    allocator.reclaim(0)
    assert callbacks == [0] and allocator.slots[0].fence == "old-reader"
    assert allocator.slots[0].ref_cnt == 0
    allocator.bind(0, ("b", 0, 0))
    assert allocator.slots[0].ref_cnt == 1
    assert allocator.occupancy["a"] == 0 and allocator.occupancy["b"] == 1


def test_every_reader_must_release_before_a_block_can_be_reused():
    allocator = DeviceAllocator(CacheConfig(gpu_pages=1, device="cpu"))
    key = ("a", 0, 0)
    allocator.bind(0, key)
    allocator.incref(0)
    allocator.incref(0)
    assert allocator.slots[0].ref_cnt == 3
    with pytest.raises(RuntimeError, match="active references"):
        allocator.release(0, None)
    allocator.unpin(0, lambda slot: allocator.release(slot, "done"))
    assert allocator.slots[0].ref_cnt == 2
    for remaining in (2, 1):
        assert allocator.slots[0].ref_cnt == remaining
        with pytest.raises(RuntimeError, match="active references"):
            allocator.reclaim(0)
        with pytest.raises(RuntimeError, match="active references"):
            allocator.release(0, None)
        assert allocator.slots[0].owner == key
        allocator.decref(0)
    assert allocator.slots[0].ref_cnt == 0
    with pytest.raises(RuntimeError, match="no reader"):
        allocator.decref(0)
    allocator.reclaim(0)
    assert allocator.free == {0} and allocator.slots[0].fence == "done"
    with pytest.raises(RuntimeError, match="unbound"):
        allocator.incref(0)
    allocator.bind(0, key)  # Reusing the same owner needs no generation counter.
    with pytest.raises(RuntimeError, match="no reader"):
        allocator.decref(0)
    allocator.release(0, None)
    assert allocator.slots[0].ref_cnt == 0
