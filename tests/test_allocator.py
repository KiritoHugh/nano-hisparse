import pytest

from nano_hisparse.allocator import DeviceAllocator
from nano_hisparse.config import CacheConfig


def test_reclaimable_blocks_keep_their_owner_until_reuse():
    allocator = DeviceAllocator(CacheConfig(gpu_pages=2, device="cpu"))
    old = ("a", 0, 0)
    allocator.bind(0, old)
    generation = allocator.slots[0].generation
    callbacks = []

    def release(slot):
        callbacks.append(slot)
        allocator.release(slot, fence="old-reader")

    allocator.unpin(0, release)
    allocator.unpin(0, release)
    assert allocator.slots[0].owner == old and 0 not in allocator.free
    allocator.validate(0, generation, old)
    allocator.reclaim(0)
    assert callbacks == [0] and allocator.slots[0].fence == "old-reader"
    allocator.bind(0, ("b", 0, 0))
    with pytest.raises(RuntimeError, match="stale"):
        allocator.validate(0, generation, old)
    assert allocator.occupancy["a"] == 0 and allocator.occupancy["b"] == 1
