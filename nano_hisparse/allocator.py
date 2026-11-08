"""Physical capacity and leases, independent of host KV contents."""

from collections import Counter, OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class DeviceSlot:
    owner: tuple | None = None
    ref_cnt: int = 0
    fence: object = None


class DeviceAllocator:
    def __init__(self, config):
        self.storage = torch.empty(
            (config.gpu_pages, 2, config.page_size, config.kv_heads, config.head_dim),
            device=config.device,
            dtype=config.dtype,
        )
        self.slots = [DeviceSlot() for _ in range(config.gpu_pages)]
        self.free = set(range(config.gpu_pages))
        # Empty blocks and readable-but-unpinned blocks are distinct. Both
        # provide capacity; reuse of the latter must revoke the old mapping.
        self.reclaimable = OrderedDict()
        self.lru = OrderedDict()
        self.occupancy = Counter()
        self.budget = config.gpu_pages

    def bind(self, slot, key):
        assert slot in self.free and slot < self.budget
        frame = self.slots[slot]
        assert frame.owner is None and frame.ref_cnt == 0
        self.free.remove(slot)
        frame.ref_cnt = 1  # allocation reference: resident page or hot block
        frame.owner = key
        self.lru[key] = slot
        self.occupancy[key[0]] += 1

    def release(self, slot, fence):
        frame = self.slots[slot]
        assert frame.owner is not None
        allocation_ref = int(slot not in self.reclaimable)
        if frame.ref_cnt != allocation_ref:
            raise RuntimeError("cannot release a block with active references")
        key = frame.owner
        frame.ref_cnt = 0
        self.reclaimable.pop(slot, None)
        del self.lru[key]
        self.occupancy[key[0]] -= 1
        frame.owner, frame.fence = None, fence
        self.free.add(slot)

    def unpin(self, slot, on_reuse):
        assert self.slots[slot].owner is not None and slot not in self.free
        if slot not in self.reclaimable:
            assert self.slots[slot].ref_cnt > 0
            self.slots[slot].ref_cnt -= 1  # relinquish the allocation reference
            self.reclaimable[slot] = on_reuse

    def reclaim(self, slot):
        """Run the ownership callback only when the block is actually taken."""
        if self.slots[slot].ref_cnt != 0:
            raise RuntimeError("cannot reclaim a block with active references")
        self.reclaimable[slot](slot)
        assert slot in self.free and slot not in self.reclaimable

    def incref(self, slot):
        """Temporarily protect readable contents, including a zero-ref block."""
        frame = self.slots[slot]
        if frame.owner is None:
            raise RuntimeError("cannot reference an unbound block")
        frame.ref_cnt += 1

    def decref(self, slot):
        """Release one reader reference; allocation ownership has its own release."""
        frame = self.slots[slot]
        allocation_ref = int(slot not in self.reclaimable)
        if frame.owner is None or frame.ref_cnt <= allocation_ref:
            raise RuntimeError("no reader reference to release")
        frame.ref_cnt -= 1
