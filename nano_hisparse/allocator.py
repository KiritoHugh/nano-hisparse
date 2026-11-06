"""Physical capacity and leases, independent of host KV contents."""

from collections import Counter, OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class DeviceSlot:
    owner: tuple | None = None
    generation: int = 0
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
        self.free.remove(slot)
        frame = self.slots[slot]
        frame.generation += 1
        frame.owner = key
        self.lru[key] = slot
        self.occupancy[key[0]] += 1

    def release(self, slot, fence):
        frame = self.slots[slot]
        assert frame.owner is not None
        key = frame.owner
        self.reclaimable.pop(slot, None)
        del self.lru[key]
        self.occupancy[key[0]] -= 1
        frame.owner, frame.fence = None, fence
        self.free.add(slot)

    def unpin(self, slot, on_reuse):
        assert self.slots[slot].owner is not None and slot not in self.free
        self.reclaimable.setdefault(slot, on_reuse)

    def reclaim(self, slot):
        """Run the ownership callback only when the block is actually taken."""
        self.reclaimable[slot](slot)
        assert slot in self.free and slot not in self.reclaimable

    def validate(self, slot, generation, owner):
        frame = self.slots[slot]
        if frame.owner != owner or frame.generation != generation:
            raise RuntimeError("stale device-slot generation (ABA)")
