"""Host-backed logical pages, append mirrors and event-safe retirement.

This component knows no eviction policy, hot region or attention window.
Callers supply the resident placement for append and detach it before retirement.
"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from math import ceil

import torch

from .allocator import DeviceAllocator
from .config import CacheConfig, CapacityError

PageKey = tuple[str, int, int]


class Location(str, Enum):
    HOST = "host"
    RESIDENT = "resident"
    HOT = "hot"


@dataclass
class Page:
    key: PageKey
    host_slot: int
    slot: int | None = None
    location: Location = Location.HOST
    valid_tokens: int = 0
    pins: int = 0
    ready: object = None
    last_use: object = None
    host_ready: object = None
    request_generation: int = 0
    content_version: int = 0
    host_enqueued_version: int = 0
    host_committed_version: int = 0
    sealed: bool = False
    last_attention_epoch: int = -1000000000


@dataclass
class RequestPages:
    max_tokens: int
    layers: int
    page_count: int
    lengths: list[int]
    summaries: torch.Tensor  # per-page mean K, always on device (the small index)
    minimum: torch.Tensor
    maximum: torch.Tensor
    host_credits: int
    generation: int
    keys: list[PageKey] = field(default_factory=list)
    extra_index_bytes: int = 0
    keepalive: tuple = ()  # caller metadata retained through the retirement event


class PageLease:
    """Pins protect mappings on CPU; events protect bytes on the accelerator."""

    def __init__(self, store: HostStore, pages: list[Page]):
        self.store, self.pages, self.released = store, pages, False
        self.handles = [(p.slot, store.allocator.slots[p.slot].generation, p.key) for p in pages]

    def _validate(self):
        self.store.check_stream()
        if self.released:
            raise RuntimeError("lease already released")
        for handle in self.handles:
            self.store.allocator.validate(*handle)

    def wait(self):
        self._validate()
        for page in self.pages:
            self.store.wait(self.store.compute, page.ready)

    def release(self):
        self._validate()
        done = self.store.event(self.store.compute)
        for page in self.pages:
            page.last_use = done
            page.pins -= 1
            if page.pins == 0:
                self.store.pinned.discard(page.key)
        self.released = True

    def __enter__(self):
        self.wait()
        return self

    def __exit__(self, *_):
        self.release()


class HostStore:
    def __init__(self, config: CacheConfig, allocator: DeviceAllocator):
        self.config = config
        self.device = torch.device(config.device)
        self.cuda = self.device.type == "cuda"
        if self.cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use device='cpu' for the reference path")
        shape = (2, config.page_size, config.kv_heads, config.head_dim)
        self.allocator = allocator
        self.pool = self.allocator.storage
        self.device = self.pool.device
        self.host = torch.empty(
            (config.host_pages, *shape), dtype=config.dtype, pin_memory=self.cuda
        )
        self.compute = torch.cuda.current_stream(self.device) if self.cuda else None
        self.transfer = torch.cuda.Stream(device=self.device) if self.cuda else None
        self.pages: dict[PageKey, Page] = {}
        self.requests: dict[str, RequestPages] = {}
        self.free_host = list(reversed(range(config.host_pages)))
        self.stats: Counter = Counter()
        self.pinned: set[PageKey] = set()
        self.retired: list[tuple[object, RequestPages, list[Page]]] = []
        self.request_generation = 0

    @property
    def budget(self):
        return self.allocator.budget

    @property
    def owners(self):
        return [slot.owner for slot in self.allocator.slots]

    @property
    def index_bytes(self):
        requests = [*self.requests.values(), *(r for _, r, _ in self.retired)]
        return sum(
            t.numel() * t.element_size()
            for r in requests
            for t in (r.summaries, r.minimum, r.maximum)
        ) + sum(r.extra_index_bytes for r in requests)

    def index_reservation_bytes(self, max_tokens, layers):
        return (
            ceil(max_tokens / self.config.page_size)
            * layers
            * self.config.kv_heads
            * self.config.head_dim
            * (4 + 2 * self.pool.element_size())
        )

    @property
    def available_host_credits(self):
        return len(self.free_host) - sum(r.host_credits for r in self.requests.values())

    def context(self, stream):
        return torch.cuda.stream(stream) if self.cuda else nullcontext()

    def event(self, stream):
        if not self.cuda:
            return None
        event = torch.cuda.Event()
        event.record(stream)
        return event

    def wait(self, stream, event):
        if event is not None:
            stream.wait_event(event)

    def check_stream(self):
        if self.cuda and torch.cuda.current_stream(self.device) != self.compute:
            raise RuntimeError("cache operations must use the construction compute stream")

    @property
    def page_bytes(self):
        return self.pool[0].numel() * self.pool.element_size()

    def reserve(self, request_id: str, max_tokens: int, layers: int, *, extra_index_bytes=0):
        """Reserve host backing and index bytes for the admitted lifetime."""
        self.check_stream()
        self.collect()
        if request_id in self.requests:
            raise ValueError(f"duplicate request {request_id}")
        if max_tokens < 1 or layers < 1:
            raise ValueError("max_tokens and layers must be positive")
        n = ceil(max_tokens / self.config.page_size)
        if n * layers > self.available_host_credits:
            raise CapacityError(
                f"need {n * layers} host credits; have {self.available_host_credits}"
            )
        index_bytes = self.index_reservation_bytes(max_tokens, layers) + extra_index_bytes
        if self.index_bytes + index_bytes > self.config.index_budget_bytes:
            raise CapacityError("GPU index budget exhausted; defer request admission")
        summaries = torch.zeros(
            (layers, n, self.config.kv_heads, self.config.head_dim),
            device=self.device,
            dtype=torch.float32,
        )
        request = RequestPages(
            max_tokens,
            layers,
            n,
            [0] * layers,
            summaries,
            # Bounds are actual K extrema and exactly representable in KV dtype.
            torch.full_like(summaries, torch.inf, dtype=self.config.dtype),
            torch.full_like(summaries, -torch.inf, dtype=self.config.dtype),
            n * layers,
            self.request_generation,
        )
        request.extra_index_bytes = extra_index_bytes
        self.request_generation += 1
        self.requests[request_id] = request

    def materialize(self, request_id, layer, index):
        key = (request_id, layer, index)
        if key not in self.pages:
            request = self.requests[request_id]
            assert request.host_credits > 0 and self.free_host
            request.host_credits -= 1
            request.keys.append(key)
            self.pages[key] = Page(key, self.free_host.pop(), request_generation=request.generation)
            self.stats["materialized_pages"] += 1
        return self.pages[key]

    def detach(self, page: Page):
        assert page.slot is not None and page.pins == 0
        slot = page.slot
        assert page.host_enqueued_version == page.content_version
        # GPU slot becomes logically free immediately, but reuse is fenced.
        # In particular, a queued D2H copy still READS this slot!
        for event in (page.ready, page.last_use, page.host_ready):
            self.wait(self.transfer, event)
        self.allocator.release(slot, self.event(self.transfer))
        page.slot, page.location = None, Location.HOST
        self.stats["evictions"] += 1

    def bind(self, page: Page, slot: int, location: Location):
        page.slot, page.location = slot, location
        page.last_use = None
        self.allocator.bind(slot, page.key)
        self.stats["peak_device_pages"] = max(
            self.stats["peak_device_pages"], len(self.allocator.lru)
        )

    @torch.inference_mode()
    def append(self, request_id, layer, k, v, prepare_page, after_write=None):
        """Append projected K/V and mirror only new rows.

        prepare_page(page) supplies resident placement and an optional read
        lease for an existing tail. after_write(page) updates caller policy
        after mirror submission. Storage never selects a replacement victim.
        """
        self.check_stream()
        request = self.requests[request_id]
        cfg = self.config
        if not 0 <= layer < request.layers:
            raise ValueError("invalid layer")
        expected = (cfg.kv_heads, cfg.head_dim)
        if k.ndim != 3 or k.shape != v.shape or tuple(k.shape[1:]) != expected:
            raise ValueError(f"expected matching [tokens, {expected[0]}, {expected[1]}] K/V")
        if k.device != self.pool.device or v.device != self.pool.device:
            raise ValueError("K/V must be on the cache device")
        if k.dtype != cfg.dtype or v.dtype != cfg.dtype:
            raise ValueError("K/V dtype must match the cache")
        start = request.lengths[layer]
        if start + len(k) > request.max_tokens:
            raise CapacityError("request exceeds its reserved host capacity")
        if any(key[0] == request_id and key[1] == layer for key in self.pinned):
            raise RuntimeError("cannot append while this layer has active attention leases")
        if len(self.pinned) >= self.budget:
            raise CapacityError("no slot available for append")
        offset = 0
        while offset < len(k):
            position = start + offset
            index, within = divmod(position, cfg.page_size)
            page = self.materialize(request_id, layer, index)
            count = min(cfg.page_size - within, len(k) - offset)
            lease = prepare_page(page)
            if lease is not None:
                lease.wait()
            # Join previous reads and host mirrors before modifying the page.
            for event in (page.ready, page.last_use, page.host_ready):
                self.wait(self.compute, event)
            target = self.pool[page.slot]
            target[0, within : within + count].copy_(k[offset : offset + count])
            target[1, within : within + count].copy_(v[offset : offset + count])
            page.valid_tokens = within + count
            page.content_version += 1
            page.sealed = page.valid_tokens == cfg.page_size
            request.summaries[layer, index].copy_(target[0, : page.valid_tokens].float().mean(0))
            new_keys = target[0, within : within + count].float()
            request.minimum[layer, index].copy_(
                torch.minimum(request.minimum[layer, index], new_keys.amin(0))
            )
            request.maximum[layer, index].copy_(
                torch.maximum(request.maximum[layer, index], new_keys.amax(0))
            )
            page.ready = self.event(self.compute)
            page.location = Location.RESIDENT
            self.wait(self.transfer, page.ready)
            with self.context(self.transfer):
                if within == 0 and count == cfg.page_size:
                    self.host[page.host_slot].copy_(target, non_blocking=self.cuda)
                else:
                    for plane in (0, 1):
                        self.host[page.host_slot, plane, within : within + count].copy_(
                            target[plane, within : within + count], non_blocking=self.cuda
                        )
            page.host_ready = self.event(self.transfer)
            page.host_enqueued_version = page.content_version
            if not self.cuda:
                page.host_committed_version = page.content_version
            self.stats["d2h_bytes"] += self.page_bytes * count // cfg.page_size
            if lease is not None:
                lease.release()
            if after_write is not None:
                after_write(page)
            offset += count
        request.lengths[layer] += len(k)

    def retire(self, request_id: str, *, keepalive=()):
        self.check_stream()
        request = self.requests[request_id]
        if any(self.pages[key].pins for key in request.keys):
            raise RuntimeError("cannot free a request with active leases")
        retired = []
        for key in request.keys:
            page = self.pages[key]
            assert page.slot is None, "detach resident blocks before retiring host backing"
            for event in (page.ready, page.last_use, page.host_ready):
                self.wait(self.transfer, event)
            retired.append(self.pages.pop(key))
        del self.requests[request_id]
        request.keepalive = tuple(keepalive)
        done = self.event(self.transfer)
        self.retired.append((done, request, retired))
        self.stats["retired_requests"] += 1
        return done

    def collect(self, wait=False):
        """Publish reusable host slots only AFTER all in-flight users finish.

        Normal scheduler polls never synchronize the CPU. When idle or shutting
        down, wait drains retirement events, rather than synchronizing a device.
        """
        pending = []
        for event, request, pages in self.retired:
            if wait and event is not None:
                event.synchronize()
            if event is None or event.query():
                self.free_host.extend(page.host_slot for page in pages)
                self.stats["reclaimed_requests"] += 1
            else:
                pending.append((event, request, pages))
        self.retired = pending

    def page_state(self, key):
        page = self.pages[key]
        backup_pending = page.host_ready is not None and not page.host_ready.query()
        if not backup_pending:
            page.host_committed_version = page.host_enqueued_version
        loading = page.ready is not None and not page.ready.query()
        return {
            "location": page.location.value,
            "loading": loading,
            "backup_pending": backup_pending,
            "content_version": page.content_version,
            "host_committed_version": page.host_committed_version,
            "sealed": page.sealed,
            "request_generation": page.request_generation,
            "slot_generation": self.allocator.slots[page.slot].generation
            if page.slot is not None
            else None,
        }

    def synchronize(self):
        if self.cuda:
            self.compute.synchronize()
            self.transfer.synchronize()

    def host_kv(self, request_id: str, layer: int):
        """Blocking diagnostic/oracle read, never called by the decode path."""
        self.synchronize()
        request = self.requests[request_id]
        chunks = [
            self.host[
                self.pages[(request_id, layer, i)].host_slot,
                :,
                : self.pages[(request_id, layer, i)].valid_tokens,
            ]
            for i in range(ceil(request.lengths[layer] / self.config.page_size))
        ]
        if not chunks:
            raise ValueError("layer has no KV")
        kv = torch.cat(chunks, dim=1)
        return kv[0], kv[1]

    def snapshot(self):
        counts = Counter(p.location.value for p in self.pages.values() if p.valid_tokens)
        return {
            "budget_pages": self.budget,
            "allocated_device_bytes": self.pool.numel() * self.pool.element_size(),
            "allocated_host_bytes": self.host.numel() * self.host.element_size(),
            "index_bytes": self.index_bytes,
            "index_budget_bytes": self.config.index_budget_bytes,
            "host_materialized_pages": len(self.pages),
            "host_unused_credits": sum(r.host_credits for r in self.requests.values()),
            "host_available_credits": self.available_host_credits,
            "retiring_requests": len(self.retired),
            "request_device_pages": dict(self.allocator.occupancy),
            "locations": dict(counts),
            "stats": dict(self.stats),
        }

    def assert_invariants(self):
        occupied = [key for key in self.owners if key is not None]
        assert len(occupied) == len(set(occupied))
        assert all(key is None for key in self.owners[self.budget :])
        retired_pages = [p for _, _, pages in self.retired for p in pages]
        assert len(self.free_host) + len(self.pages) + len(retired_pages) == self.config.host_pages
        host_slots = [p.host_slot for p in [*self.pages.values(), *retired_pages]]
        assert len(set(host_slots + self.free_host)) == self.config.host_pages
        assert self.available_host_credits >= 0
        assert self.pinned == {p.key for p in self.pages.values() if p.pins}
        assert set(self.allocator.lru) == set(occupied)
        assert self.allocator.free == {i for i, key in enumerate(self.owners) if key is None}
        assert +self.allocator.occupancy == Counter(key[0] for key in occupied)
        assert self.index_bytes <= self.config.index_budget_bytes
        for request in self.requests.values():
            assert len(request.keys) + request.host_credits == request.layers * request.page_count
        for key, page in self.pages.items():
            assert page.pins >= 0
            assert 0 <= page.valid_tokens <= self.config.page_size
            if page.slot is None:
                assert page.location == Location.HOST
            else:
                assert self.owners[page.slot] == key
                assert page.location in (Location.RESIDENT, Location.HOT)
            assert page.pins == 0 or page.slot is not None
            assert page.request_generation == self.requests[key[0]].generation
            assert page.host_committed_version <= page.host_enqueued_version == page.content_version
