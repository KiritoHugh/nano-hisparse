"""HiSparse-style block leases with independent token-row hot contents.

HostStore owns host/append lifetimes; this class owns hybrid policy. Device block
allocation is scheduler-side; GPU row replacement never allocates blocks.
"""

from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from math import ceil

import torch

from .allocator import DeviceAllocator
from .config import CapacityError
from .row_index import LogicalTopK, resolve_rows
from .storage import HostStore, Location, PageLease

ACTIVE_TAIL_PAGES = 2  # vLLM snapshot: v1/hisparse/types.py


@dataclass(frozen=True)
class HotConfig:
    max_union_rows: int
    hot_rows: int | None = None
    resident_window_pages: int = 64
    transition_watermark: int | None = None

    def __post_init__(self):
        if self.max_union_rows < 1:
            raise ValueError("max_union_rows must be positive")
        if self.hot_rows is None:
            object.__setattr__(self, "hot_rows", 2 * self.max_union_rows)
        if self.hot_rows < self.max_union_rows or self.hot_rows > 32768:
            raise ValueError("hot_rows must cover max_union_rows and be <= 32768")
        if self.resident_window_pages < ACTIVE_TAIL_PAGES:
            raise ValueError("resident window must cover both active tail pages")
        if self.transition_watermark is not None and self.transition_watermark < 0:
            raise ValueError("transition watermark must be nonnegative")

    def admission_blocks(self, page_size, layers=1):
        return layers * (ceil(self.hot_rows / page_size) + ACTIVE_TAIL_PAGES)


@dataclass
class LayerTables:
    source: torch.Tensor
    resident: torch.Tensor
    hot: torch.Tensor
    tags: torch.Tensor
    lru: torch.Tensor
    blocks: list[int] = field(default_factory=list)
    ready: object = None
    last_use: object = None
    busy: bool = False
    hot_required: bool = False
    trigger: str | None = None

    @property
    def nbytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (self.source, self.resident, self.hot, self.tags, self.lru)
        )


class RowLease:
    def __init__(self, cache, tables, plan, pages):
        self.cache, self.tables, self.plan, self.pages = cache, tables, plan, pages
        self.handles = [
            (i, cache.allocator.slots[i].generation, cache.allocator.slots[i].owner)
            for i in tables.blocks + [p.slot for p in pages]
        ]
        self.released = False

    def _validate(self):
        self.cache.storage.check_stream()
        if self.released:
            raise RuntimeError("row lease already released")
        for handle in self.handles:
            self.cache.allocator.validate(*handle)

    def wait(self):
        self._validate()
        self.cache.storage.wait(self.cache.storage.compute, self.tables.ready)

    def release(self):
        self._validate()
        self.tables.last_use = self.cache.storage.event(self.cache.storage.compute)
        for page in self.pages:
            page.last_use = self.tables.last_use
            page.pins -= 1
            if not page.pins:
                self.cache.storage.pinned.discard(page.key)
        self.tables.busy = False
        self.released = True

    def __enter__(self):
        self.wait()
        return self

    def __exit__(self, *_):
        self.release()


class HybridKVCache:
    """One slab, separate resident-block and hot-region ownership.

    Admission reserves credits for each layer's fixed hot region and two
    active tail blocks. Resident history may borrow all unused credits. Actual
    hot blocks are allocated only on the first transition to host-backed KV.
    This conservative single-process admission is simpler than vLLM HMA.
    """

    def __init__(self, config, hot: HotConfig):
        self.hot_config = hot
        self.hot_blocks = ceil(hot.hot_rows / config.page_size)
        self.tables = {}
        self._writing_key = None
        self.pending_clean = set()
        self.config = config
        self.allocator = DeviceAllocator(config)
        self.storage = HostStore(config, self.allocator)
        self.pool, self.device = self.allocator.storage, self.storage.device
        if config.host_pages * config.page_size >= 2**31:
            raise ValueError("host row tags require int32 addresses")
        self.row_stats = torch.zeros(3, dtype=torch.int64, device=self.device)
        self._host_pointer = None

    @property
    def index_bytes(self):
        return self.storage.index_bytes

    @property
    def budget(self):
        return self.allocator.budget

    @property
    def owners(self):
        return self.storage.owners

    def index_reservation_bytes(self, max_tokens, layers):
        return self.storage.index_reservation_bytes(max_tokens, layers) + layers * (
            ceil(max_tokens / self.config.page_size) * 8
            + self.hot_blocks * 4
            + self.hot_config.hot_rows * 12
        )

    def admission_blocks(self, layers):
        return self.hot_config.admission_blocks(self.config.page_size, layers)

    @property
    def transition_watermark(self):
        configured = self.hot_config.transition_watermark
        return max(self.hot_blocks, self.budget // 10) if configured is None else configured

    def _available_blocks(self):
        # Unpinned rows may be read again. Their temporary consumer leases are
        # independent of allocation pins and exclude them from this list.
        return sorted(i for i in self.allocator.free if i < self.budget) + [
            i
            for i in self.allocator.reclaimable
            if i < self.budget and not self.storage.pages[self.owners[i]].pins
        ]

    def _take_block(self, slot):
        if slot in self.allocator.reclaimable:
            self.allocator.reclaim(slot)
        return slot

    def _on_block_reused(self, slot):
        page = self.storage.pages[self.owners[slot]]
        assert not page.pins and page.host_committed_version == page.content_version
        self._detach(page)
        self.storage.stats["unpinned_block_reuses"] += 1

    def reserve(self, request_id, max_tokens, layers):
        self.collect()
        promised = sum(self.admission_blocks(r.layers) for r in self.storage.requests.values())
        if promised + self.admission_blocks(layers) > self.budget:
            raise CapacityError("hot-region + active-tail credits exhausted; defer admission")
        extra = self.index_reservation_bytes(
            max_tokens, layers
        ) - self.storage.index_reservation_bytes(max_tokens, layers)
        self.storage.reserve(request_id, max_tokens, layers, extra_index_bytes=extra)
        n = self.storage.requests[request_id].page_count
        for layer in range(layers):

            def empty(size):
                return torch.full((size,), -1, dtype=torch.int32, device=self.device)

            self.tables[request_id, layer] = LayerTables(
                empty(n),
                empty(n),
                empty(self.hot_blocks),
                empty(self.hot_config.hot_rows),
                torch.arange(self.hot_config.hot_rows, device=self.device, dtype=torch.int64),
            )

    def _prepare_append(self, page):
        self._writing_key = page.key
        tables = self.tables[page.key[:2]]
        tables.source[page.key[2]] = page.host_slot
        if page.valid_tokens:
            if page.slot is None:
                raise RuntimeError("active append tail lost its resident lease")
            page.pins += 1
            self.storage.pinned.add(page.key)
            self.storage.stats["resident_hits"] += 1
            self.storage.stats["append_resident_hits"] += 1
            return PageLease(self.storage, [page])
        slot = self._slot()
        self.storage.wait(self.storage.compute, self.allocator.slots[slot].fence)
        self._bind(page, slot, Location.RESIDENT)

    def _bind(self, page, slot, location):
        self.storage.bind(page, slot, location)
        self.tables[page.key[:2]].resident[page.key[2]] = slot

    def _detach(self, page):
        self.tables[page.key[:2]].resident[page.key[2]] = -1
        self.storage.detach(page)

    def _historical_pages(self, group=None):
        candidates = []
        for key in self.allocator.lru:
            if key[2] < 0:  # a hot BLOCK lease, not a logical KV page
                continue
            page = self.storage.pages[key]
            if group is not None and key[:2] != group:
                continue
            length = self.storage.requests[key[0]].lengths[key[1]]
            tail = (length - 1) // self.config.page_size
            if self._writing_key is not None and self._writing_key[:2] == key[:2]:
                tail = self._writing_key[2]
            if page.sealed and not page.pins and key[2] <= tail - ACTIVE_TAIL_PAGES:
                candidates.append(page)
        return candidates

    def _publish_clean(self):
        for key in list(self.pending_clean):
            page = self.storage.pages.get(key)
            if page is None:
                self.pending_clean.discard(key)
            elif page.host_ready is None or page.host_ready.query():
                page.host_committed_version = page.host_enqueued_version
                self.pending_clean.discard(key)
                self.storage.stats["host_mirror_completions"] += 1

    def _unpin_clean_pages(self, group):
        if not self.tables[group].blocks:
            return
        # Match upstream's tail-first release order; truly empty blocks still
        # take precedence over this last-resort readable free queue.
        for page in sorted(self._historical_pages(group), key=lambda p: -p.key[2]):
            if (
                page.host_committed_version == page.content_version
                and page.slot not in self.allocator.reclaimable
            ):
                self.allocator.unpin(page.slot, self._on_block_reused)
                self.storage.stats["clean_page_unpins"] += 1

    def _require_hot(self, group, reason):
        tables = self.tables[group]
        if not tables.hot_required:
            tables.hot_required, tables.trigger = True, reason
            self.storage.stats[f"hot_trigger_{reason}"] += 1

    def update_residency(self, request_id=None, layer=None, *, upcoming_allocation=False):
        """Nonblocking completion publication and persistent hot demand.

        A per-layer resident-window cap substitutes for vLLM's scheduler-derived
        in-flight token window. Before an append allocation, use projected free
        capacity so that hot demand is raised before the last free block is used.
        """
        self.storage.check_stream()
        self._publish_clean()
        groups = [
            g
            for g in self.tables
            if g[0] in self.storage.requests
            and (request_id is None or g[0] == request_id)
            and (layer is None or g[1] == layer)
        ]
        for group in groups:
            self._unpin_clean_pages(group)
        for group in groups:
            tables = self.tables[group]
            residents = sum(
                p.slot is not None for p in self.storage.pages.values() if p.key[:2] == group
            )
            if not residents:
                continue
            if not tables.hot_required:
                free = len(self._available_blocks()) - int(upcoming_allocation)
                if free < self.transition_watermark:
                    self._require_hot(group, "watermark")
                elif residents >= self.hot_config.resident_window_pages:
                    self._require_hot(group, "window")
            if tables.hot_required and not tables.blocks and not tables.busy:
                try:
                    self.transition_to_host(*group)
                except CapacityError:
                    # Allocation is retried at the next policy tick; a failed
                    # attempt must not discard resident mappings or hot demand.
                    pass
            self._unpin_clean_pages(group)

    def transition_to_host(self, request_id, layer):
        """Allocate the whole hot region atomically, then make history reclaimable."""
        self.storage.check_stream()
        group = request_id, layer
        tables = self.tables[group]
        if tables.blocks:
            return
        self._require_hot(group, "explicit")
        if tables.busy:
            raise CapacityError("cannot transition an active row lease")
        self._publish_clean()
        free = self._available_blocks()
        victims = [
            p
            for p in self._historical_pages(group)
            if p.host_committed_version == p.content_version and p.slot not in free
        ]
        needed = max(0, self.hot_blocks - len(free))
        if len(victims) < needed:
            raise CapacityError("cannot acquire complete hot region while blocks are leased")
        # All checks precede mutations. Fences preserve copies/last readers even
        # when a just-detached resident block becomes a hot block immediately.
        for page in victims[:needed]:
            free.append(page.slot)
            self.allocator.unpin(page.slot, self._on_block_reused)
            self.storage.stats["clean_page_unpins"] += 1
        blocks = free[: self.hot_blocks]
        for i, block in enumerate(blocks):
            self._take_block(block)
            self.storage.wait(self.storage.transfer, self.allocator.slots[block].fence)
            self.allocator.bind(block, (request_id, layer, -i - 1))
        tables.blocks = blocks
        tables.hot.copy_(torch.tensor(blocks, dtype=torch.int32, device=self.device))
        tables.ready = self.storage.event(self.storage.transfer)
        self.storage.stats["hot_region_allocations"] += 1
        self._unpin_clean_pages(group)

    def _slot(self):
        while True:
            self.update_residency(upcoming_allocation=True)
            free = self._available_blocks()
            if free:
                return self._take_block(free[0])
            candidates = self._historical_pages()
            counts = Counter(p.key[:2] for p in candidates)
            for group, count in counts.most_common():
                if count >= self.hot_blocks + 1 and not self.tables[group].blocks:
                    self._require_hot(group, "pressure")
                    try:
                        self.transition_to_host(*group)
                    except CapacityError:
                        continue
                    free = self._available_blocks()
                    if free:
                        return self._take_block(free[0])
            pending = [
                p
                for p in candidates
                if p.host_committed_version < p.content_version and p.host_ready is not None
            ]
            if not pending:
                raise CapacityError("no reclaimable resident block; tails/readers are protected")
            # Synchronous teaching runner cannot suspend midway through prefill.
            # Wait only a useful mirror event, never device/stream synchronize.
            pending[0].host_ready.synchronize()
            self.storage.stats["pressure_mirror_waits"] += 1

    def reclaim_history(self, request_id, layer, *, evict=False):
        """Make completed historical pages allocatable while still readable.

        evict=True is an explicit diagnostic hook that also revokes mappings;
        normal pressure uses the allocator's on-reuse callback instead.
        """
        self.transition_to_host(request_id, layer)
        self._unpin_clean_pages((request_id, layer))
        if evict:
            for page in self._historical_pages((request_id, layer)):
                if page.slot in self.allocator.reclaimable:
                    self._take_block(page.slot)

    def _after_write(self, page):
        self.pending_clean.add(page.key)
        self.update_residency(*page.key[:2])

    def write(self, request_id, layer, k, v):
        if self.tables[request_id, layer].busy:
            raise RuntimeError("cannot append during a row lease")
        try:
            self.storage.append(request_id, layer, k, v, self._prepare_append, self._after_write)
        finally:
            self._writing_key = None

    def attention(self, query, request_id, layer, selection, backend="auto"):
        from .attention import row_attention

        length = self.storage.requests[request_id].lengths[layer]
        if selection.length != length or selection.page_size != self.config.page_size:
            raise ValueError("selection has stale length or wrong page size")
        if len(selection.positions) != query.shape[0]:
            raise ValueError("selection must have one row per query head")
        return row_attention(
            query,
            self,
            request_id,
            layer,
            LogicalTopK.from_selection(selection, query.device),
            backend,
        )

    def resolve(self, request_id, layer, topk: LogicalTopK):
        self.storage.check_stream()
        topk.validate(self.device, self.hot_config.max_union_rows)
        tables = self.tables[request_id, layer]
        if tables.busy:
            raise RuntimeError("hot contents already leased by this layer")
        length = self.storage.requests[request_id].lengths[layer]
        if length < 1:
            raise ValueError("cannot resolve an empty history")
        # Conservative pinning avoids reading GPU-selected page IDs on CPU.
        # Other requests/layers still progress; only this layer's residents pin.
        pages = [
            p
            for p in self.storage.pages.values()
            if p.key[:2] == (request_id, layer) and p.slot is not None
        ]
        for page in pages:
            page.pins += 1
            self.storage.pinned.add(page.key)
        tables.busy = True
        try:
            self.storage.wait(self.storage.compute, tables.ready)
            plan = resolve_rows(
                topk,
                tables.source,
                tables.resident,
                tables.hot,
                tables.tags,
                tables.lru,
                length,
                self.config.page_size,
            )
            self.row_stats.add_(plan.counts)
            resolved = self.storage.event(self.storage.compute)
            self.storage.wait(self.storage.transfer, resolved)
            self.storage.wait(self.storage.transfer, tables.last_use)
            # Append mirrors and every earlier spill already precede us on
            # this single transfer stream. The gather reads only valid host KV.
            with self.storage.context(self.storage.transfer):
                for host, device in zip(
                    plan.miss_host_rows.tolist(), plan.miss_device_rows.tolist()
                ):
                    if host >= 0:
                        hb, ho = divmod(host, self.config.page_size)
                        db, do = divmod(device, self.config.page_size)
                        self.pool[db, :, do].copy_(self.storage.host[hb, :, ho])
            tables.ready = self.storage.event(self.storage.transfer)
            # Plan tensors are produced on compute but consumed on transfer.
            if self.storage.cuda:
                plan.miss_host_rows.record_stream(self.storage.transfer)
                plan.miss_device_rows.record_stream(self.storage.transfer)
            return RowLease(self, tables, plan, pages)
        except Exception:
            for page in pages:
                page.pins -= 1
                if not page.pins:
                    self.storage.pinned.discard(page.key)
            tables.busy = False
            raise

    def release_request(self, request_id, wait=True):
        groups = [t for (name, _), t in self.tables.items() if name == request_id]
        if any(t.busy for t in groups) or any(
            self.storage.pages[key].pins for key in self.storage.requests[request_id].keys
        ):
            raise RuntimeError("cannot release a request with active row leases")
        for tables in groups:
            self.storage.wait(self.storage.transfer, tables.ready)
            self.storage.wait(self.storage.transfer, tables.last_use)
            fence = self.storage.event(self.storage.transfer)
            for block in tables.blocks:
                self.allocator.release(block, fence)
        for key in self.storage.requests[request_id].keys:
            page = self.storage.pages[key]
            if page.slot is not None:
                self._detach(page)
        self.storage.retire(request_id, keepalive=groups)
        for key in [key for key in self.tables if key[0] == request_id]:
            del self.tables[key]
        self.collect(wait=wait)

    def collect(self, wait=False):
        self.storage.collect(wait)
        self.update_residency()

    def page_state(self, key):
        state = self.storage.page_state(key)
        page = self.storage.pages[key]
        unpinned = page.slot in self.allocator.reclaimable
        clean = page.host_committed_version == page.content_version
        state.update(
            allocation_pinned=page.slot is not None and not unpinned,
            residency="host_only"
            if page.slot is None
            else ("clean_unpinned" if unpinned else "clean_pinned" if clean else "dirty"),
        )
        return state

    def set_budget(self, pages):
        if pages != self.budget:
            raise CapacityError(
                "hybrid block budget is fixed; hot leases need readmission to resize"
            )

    def close(self):
        # Pinned allocators cannot infer Triton SM readers from torch.copy_.
        # Drain at teardown, never in the normal decode path.
        self.storage.synchronize()

    def __del__(self):
        with suppress(RuntimeError, AttributeError, TypeError):
            if getattr(getattr(self, "storage", None), "cuda", False):
                self.close()

    def reset_stats(self):
        self.storage.stats.clear()
        self.row_stats.zero_()

    def snapshot(self):
        result = self.storage.snapshot()
        resident, hot, misses = self.row_stats.cpu().tolist()
        result.update(
            cache_layout="hybrid_rows",
            eviction_policy="resident-reclaim + per-layer row-LRU",
            max_union_rows=self.hot_config.max_union_rows,
            hot_rows=self.hot_config.hot_rows,
            active_tail_pages=ACTIVE_TAIL_PAGES,
            resident_window_pages=self.hot_config.resident_window_pages,
            transition_watermark=self.transition_watermark,
            empty_device_blocks=len(self.allocator.free),
            readable_unpinned_blocks=len(self.allocator.reclaimable),
            allocatable_device_blocks=len(self._available_blocks()),
            pending_clean_pages=len(self.pending_clean),
            hot_pending_groups=sum(t.hot_required and not t.blocks for t in self.tables.values()),
            hot_leased_blocks=sum(len(t.blocks) for t in self.tables.values()),
            admission_block_credits=sum(
                self.admission_blocks(r.layers) for r in self.storage.requests.values()
            ),
            row_stats={
                "resident_hits": resident,
                "hot_hits": hot,
                "misses": misses,
                "h2d_bytes": misses * self.storage.page_bytes // self.config.page_size,
            },
        )
        return result

    def assert_invariants(self):
        self.storage.assert_invariants()
        for name, request in self.storage.requests.items():
            assert request.extra_index_bytes == sum(
                t.nbytes for group, t in self.tables.items() if group[0] == name
            )
        assert not (self.allocator.free & self.allocator.reclaimable.keys())
        for slot in self.allocator.reclaimable:
            page = self.storage.pages[self.owners[slot]]
            assert page.slot == slot and page.sealed
            assert page.host_committed_version == page.content_version
            assert self.tables[page.key[:2]].blocks
            length = self.storage.requests[page.key[0]].lengths[page.key[1]]
            assert page.key[2] <= (length - 1) // self.config.page_size - ACTIVE_TAIL_PAGES
        assert (
            sum(self.admission_blocks(r.layers) for r in self.storage.requests.values())
            <= self.budget
        )
        for group, tables in self.tables.items():
            assert len(tables.blocks) in (0, self.hot_blocks)
            assert sorted(tables.lru.tolist()) == list(range(self.hot_config.hot_rows))
            tags = [tag for tag in tables.tags.tolist() if tag >= 0]
            assert len(tags) == len(set(tags))
            if not tables.blocks:
                assert not tags
            else:
                assert tables.hot_required
            for i, block in enumerate(tables.blocks):
                assert self.owners[block] == (*group, -i - 1)
            for index, host in enumerate(tables.source.tolist()):
                page = self.storage.pages.get((*group, index))
                assert host == (-1 if page is None else page.host_slot)
                slot = tables.resident[index].item()
                assert slot == (-1 if page is None or page.slot is None else page.slot)
                if page is not None and page.slot is None:
                    assert tables.blocks
