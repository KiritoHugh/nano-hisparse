"""Logical selection, address translation and hot content tags are separate.

The resolver uses fixed-size Torch tensors on their original device. There is
no nonzero(), item(), tolist() or device-to-host decision in resolve_rows.
This is an inspectable GPU implementation, not vLLM's fused CUDA resolver.
"""

from dataclasses import dataclass

import torch

from .selection import Selection


@dataclass(frozen=True)
class LogicalTopK:
    positions: torch.Tensor  # shared [K] token positions, or the union for GQA
    head_columns: torch.Tensor | None = None  # [heads, K_head], -1 padding

    @classmethod
    def from_selection(cls, selection: Selection, device):
        union = sorted({p for row in selection.positions for p in row})
        columns = {token: i for i, token in enumerate(union)}
        width = max(map(len, selection.positions))
        rows = [
            [columns[p] for p in row] + [-1] * (width - len(row)) for row in selection.positions
        ]
        return cls(
            torch.tensor(union, dtype=torch.int32, device=device),
            torch.tensor(rows, dtype=torch.int64, device=device),
        )

    def validate(self, device, maximum):
        if (
            self.positions.ndim != 1
            or self.positions.numel() < 1
            or self.positions.dtype not in (torch.int32, torch.int64)
            or self.positions.device != device
        ):
            raise ValueError("logical top-k must be a nonempty device integer vector")
        if self.positions.numel() > maximum:
            raise ValueError("top-k width/union exceeds the admitted max_union_rows")
        if self.head_columns is not None and (
            self.head_columns.ndim != 2
            or self.head_columns.device != device
            or self.head_columns.dtype != torch.int64
        ):
            raise ValueError("head columns must be a device int64 matrix")


@dataclass
class RowPlan:
    physical_rows: torch.Tensor  # shared [K] or per-head [H, K], -1 padding
    miss_host_rows: torch.Tensor  # fixed-width compact prefix; -1 after misses
    miss_device_rows: torch.Tensor
    counts: torch.Tensor  # unique resident hits, hot hits, misses


def resolve_rows(topk, source_table, resident_table, hot_table, tags, lru, length, page_size):
    """Resident lookup first; protect ALL selected hot hits before any eviction.

    Tags identify host physical rows, never logical page numbers. Hot logical
    slot numbers are translated through a separate, possibly fragmented block
    table. Duplicates share a destination; invalid/padded inputs produce -1.
    """
    raw = topk.positions.long()
    valid_raw = (raw >= 0) & (raw < length)
    normalized = torch.where(valid_raw, raw, length)
    logical = normalized.sort().values
    unique = logical < length
    unique[1:] = unique[1:] & (logical[1:] != logical[:-1])
    page = (logical // page_size).clamp(max=source_table.numel() - 1)
    offset = logical % page_size
    source = source_table[page].long()
    resident = resident_table[page].long()
    host = source * page_size + offset
    valid = unique & (source >= 0)
    resident_hit = valid & (resident >= 0)
    needs_hot = valid & ~resident_hit
    wanted = torch.where(needs_hot, host, -1)
    size = tags.numel()

    # Sort/search instead of constructing a quadratic [top-k, hot-size] matrix.
    sorted_tags, tag_slots = tags.sort()
    lookup = torch.searchsorted(sorted_tags, host.to(tags.dtype)).clamp(max=size - 1)
    hot_hit = needs_hot & (sorted_tags[lookup].long() == host)
    existing_slots = tag_slots[lookup]
    sorted_wanted = wanted.sort().values
    lookup_wanted = torch.searchsorted(sorted_wanted, tags.long()).clamp(max=raw.numel() - 1)
    protected = (tags >= 0) & (sorted_wanted[lookup_wanted] == tags)

    # Least-recent non-hit slots first. A hit late in the input cannot be
    # overwritten by a miss early in the input (including duplicate inputs).
    rank = torch.arange(size, device=tags.device)
    victims = lru[torch.argsort(protected[lru].long() * size + rank)]
    miss = needs_hot & ~hot_hit
    miss_rank = miss.long().cumsum(0) - 1
    assigned = victims[miss_rank.clamp(min=0, max=size - 1)]
    slots = torch.where(hot_hit, existing_slots, assigned)
    hot_blocks = hot_table[slots // page_size].long()
    hot_physical = hot_blocks * page_size + slots % page_size
    physical = torch.where(resident_hit, resident * page_size + offset, hot_physical)
    physical = torch.where(valid, physical, -1)

    # A sentinel destination avoids data-dependent shapes and CPU readback.
    updates = torch.full((size + 1,), -1, dtype=tags.dtype, device=tags.device)
    updates.scatter_(
        0, torch.where(miss, assigned, size), torch.where(miss, host, -1).to(tags.dtype)
    )
    inserted = updates[:size] >= 0
    tags.copy_(torch.where(inserted, updates[:size], tags))
    category = inserted.long() + 2 * protected.long()
    lru.copy_(lru[torch.argsort(category[lru] * size + rank)])

    n = raw.numel()
    host_plan = torch.full((n + 1,), -1, dtype=torch.int64, device=raw.device)
    device_plan = torch.full_like(host_plan, -1)
    destinations = torch.where(miss, miss_rank, n)
    host_plan.scatter_(0, destinations, torch.where(miss, host, -1))
    device_plan.scatter_(0, destinations, torch.where(miss, hot_physical, -1))
    first = torch.searchsorted(logical, normalized).clamp(max=n - 1)
    input_rank = torch.arange(n, device=raw.device)
    first_input = torch.full((n,), n, dtype=torch.int64, device=raw.device)
    first_input.scatter_reduce_(0, first, input_rank, reduce="amin")
    keep = valid_raw & (input_rank == first_input[first])
    result = torch.where(keep, physical[first], -1)
    if topk.head_columns is not None:
        columns = topk.head_columns
        result = torch.where(
            (columns >= 0) & (columns < n), result[columns.clamp(min=0, max=n - 1)], -1
        )
    return RowPlan(
        result,
        host_plan[:n],
        device_plan[:n],
        torch.stack((resident_hit.sum(), hot_hit.sum(), miss.sum())),
    )
