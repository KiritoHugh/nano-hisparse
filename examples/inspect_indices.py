"""A deterministic address-translation example, using the actual row resolver."""

import argparse
import json
from pathlib import Path

import torch

from nano_hisparse.row_index import LogicalTopK, resolve_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    def tensor(values):
        return torch.tensor(values, dtype=torch.int64, device=args.device)

    # Constructed tables, not a captured model trace. GPU blocks are fragmented.
    source = tensor([5, 2, 9, 4])
    resident = tensor([-1, 7, -1, 3])
    hot = tensor([6, 1])
    tags = tensor([-1, -1, -1, -1, -1, 22, -1, -1])
    lru = tensor([0, 1, 2, 3, 4, 6, 7, 5])
    logical = tensor([9, 2, 6, 9, 14, -1])
    before = tags.tolist()
    plan = resolve_rows(LogicalTopK(logical), source, resident, hot, tags, lru, 16, 4)
    assert plan.physical_rows.tolist() == [24, 5, 30, -1, 14, -1]
    assert plan.counts.tolist() == [2, 1, 1]
    assert plan.miss_host_rows.tolist() == [37, -1, -1, -1, -1, -1]
    assert plan.miss_device_rows.tolist() == [24, -1, -1, -1, -1, -1]
    assert tags[0].item() == 37 and tags[5].item() == 22
    report = {
        "device": args.device,
        "source": "constructed tables; real resolve_rows execution, no KV gather",
        "page_size": 4,
        "logical_tokens": logical.tolist(),
        "source_table": source.tolist(),
        "resident_table": resident.tolist(),
        "hot_block_table": hot.tolist(),
        "tags_before": before,
        "tags_after": tags.tolist(),
        "physical_rows": plan.physical_rows.tolist(),
        "miss_host_rows": plan.miss_host_rows.tolist(),
        "miss_device_rows": plan.miss_device_rows.tolist(),
        "counts": dict(zip(("resident_hits", "hot_hits", "misses"), plan.counts.tolist())),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
