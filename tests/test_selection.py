import pytest
import torch

from nano_hisparse.row_index import LogicalTopK, resolve_rows
from nano_hisparse.selection import Selection, quest_selection


def test_sparse_masks_remain_per_head_and_reject_noncausal_positions():
    selected = Selection.from_tokens([[3, 1, 1, -1], [0]], 4, 2, 2)
    topk = LogicalTopK.from_selection(selected, "cpu")
    assert topk.positions.tolist() == [0, 1, 3]
    assert topk.head_columns.tolist() == [[1, 2], [0, -1]]
    for rows in [[4], [-2], [-1], [1.2], []]:
        with pytest.raises((ValueError, TypeError)):
            Selection.from_tokens(rows, 4, 2, 2)


def test_quest_bounds_preserve_a_page_whose_mean_cancels():
    low = torch.tensor([[[-10.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.0]]])
    high = torch.tensor([[[10.0, 0.0]], [[1.0, 0.0]], [[0.0, 0.0]]])
    selection = quest_selection(torch.tensor([[1.0, 0.0]]), low, high, 5, 2, 2, keep_sink=False)
    assert selection.pages == [0, 2]


def test_resolver_protects_late_hits_deduplicates_and_maps_fragmented_blocks():
    def tensor(values):
        return torch.tensor(values, dtype=torch.int64)

    source, resident = tensor([5, 2, 9, 4]), tensor([-1, 7, -1, 3])
    hot = tensor([6, 1])
    tags = tensor([-1, -1, -1, -1, -1, 22, -1, -1])
    lru = tensor([0, 1, 2, 3, 4, 6, 7, 5])
    selection = LogicalTopK(tensor([9, 2, 6, 9, 14, -1]))
    plan = resolve_rows(selection, source, resident, hot, tags, lru, 16, 4)
    assert plan.physical_rows.tolist() == [24, 5, 30, -1, 14, -1]
    assert plan.counts.tolist() == [2, 1, 1]
    assert plan.miss_host_rows.tolist() == [37, -1, -1, -1, -1, -1]
    assert plan.miss_device_rows.tolist() == [24, -1, -1, -1, -1, -1]
    assert tags[0] == 37 and tags[5] == 22
