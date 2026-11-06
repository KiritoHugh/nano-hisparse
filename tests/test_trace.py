import copy

import pytest
import torch

from nano_hisparse.model import Model, ModelConfig
from nano_hisparse.trace import (
    capture_trace,
    load_trace,
    replay_trace,
    save_trace,
    synthetic_trace,
    token_topk,
    validate_trace,
)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "pattern,capacity,hits,misses,row_j,page_j,density",
    [
        ("reuse", 4, 28, 4, 1, 1, 0.25),
        ("alternating", 4, 0, 32, 0, 1, 0.25),
        ("alternating", 8, 24, 8, 0, 1, 0.25),
        ("streaming", 8, 0, 32, 0, 6 / 7, 0.25),
        ("packed", 4, 28, 4, 1, 1, 1),
    ],
)
def test_row_locality_counts(device, pattern, capacity, hits, misses, row_j, page_j, density):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    report = replay_trace(synthetic_trace(pattern), capacity, device)
    summary = report["summary"]
    assert summary["resident_hits"] == 0
    assert (summary["hot_hits"], summary["misses"]) == (hits, misses)
    assert summary["h2d_bytes"] == misses * 2 * 1 * 8 * 4
    assert summary["mean_row_jaccard"] == pytest.approx(row_j)
    assert summary["mean_page_jaccard"] == pytest.approx(page_j)
    assert summary["mean_page_density"] == density
    assert (
        summary["mean_reuse_gap_steps"]
        == {
            "reuse": 1,
            "alternating": 2,
            "streaming": None,
            "packed": 1,
        }[pattern]
    )
    for event in report["events"]:
        # Heads overlap: five head/token pairs cause only four row accesses.
        assert sum(map(len, event["positions"])) == 5
        assert event["hot_hits"] + event["misses"] == 4
        assert set(event["union_rows"]) <= set(event["hot_slots"])
    if pattern == "alternating" and capacity == 8:
        assert set(report["events"][1]["hot_slots"]) == {0, 1, 4, 5, 8, 9, 12, 13}


def test_token_ranking_selects_rows_without_expanding_pages():
    keys = torch.zeros(16, 1, 2)
    keys[[0, 4, 8, 12], 0, 0] = torch.tensor([4.0, 3, 2, 1])
    keys[[1, 5, 9, 13], 0, 1] = torch.tensor([4.0, 3, 2, 1])
    assert token_topk(torch.eye(2), keys, 4) == [[0, 4, 8, 12], [1, 5, 9, 13]]


def test_trace_roundtrip_and_invalid_input(tmp_path):
    trace = synthetic_trace("alternating")
    path = tmp_path / "trace.pt"
    save_trace(trace, path)
    loaded = load_trace(path)
    assert replay_trace(loaded, 8) == replay_trace(trace, 8)
    invalid = copy.deepcopy(trace)
    invalid["events"][0]["positions"][0] = [65]
    with pytest.raises(ValueError, match="noncausal"):
        validate_trace(invalid)
    invalid = copy.deepcopy(trace)
    invalid["events"][0]["k"] = torch.zeros(2, 1, 8)
    with pytest.raises(ValueError, match="incompatible"):
        validate_trace(invalid)
    with pytest.raises(ValueError):
        replay_trace(trace, 3)


def test_capture_records_real_queries_and_preserves_dense_trajectory(tmp_path):
    torch.manual_seed(3)
    torch.set_num_threads(1)
    model = Model(
        ModelConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
        )
    ).eval()
    tokens = list(range(1, 34))
    trace = capture_trace(model, tokens, steps=3, token_budget=2, layer=1, page_size=4)
    # Independently recompute a dense forward at every step, recording the
    # projected Q/K. An accidental sparse teacher changes subsequent queries.
    observed = []
    project = model._project

    def record(layer, x, start):
        q, k, v = project(layer, x, start)
        if layer is model.model.layers[1]:
            observed.append((q[-1], k, v))
        return q, k, v

    model._project = record
    history = list(tokens)
    logits = model.prefill(history)
    observed.clear()
    for _ in range(3):
        history.append(int(logits.argmax()))
        logits = model.prefill(history)
    for event, (q, k, v) in zip(trace["events"], observed, strict=True):
        torch.testing.assert_close(event["q"], q, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(event["k"], k[-1:], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(event["v"], v[-1:], atol=2e-6, rtol=2e-5)
        assert event["positions"] == token_topk(q, k, 2)
    save_trace(trace, tmp_path / "model.pt")
    report = replay_trace(load_trace(tmp_path / "model.pt"))
    assert report["summary"]["misses"] > 0
    assert report["max_union_rows"] <= 2 * 4
    assert report["source"]["kind"] == "dense_teacher_token_oracle"
