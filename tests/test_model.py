import pytest
import torch

from nano_hisparse.config import CacheConfig
from nano_hisparse.hybrid import HotConfig, HybridKVCache
from nano_hisparse.model import Model, ModelConfig


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_incremental_logits_match_full_causal_forward_after_offload(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(31)
    torch.set_num_threads(1)
    model = (
        Model(
            ModelConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=64,
            )
        )
        .eval()
        .to(device)
    )
    hot = HotConfig(24, 24)
    cache = HybridKVCache(
        CacheConfig(
            gpu_pages=hot.admission_blocks(4, 2),
            host_pages=64,
            page_size=4,
            kv_heads=2,
            head_dim=8,
            device=device,
        ),
        hot,
    )
    history = list(range(1, 18))
    cache.reserve("a", 24, 2)
    torch.testing.assert_close(model.prefill(history, cache, "a"), model.prefill(history))
    cache.storage.synchronize()
    for layer in range(2):
        cache.reclaim_history("a", layer, evict=True)
    for token in (18, 19, 20):
        history.append(token)
        actual = model.decode(token, "a", cache)
        torch.testing.assert_close(actual, model.prefill(history), atol=3e-6, rtol=3e-5)
    assert cache.snapshot()["row_stats"]["misses"] > 0
    with pytest.raises(ValueError, match="exactly once"):
        model.prefill(history, cache, "a")
    cache.release_request("a")
    cache.assert_invariants()


def test_qwen2_huggingface_parity(tmp_path):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(8)
    cfg = ModelConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
    )
    hf_cfg = transformers.Qwen2Config(**vars(cfg), attention_dropout=0, tie_word_embeddings=False)
    hf = transformers.Qwen2ForCausalLM(hf_cfg).eval()
    model = Model(cfg).eval()
    model.load_state_dict(hf.state_dict(), strict=True)
    # Also exercise the actual pretrained adapter, including config migrations.
    hf.save_pretrained(tmp_path)
    loaded = Model.from_pretrained(str(tmp_path), device="cpu", dtype=torch.float32)
    tokens = [1, 7, 3, 4, 9, 11, 17]
    with torch.inference_mode():
        expected = hf(torch.tensor([tokens])).logits[0, -1]
    torch.testing.assert_close(model.prefill(tokens), expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(loaded.prefill(tokens), expected, atol=2e-6, rtol=2e-5)
    cache = HybridKVCache(
        CacheConfig(gpu_pages=12, host_pages=16, page_size=3, kv_heads=2, head_dim=8, device="cpu"),
        HotConfig(12, 12),
    )
    cache.reserve("a", 12, 2)
    model.prefill(tokens, cache, "a")
    for token in [12, 19, 5]:
        tokens.append(token)
        actual = model.decode(token, "a", cache)
        with torch.inference_mode():
            expected = hf(torch.tensor([tokens])).logits[0, -1]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    cache.release_request("a")
    cache.assert_invariants()
