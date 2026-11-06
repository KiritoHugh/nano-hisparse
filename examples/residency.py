"""Watch a resident page remain readable until its block is actually reused."""

import argparse

import torch

from nano_hisparse import CacheConfig, HotConfig, HybridKVCache, LogicalTopK
from nano_hisparse.attention import row_attention
from nano_hisparse.selection import dense_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    cache = HybridKVCache(
        CacheConfig(
            gpu_pages=7, host_pages=64, page_size=4, kv_heads=1, head_dim=8, device=args.device
        ),
        HotConfig(4, 4, transition_watermark=0),
    )
    for request in ("a", "b"):
        cache.reserve(request, 32, 1)

    def write(name, tokens):
        k, v = [torch.randn(tokens, 1, 8, device=cache.device) for _ in range(2)]
        cache.write(name, 0, k, v)
        return k, v

    def show(stage):
        cache.storage.synchronize()  # diagnostics only; not normal decode
        state = cache.page_state(("a", 0, 0))
        slot = cache.storage.pages["a", 0, 0].slot
        misses = cache.snapshot()["row_stats"]["misses"]
        print(f"{stage:26} {state['residency']:16} slot={str(slot):4} misses={misses}")

    k, v = write("a", 12)
    show("Initial resident page")
    cache.reclaim_history("a", 0)
    show("Made reclaimable")
    query = torch.randn(2, 8, device=cache.device)
    selected = LogicalTopK(torch.tensor([0], dtype=torch.int32, device=cache.device))
    expected = dense_attention(query, k[:1], v[:1])
    torch.testing.assert_close(row_attention(query, cache, "a", 0, selected), expected)
    show("Read before reuse")
    write("b", 12)
    show("Free blocks consumed")
    write("b", 1)
    show("Old block reused")
    torch.testing.assert_close(row_attention(query, cache, "a", 0, selected), expected)
    show("Missing row loaded")
    for request in ("a", "b"):
        cache.release_request(request)
    cache.assert_invariants()
    cache.close()


if __name__ == "__main__":
    main()
