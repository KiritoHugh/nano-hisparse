"""Use the Python API with Qwen2.5. Requires nano-hisparse[models]."""

from math import ceil

import torch
from transformers import AutoTokenizer

from nano_hisparse import CacheConfig, HotConfig, HybridKVCache
from nano_hisparse.engine import Engine
from nano_hisparse.model import Model


def main():
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Explain why autoregressive models cache keys and values."}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    model = Model.from_pretrained(name, device, dtype)
    cfg = model.config
    max_new_tokens, page_size, top_pages = 64, 16, 4
    maximum = len(tokens) + max_new_tokens
    # One page selection per query token, shared by all attention heads.
    selected_rows = min(maximum, top_pages * page_size)
    hot = HotConfig(selected_rows)
    cache = HybridKVCache(
        CacheConfig(
            gpu_pages=hot.admission_blocks(page_size, cfg.num_hidden_layers),
            host_pages=cfg.num_hidden_layers * ceil(maximum / page_size),
            page_size=page_size,
            kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            device=device,
            dtype=dtype,
        ),
        hot,
    )
    try:
        engine = Engine(model, cache, top_k=top_pages, eos_token_id=tokenizer.eos_token_id)
        engine.add_request("example", tokens, max_new_tokens)
        output = engine.run()["example"]
        print(tokenizer.decode(output, skip_special_tokens=True))
    finally:
        cache.close()


if __name__ == "__main__":
    main()
