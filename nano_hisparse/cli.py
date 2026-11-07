"""Run a local mechanics demo or generate text with Qwen weights."""

import argparse
import json
from math import ceil
from pathlib import Path

import torch

from .config import CacheConfig
from .engine import Engine
from .hybrid import HotConfig, HybridKVCache
from .model import Model


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _top_k(value):
    return None if value == "all" else _positive(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "generate"):
        command = commands.add_parser(name)
        command.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
        command.add_argument(
            "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
        )
        command.add_argument("--gpu-pages", type=_positive)
        command.add_argument("--host-pages", type=_positive)
        command.add_argument("--page-size", type=_positive, default=16)
        command.add_argument("--hot-rows", type=_positive)
        command.add_argument(
            "--max-selected-rows",
            "--max-union-rows",
            dest="max_selected_rows",
            type=_positive,
            help="maximum distinct selected token rows per query (legacy alias: --max-union-rows)",
        )
        command.add_argument("--index-budget-mib", type=_positive, default=64)
        command.add_argument("--backend", choices=["auto", "torch", "triton"], default="auto")
        command.add_argument(
            "--top-k",
            type=_top_k,
            default=2 if name == "demo" else None,
            help="shared selected pages per query token, or all",
        )
        command.add_argument("--max-new-tokens", type=_positive, default=12)
        command.add_argument("--output", type=Path)
        if name == "demo":
            command.add_argument("--requests", type=_positive, default=3)
            command.add_argument("--prompt-tokens", type=_positive, default=257)
        else:
            command.add_argument(
                "--model", required=True, help="Qwen2/2.5 directory or HF repository"
            )
            command.add_argument("--prompt", default="Explain how a KV cache works.")
            command.add_argument("--chat", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)
    tokenizer = None
    if args.command == "generate":
        try:
            from transformers import AutoTokenizer
        except ImportError:
            parser.error("install nano-hisparse[models] to load pretrained weights")
        model = Model.from_pretrained(args.model, args.device, dtype)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompt = (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            if args.chat
            else tokenizer.encode(args.prompt)
        )
        prompts = [prompt]
    else:
        model = Model().eval().to(device=args.device, dtype=dtype)
        prompts = [
            torch.randint(0, model.config.vocab_size, (args.prompt_tokens + i * 7,)).tolist()
            for i in range(args.requests)
        ]

    cfg = model.config
    maximum = max(map(len, prompts)) + args.max_new_tokens
    selected_rows = maximum if args.top_k is None else min(maximum, args.top_k * args.page_size)
    hot = HotConfig(
        selected_rows if args.max_selected_rows is None else args.max_selected_rows, args.hot_rows
    )
    cache = HybridKVCache(
        CacheConfig(
            gpu_pages=args.gpu_pages or hot.admission_blocks(args.page_size, cfg.num_hidden_layers),
            host_pages=args.host_pages
            or cfg.num_hidden_layers * ceil(maximum / args.page_size) * len(prompts),
            page_size=args.page_size,
            kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            device=args.device,
            dtype=dtype,
            index_budget_bytes=args.index_budget_mib * 1024 * 1024,
        ),
        hot,
    )
    try:
        engine = Engine(
            model, cache, args.top_k, args.backend, tokenizer.eos_token_id if tokenizer else None
        )
        for i, prompt in enumerate(prompts):
            engine.add_request(str(i), prompt, args.max_new_tokens)
        outputs = engine.run()
        cache.assert_invariants()
        report = {
            "model": args.model if tokenizer else "random decoder (mechanics demo)",
            "selection": "full"
            if args.top_k is None
            else f"quest: {args.top_k} shared pages/query",
            "outputs": {
                name: tokenizer.decode(tokens, skip_special_tokens=True) if tokenizer else tokens
                for name, tokens in outputs.items()
            },
            "requests": {
                name: {
                    "prefill_count": r.prefill_count,
                    "decode_count": r.decode_count,
                    "generated_tokens": len(r.generated),
                }
                for name, r in engine.requests.items()
            },
            "cache": cache.snapshot(),
        }
    finally:
        cache.close()
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
