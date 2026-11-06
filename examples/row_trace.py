"""Generate, capture and replay token-row accesses through the actual KV cache."""

import argparse
import json
from pathlib import Path

import torch

from nano_hisparse.trace import capture_trace, load_trace, replay_trace, save_trace, synthetic_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="compare controlled locality at H=4 and H=8")
    capture = sub.add_parser("capture", help="record one dense-teacher layer with token QK top-k")
    replay = sub.add_parser("replay", help="replay a saved trace and inspect hot capacity")
    for command in (demo, capture, replay):
        command.add_argument("--device", default="cpu")
    for command in (demo, replay):
        command.add_argument(
            "--output", type=Path, help="write detailed JSON, including per-step rows"
        )
    capture.add_argument(
        "--model", help="optional Qwen2/2.5 model; default is a tiny random decoder"
    )
    capture.add_argument("--prompt-file", type=Path)
    capture.add_argument("--max-context", type=int, default=256)
    capture.add_argument("--steps", type=int, default=8)
    capture.add_argument("--tokens-per-head", type=int, default=8)
    capture.add_argument("--layer", type=int, default=0)
    capture.add_argument(
        "--output", type=Path, required=True, help="save frozen CPU Q/K/V and token IDs"
    )
    replay.add_argument("trace", type=Path)
    replay.add_argument("--hot-rows", type=int, help="default: observed maximum token union")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if args.command == "capture":
        from nano_hisparse.model import Model

        if args.max_context < 1:
            parser.error("max-context must be positive")
        if args.model:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model)
            text = (
                args.prompt_file.read_text()
                if args.prompt_file
                else "Key value caches store history. " * 64
            )
            tokens = tokenizer.encode(text)[: args.max_context]
            dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
            model = Model.from_pretrained(args.model, args.device, dtype)
        else:
            if args.prompt_file:
                parser.error("prompt-file requires a pretrained model/tokenizer")
            model = Model().eval().to(args.device)
            tokens = torch.randint(0, model.config.vocab_size, (args.max_context,)).tolist()
        revision = getattr(getattr(model.model, "config", None), "_commit_hash", None)
        trace = capture_trace(
            model,
            tokens,
            args.steps,
            args.tokens_per_head,
            args.layer,
            source={
                "model": args.model or "random decoder",
                "revision": revision,
                "seed": 42,
                "context_truncation": "prefix",
            },
        )
        save_trace(trace, args.output)
        print(
            json.dumps(
                {
                    "trace": str(args.output),
                    "events": len(trace["events"]),
                    "source": trace["source"],
                },
                indent=2,
            )
        )
        return
    if args.command == "demo":
        reports = [
            replay_trace(synthetic_trace(pattern), hot, args.device)
            for pattern in ("reuse", "alternating", "streaming", "packed")
            for hot in (4, 8)
        ]
    else:
        reports = [replay_trace(load_trace(args.trace), args.hot_rows, args.device)]
    print("pattern          H   resident  hot  miss   H2D bytes  row J   page J  density")
    for report in reports:
        s = report["summary"]
        name = report["source"].get("pattern", "model trace")
        row_j = "n/a" if s["mean_row_jaccard"] is None else f"{s['mean_row_jaccard']:.3f}"
        page_j = "n/a" if s["mean_page_jaccard"] is None else f"{s['mean_page_jaccard']:.3f}"
        print(
            f"{name:14} {report['hot_rows']:3} {s['resident_hits']:10} {s['hot_hits']:4} "
            f"{s['misses']:5} {s['h2d_bytes']:11} {row_j:>6} {page_j:>8} {s['mean_page_density']:8.3f}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2) + "\n")
        print(f"Detailed trace: {args.output}")


if __name__ == "__main__":
    main()
