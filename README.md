# nano-hisparse

English | [简体中文](README.zh-CN.md)

A small inference engine for decoding with partially resident KV caches. Inspired by [HiSparse in vLLM](https://github.com/vllm-project/vllm/tree/abf98672ce7e822701f7a062bf240efb2c9900d1/vllm/v1/hisparse) and the compact organization of [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).

The full KV history has host backing. Resident pages and reusable hot buffers share **one GPU tensor**. Each decode step reads resident rows directly and loads only missing selected rows from host memory. A request keeps generating after its historical pages are evicted, without restoring the entire KV cache or repeating prefill.

## Quick start

Python 3.10+ and PyTorch 2.5+ are required. CPU execution needs no model download. The CUDA path targets Linux with an NVIDIA GPU and uses Triton from the CUDA-enabled PyTorch installation.

```bash
git clone https://github.com/KiritoHugh/nano-hisparse.git
cd nano-hisparse
python -m venv .venv
source .venv/bin/activate

# For CUDA, install a matching CUDA-enabled PyTorch build instead.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
nano-hisparse demo --device cpu
```

The demo uses random model weights and reports execution counters, not meaningful text. Its default KV pool holds one request's minimum hot/tail footprint. Three longer prompts take turns through admission; each is prefilled once and decoded incrementally.

With CUDA-enabled PyTorch installed:

```bash
nano-hisparse demo --device cuda:0 --dtype bfloat16
```

## Pretrained generation

The optional adapter loads Qwen2/Qwen2.5 models with standard RoPE. KV management and decode attention use this runtime; Transformers supplies the tokenizer and pretrained modules.

```bash
python -m pip install -e '.[models]'
nano-hisparse generate --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda:0 --dtype bfloat16 --chat \
  --prompt 'Explain how a KV cache works.' --top-k 4 --max-new-tokens 64
```

The first run downloads weights. `--top-k 4` uses a Quest-style approximation selecting four pages per query head. `--top-k all` uses full attention and reserves a correspondingly larger working set. Sparse selection can change model outputs; it is separate from storage correctness. [examples/generate.py](examples/generate.py) shows the Python API.

## How it works

```text
logical tokens → resident lookup → hot lookup → miss plan
                      │                │           │
                      │                │     host-row gather
                      └────────────────┴───────────┘
                                       ↓
                              physical-row attention
```

- **Stable history.** Logical pages retain a host address while their device blocks can be reassigned. New KV is appended to resident tail pages and mirrored to host.
- **Shared capacity.** Every request/layer reserves a hot-region credit and two active tail blocks. History borrows unused capacity until pressure triggers a complete hot allocation.
- **Safe reuse.** Clean historical pages remain readable until actual block reuse revokes their mapping. Leases, generations and CUDA events protect readers and in-flight copies.
- **Row replacement.** The resolver protects all selected hot hits before choosing LRU victims for misses. GQA heads share loaded rows while preserving their individual selection masks.

For page size `P`, hot rows `H`, layers `L` and maximum selected token union `U`:

```text
H >= U
minimum device blocks per request = L * (ceil(H / P) + 2)
```

`U` is the union across query heads, not a single head's selection width. The default is `H = 2U`; `H = U` also supports progress. A larger hot region leaves less space for resident history. Admission also reserves host backing and index capacity.

`--gpu-pages` bounds the persistent KV pool, not total GPU memory. Weights, indices, activations and prefill workspace need additional capacity. Run `nano-hisparse demo --help` for the available options.

## Examples and tests

Two small examples expose the cache mechanics without loading a model:

```bash
python examples/inspect_indices.py --device cpu
python examples/residency.py --device cpu
# The same lifecycle example also runs on CUDA.
python examples/residency.py --device cuda:0
```

The tests check same-selection attention, fragmented hot blocks, delayed readers, cancellation, admission, and continued decoding when every selected historical row misses. Model tests compare incremental logits with full causal forward passes; the optional Transformers test creates local random Qwen weights and needs no download.

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

CI runs CPU checks on Python 3.10 and 3.12. CUDA cases run when a GPU is available; CPU tests alone do not verify event ordering or Triton kernels.

## Code map

| File | Responsibility |
|---|---|
| [allocator.py](nano_hisparse/allocator.py), [storage.py](nano_hisparse/storage.py) | Physical blocks, logical pages, host mirrors and retirement |
| [hybrid.py](nano_hisparse/hybrid.py) | Admission, residency policy, hot leases and row lifetimes |
| [row_index.py](nano_hisparse/row_index.py), [row_kernels.py](nano_hisparse/row_kernels.py) | Address resolution, row gather and attention kernels |
| [selection.py](nano_hisparse/selection.py), [attention.py](nano_hisparse/attention.py) | Per-head token semantics and attention dispatch |
| [model.py](nano_hisparse/model.py), [engine.py](nano_hisparse/engine.py) | Transformer execution and round-robin request scheduling |

## Scope

This is a single-process, single-GPU implementation with sequential request execution. It does not implement continuous batching, tensor parallelism, prefix sharing, MLA/FP8, chunked prefill or cross-layer prefetch. The selector has Python-side control; the row resolver uses multiple PyTorch operations. Host gather uses SM-initiated loads, not copy-engine DMA. A transfer stream expresses dependencies but does not by itself guarantee overlap with attention.

Design references: [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), and [Quest](https://github.com/mit-han-lab/Quest). No upstream implementation is vendored.

Licensed under [MIT](LICENSE).
