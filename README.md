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

The first run downloads weights. `--top-k 4` is a **page budget**: the Quest-style selector scores page min/max key bounds, keeps the sink and tail within the budget, and expands the chosen pages to all their valid tokens. With the default 16-token pages, four pages cover up to 64 tokens per query head. The cache itself resolves and loads individual token rows; the token-level trace example below exercises that finer granularity directly. `--top-k all` uses full attention and reserves a correspondingly larger working set. Sparse selection can change model outputs; it is separate from storage correctness. [examples/generate.py](examples/generate.py) shows the Python API.

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

## Token-row traces and locality

The trace example separates **which tokens are selected** from **how their KV rows are cached**. It feeds per-head token IDs directly into `Selection.from_tokens`, runs the real row resolver and attention, and checks every output against a reference using exactly the same selection.

```text
per-head token IDs → deduplicated union → resident / hot hit / miss
                                                   ↓
                                      load only missing KV rows
                                                   ↓
                                      per-head masked attention
```

### Controlled access patterns

No model download is needed. The JSON contains every step's per-head selection, row/page unions, hot-slot contents, hits, misses and numerical error.

```bash
python examples/row_trace.py demo --device cpu --output results/row-locality.json
# Run the same inputs through the CUDA row gather and attention kernels.
python examples/row_trace.py demo --device cuda:0 --output results/row-locality-cuda.json
```

Each trace has eight decode steps, four selected rows in the union, four tokens per page, and a cold hot buffer. Historical pages start evicted; two tail pages remain protected. Two query heads overlap on one token: five head/token pairs require only four KV records. Every row holds both K and V for all KV heads (64 bytes in this example).

| Pattern | Hot rows H | Hot hits | Misses | Missing-row payload | Mean row J | Mean page J | Page density |
|---|---:|---:|---:|---:|---:|---:|---:|
| reuse | 4 | 28 | 4 | 256 B | 1.000 | 1.000 | 0.25 |
| alternating | 4 | 0 | 32 | 2048 B | 0.000 | 1.000 | 0.25 |
| alternating | 8 | 24 | 8 | 512 B | 0.000 | 1.000 | 0.25 |
| streaming | 8 | 0 | 32 | 2048 B | 0.000 | 0.857 | 0.25 |
| packed | 4 | 28 | 4 | 256 B | 1.000 | 1.000 | 1.00 |

`reuse` repeatedly selects rows `[0,4,8,12]`; `packed` selects `[0,1,2,3]`. Both have temporal reuse, but different spatial density. `alternating` cycles between these two sets:

```text
step 0:  rows 0,4,8,12    → pages 0,1,2,3
step 1:  rows 1,5,9,13    → pages 0,1,2,3
step 2:  rows 0,4,8,12    → pages 0,1,2,3
```

The pages never change, yet H=4 misses on every step. H=8 retains both row sets and hits after warm-up. `streaming` visits new rows at every step, so even H=8 cannot create reuse. These controls demonstrate why page overlap alone is insufficient to establish row locality.

For selected row union `R_t` and its page set `P_t`, row J is `|R_t ∩ R_(t-1)| / |R_t ∪ R_(t-1)|`; page J uses `P_t` instead. The table averages adjacent-step Jaccard values, excluding the first step. Page density is selected rows divided by valid rows in the selected pages. JSON also reports the mean gap between repeated row accesses and `hot_hits / (hot_hits + misses)`; resident hits are separate. Adjacent overlap is not a cache hit rate: alternating rows have a two-step reuse gap even when row J is zero.

The H=4 and H=8 runs use three and four device blocks respectively, isolating hot capacity rather than comparing equal total memory. Payload counts only missing-row K/V bytes, excluding metadata and bus transaction overhead; CPU runs simulate that payload. Diagnostics synchronize and compute reference attention, so these runs are not latency benchmarks. Synthetic patterns validate cache behavior; they do not establish locality in a trained model.

### Capture a model's token-level trace

Capture records Q and incremental K/V from **one layer of a dense teacher**, then ranks individual historical tokens by each query head's exact QK score. `--tokens-per-head 8` selects eight **tokens**, without expanding their pages or forcing sink/tail tokens. This oracle reads all keys offline; it is not a learned DSA indexer or an efficient online selector. The teacher continues using full attention, so changing the oracle budget does not change the generation trajectory.

```bash
# Exercise capture/replay with the tiny random decoder, without downloading weights.
python examples/row_trace.py capture --max-context 256 --steps 8 \
  --tokens-per-head 8 --output results/tiny-rows.pt
python examples/row_trace.py replay results/tiny-rows.pt --output results/tiny-rows.json

# Optional: capture actual Q/K from pretrained Qwen (requires .[models]).
python examples/row_trace.py capture --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda:0 --prompt-file README.zh-CN.md --max-context 512 \
  --steps 32 --tokens-per-head 8 --layer 2 --output results/qwen-rows.pt
python examples/row_trace.py replay results/qwen-rows.pt --device cuda:0 \
  --output results/qwen-rows.json
```

The `.pt` file freezes CPU Q/K/V tensors and token selections, plus model identity, revision when available, prompt-token hash, layer and budget metadata. Replay needs no model weights; use `--hot-rows H` to compare capacities on the same trace. The default H is the largest observed cross-head row union, not the per-head token budget. Outputs go to the ignored `results/` directory.

An A100/BF16 run of the Qwen command above (layer index 2, 512 prompt tokens, 32 steps, PyTorch 2.15.0) produced the following replay results. The prompt-token SHA-256 starts with `4a6e9d526559d6ce`; the loader did not expose a model revision, which is recorded as `null`. To compare the second capacity, repeat replay with `--hot-rows 68`.

| Hot rows H | Device blocks | Resident hits | Hot hits | Misses | Missing-row payload | Hot hit rate |
|---|---:|---:|---:|---:|---:|---:|
| 34 | 5 | 434 | 264 | 131 | 67,072 B | 66.8% |
| 68 | 7 | 434 | 281 | 114 | 58,368 B | 71.1% |

The same frozen trace has mean row J **0.525**, page J **0.664**, page density **0.182**, and a mean reuse gap of **1.84 steps**. Its maximum same-selection attention error was 0.00196 in BF16. This sample exhibits row reuse and benefits from retaining more rows; increasing H also increases the device footprint.

For a model trace, inspect row/page J, reuse gaps, page density and measured hot hits together. Conclusions apply to the captured layer, prompt and dense-teacher trajectory. Broader locality claims require multiple prompts and layers; sparse-generation quality and selector cost are separate measurements.

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
| [trace.py](nano_hisparse/trace.py), [row_trace.py](examples/row_trace.py) | Offline token capture, replay and locality diagnostics |
| [model.py](nano_hisparse/model.py), [engine.py](nano_hisparse/engine.py) | Transformer execution and round-robin request scheduling |

## Scope

This is a single-process, single-GPU implementation with sequential request execution. It does not implement continuous batching, tensor parallelism, prefix sharing, MLA/FP8, chunked prefill or cross-layer prefetch. The selector has Python-side control; the row resolver uses multiple PyTorch operations. Host gather uses SM-initiated loads, not copy-engine DMA. A transfer stream expresses dependencies but does not by itself guarantee overlap with attention.

Design references: [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), and [Quest](https://github.com/mit-han-lab/Quest). No upstream implementation is vendored.

Licensed under [MIT](LICENSE).
