# nano-hisparse

[English](README.md) | 简体中文

一个支持 KV Cache 部分驻留的轻量推理引擎。参考 [vLLM 中的 HiSparse](https://github.com/vllm-project/vllm/tree/abf98672ce7e822701f7a062bf240efb2c9900d1/vllm/v1/hisparse) 的设计，以及 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的简洁代码组织方式。

完整 KV 历史由主存提供后备存储。驻留页与可复用的热缓冲区（hot buffer）共享**同一个 GPU tensor**。每步 decode 直接读取驻留行，只从主存加载本步选中但缺失的行。历史页被逐出后，请求仍可继续生成，无需恢复整个 KV Cache，也无需重新 prefill。

## 快速开始

需要 Python 3.10+ 和 PyTorch 2.5+。CPU 演示无需下载模型。CUDA 路径面向配备 NVIDIA GPU 的 Linux 环境，使用 CUDA 版 PyTorch 安装所提供的 Triton。

```bash
git clone https://github.com/KiritoHugh/nano-hisparse.git
cd nano-hisparse
python -m venv .venv
source .venv/bin/activate

# 使用 CUDA 时，请改为安装与环境匹配的 CUDA 版 PyTorch。
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
nano-hisparse demo --device cpu
```

演示使用随机模型权重，输出用于观察执行过程和计数器，不代表有意义的文本。默认 KV 池只能容纳一个请求所需的最小 hot/tail 配额。三个较长的输入依次通过准入，每个请求只执行一次 prefill，之后增量解码。

安装 CUDA 版 PyTorch 后，可运行：

```bash
nano-hisparse demo --device cuda:0 --dtype bfloat16
```

## 使用预训练模型生成文本

可选的模型适配器支持采用标准 RoPE 的 Qwen2/Qwen2.5。Transformers 提供分词器与预训练模块，KV 管理和 decode attention 由本项目执行。

```bash
python -m pip install -e '.[models]'
nano-hisparse generate --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda:0 --dtype bfloat16 --chat \
  --prompt '请解释 KV Cache 的工作原理。' --top-k 4 --max-new-tokens 64
```

首次运行会下载权重。`--top-k 4` 是**页预算**：Quest 风格选择器通过页内 K 的 min/max 边界评分，将 sink 页和尾页计入预算，再把选中页展开为页内全部有效 token。默认每页 16 个 token，四页对应每个 query head 最多 64 个 token。缓存本身按 token row 解析和加载，下文的 row trace 示例直接验证这个更细的粒度。`--top-k all` 使用完整 attention，并预留相应更大的工作集。稀疏选择可能改变模型输出，其近似误差与缓存存储是否正确是两个独立问题。[examples/generate.py](examples/generate.py) 展示了 Python API 的用法。

## 工作原理

```text
logical tokens → resident lookup → hot lookup → miss plan
                      │                │           │
                      │                │     host-row gather
                      └────────────────┴───────────┘
                                       ↓
                              physical-row attention
```

- **稳定的历史地址。** 逻辑页保留主存地址，其 GPU 物理块可以重新分配。新增 KV 先追加到驻留尾页，再镜像至主存。
- **共享的容量池。** 每个请求的每一层预留 hot 区域额度和两个活跃尾页块。历史页可借用尚未使用的容量，压力出现后再完整分配 hot 区域。
- **安全的空间复用。** 已完成主存镜像的历史页在变为可回收状态后仍可读取，直到物理块实际复用时才撤销驻留映射。租约、代次与 CUDA 事件共同保护读者和在途拷贝。
- **按行替换。** 地址解析器先保护本步所有命中的 hot 行，再按 LRU 为缺失行选择替换位置。GQA 的多个头共享已加载的数据行，同时保留各自的选择掩码。

设每页容纳 `P` 个 tokens，每层 hot 容量为 `H` 行，模型层数为 `L`，本步选中 token 的最大并集大小为 `U`：

```text
H >= U
每个请求的最小 GPU 块数 = L * (ceil(H / P) + 2)
```

`U` 按所有 query heads 的选择并集计算，而非单个头的选择宽度。默认使用 `H = 2U`，`H = U` 也能保证持续前进。扩大 hot 区域会减少可用于历史页驻留的空间。准入时还会预留主存后备存储和索引容量。

`--gpu-pages` 只限制持久 KV 池的大小，模型权重、索引、激活和 prefill 工作区还需额外显存。运行 `nano-hisparse demo --help` 可查看配置选项。

## Token-row 轨迹与局部性

trace 示例把**选择哪些 token**与**如何缓存这些 token 的 KV 行**分开：将每个 head 的 token ID 直接交给 `Selection.from_tokens`，执行真实的 row resolver 和 attention，并逐步与相同选择下的参考结果做数值对照。

```text
各 head 的 token ID → 去重后的行并集 → resident / hot 命中 / miss
                                                   ↓
                                          只加载缺失 KV 行
                                                   ↓
                                        保留各 head 掩码的 attention
```

### 用受控访问验证 row 管理

无需下载模型。输出 JSON 包含每步各 head 的选择、row/page 并集、hot 槽位内容、命中、缺失和数值误差。

```bash
python examples/row_trace.py demo --device cpu --output results/row-locality.json
# 相同输入也可验证 CUDA 的按行加载与 attention 内核。
python examples/row_trace.py demo --device cuda:0 --output results/row-locality-cuda.json
```

每条轨迹包含八步 decode，每步选中行的并集为四行，每页四个 token。hot 从空缓存开始，历史页已卸载，保留两个受保护的尾页。两个 query head 重复选择其中一个 token，因此五个 head/token 组合只需访问四条 KV 记录。这里一行包含该 token 所有 KV heads 的 K 和 V，共 64 字节。

| 访问模式 | Hot 行数 H | Hot 命中 | Miss | 缺失行搬运量 | 平均 row J | 平均 page J | 页内密度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| reuse | 4 | 28 | 4 | 256 B | 1.000 | 1.000 | 0.25 |
| alternating | 4 | 0 | 32 | 2048 B | 0.000 | 1.000 | 0.25 |
| alternating | 8 | 24 | 8 | 512 B | 0.000 | 1.000 | 0.25 |
| streaming | 8 | 0 | 32 | 2048 B | 0.000 | 0.857 | 0.25 |
| packed | 4 | 28 | 4 | 256 B | 1.000 | 1.000 | 1.00 |

`reuse` 重复访问 `[0,4,8,12]`，`packed` 重复访问 `[0,1,2,3]`：两者都有时间复用，但页内空间密度不同。`alternating` 则交替访问两组行：

```text
step 0:  rows 0,4,8,12    → pages 0,1,2,3
step 1:  rows 1,5,9,13    → pages 0,1,2,3
step 2:  rows 0,4,8,12    → pages 0,1,2,3
```

虽然始终访问相同的页，H=4 时仍然每步全部 miss；H=8 能保留两组行，预热后便可命中。`streaming` 每步访问新行，即使 H=8 也没有复用。这些对照说明，仅凭 page 重合率不能判断 row locality。

设第 t 步选中行并集为 `R_t`，其所在页集合为 `P_t`。row J 为 `|R_t ∩ R_(t-1)| / |R_t ∪ R_(t-1)|`，page J 将其中的集合替换为 `P_t`。表中对相邻步的 Jaccard 取平均，排除第一步。页内密度是选中行数除以选中页中的有效行数。JSON 还记录重复访问的平均步间隔，以及 `hot_hits / (hot_hits + misses)`；resident 命中单独计数。相邻步重合率不等于缓存命中率：交替访问的 row J 为零，但仍有间隔两步的复用。

H=4 和 H=8 分别使用三个和四个 device blocks，这里比较的是 hot 容量影响，而非相同总显存下的优劣。搬运量只计算缺失行的 K/V 有效载荷，不含元数据与总线事务开销；CPU 运行模拟这部分字节数。诊断路径会同步并计算参考 attention，不用于延迟测量。受控模式验证缓存行为，不能据此宣称真实模型具有局部性。

### 从模型采集 token 级 trace

采集器记录**使用完整 attention 的模型某一层**的 Q 和增量 K/V，再离线按每个 query head 的精确 QK 分数选择历史 token。`--tokens-per-head 8` 选择八个 **token**，不展开整页，也不强制加入 sink/tail token。这个 oracle 需要读取全部 K，用于获得 trace；它不是 learned DSA indexer，也不是高效的在线选择器。模型仍使用完整 attention，因此更改 oracle 预算不会改变生成轨迹。

```bash
# 用随机小模型跑通采集与回放，无需下载权重。
python examples/row_trace.py capture --max-context 256 --steps 8 \
  --tokens-per-head 8 --output results/tiny-rows.pt
python examples/row_trace.py replay results/tiny-rows.pt --output results/tiny-rows.json

# 可选：从预训练 Qwen 的真实 Q/K 采集（需安装 .[models]）。
python examples/row_trace.py capture --model Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda:0 --prompt-file README.zh-CN.md --max-context 512 \
  --steps 32 --tokens-per-head 8 --layer 2 --output results/qwen-rows.pt
python examples/row_trace.py replay results/qwen-rows.pt --device cuda:0 \
  --output results/qwen-rows.json
```

`.pt` 文件保存冻结的 CPU Q/K/V 张量和 token 选择，以及模型名称、可获取的 revision、prompt token 哈希、层号与预算等来源信息。回放不需要模型权重；通过 `--hot-rows H` 可以在同一条 trace 上比较不同容量。默认 H 为所有步骤中最大的跨 head 行并集，而不是单个 head 的 token 预算。产物写入已忽略的 `results/` 目录。

在 A100/BF16 上执行上述 Qwen 命令（层索引 2、512 个 prompt tokens、32 步、PyTorch 2.15.0），得到如下回放结果。prompt token 的 SHA-256 前缀为 `4a6e9d526559d6ce`；加载器未提供模型 revision，文件中记为 `null`。重复 replay 并添加 `--hot-rows 68` 即可比较第二种容量。

| Hot 行数 H | Device blocks | Resident 命中 | Hot 命中 | Miss | 缺失行搬运量 | Hot 命中率 |
|---|---:|---:|---:|---:|---:|---:|
| 34 | 5 | 434 | 264 | 131 | 67,072 B | 66.8% |
| 68 | 7 | 434 | 281 | 114 | 58,368 B | 71.1% |

同一条冻结轨迹的平均 row J 为 **0.525**、page J 为 **0.664**、页内密度为 **0.182**，平均复用间隔为 **1.84 步**。BF16 下，相同选择的 attention 最大绝对误差为 0.00196。这个样本具有 row 复用，保留更多行可以减少 miss，同时也增加了 device 空间占用。

分析模型 trace 时，应结合 row/page J、复用间隔、页内密度和实际 hot 命中一起判断。结论只适用于采集的层、prompt 和完整 attention 生成轨迹；更广泛的 locality 结论需要覆盖多个输入和层。稀疏生成质量与选择器开销需要另外评估。

## 示例与测试

以下两个小示例无需加载模型，可直接观察缓存机制：

```bash
python examples/inspect_indices.py --device cpu
python examples/residency.py --device cpu
# 生命周期示例也支持 CUDA。
python examples/residency.py --device cuda:0
```

测试覆盖相同选择下的 attention 数值一致性、非连续 hot 块、延迟读者、请求取消、容量准入，以及每步选中的历史行全部缺失时的持续解码。模型测试对比增量解码与完整因果前向计算的 logits；可选的 Transformers 对照测试在本地创建随机 Qwen 权重，无需下载模型。

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

CI 在 Python 3.10 和 3.12 上运行 CPU 检查。有可用 GPU 时会运行 CUDA 用例；仅通过 CPU 测试不能验证 CUDA 事件顺序或 Triton 内核。

## 代码结构

| 文件 | 职责 |
|---|---|
| [allocator.py](nano_hisparse/allocator.py)、[storage.py](nano_hisparse/storage.py) | 物理块、逻辑页、主存镜像与延迟回收 |
| [hybrid.py](nano_hisparse/hybrid.py) | 容量准入、驻留策略、hot 租约与行生命周期 |
| [row_index.py](nano_hisparse/row_index.py)、[row_kernels.py](nano_hisparse/row_kernels.py) | 地址解析、按行加载与 attention 内核 |
| [selection.py](nano_hisparse/selection.py)、[attention.py](nano_hisparse/attention.py) | 各头的 token 选择语义与 attention 分发 |
| [trace.py](nano_hisparse/trace.py)、[row_trace.py](examples/row_trace.py) | 离线 token 采集、回放与局部性诊断 |
| [model.py](nano_hisparse/model.py)、[engine.py](nano_hisparse/engine.py) | Transformer 执行与轮转请求调度 |

## 实现范围

当前实现采用单进程、单 GPU，顺序执行各请求。尚未实现连续批处理、张量并行、前缀共享、MLA/FP8、分块 prefill 或跨层预取。选择器包含 Python 侧控制逻辑，行地址解析器由多个 PyTorch 操作组成。主存行加载通过 SM 发起读取，不是拷贝引擎 DMA；独立的搬运流用于表达依赖关系，本身并不保证与 attention 重叠执行。

设计参考：[vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)、[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 和 [Quest](https://github.com/mit-han-lab/Quest)。本仓库未直接收录上游实现代码。

采用 [MIT](LICENSE) 许可证。
