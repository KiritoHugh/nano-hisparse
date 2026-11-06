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

首次运行会下载权重。`--top-k 4` 使用 Quest 风格的近似选择，为每个 query head 选取四个页；`--top-k all` 使用完整 attention，并预留相应更大的工作集。稀疏选择可能改变模型输出，其近似误差与缓存存储是否正确是两个独立问题。[examples/generate.py](examples/generate.py) 展示了 Python API 的用法。

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
| [model.py](nano_hisparse/model.py)、[engine.py](nano_hisparse/engine.py) | Transformer 执行与轮转请求调度 |

## 实现范围

当前实现采用单进程、单 GPU，顺序执行各请求。尚未实现连续批处理、张量并行、前缀共享、MLA/FP8、分块 prefill 或跨层预取。选择器包含 Python 侧控制逻辑，行地址解析器由多个 PyTorch 操作组成。主存行加载通过 SM 发起读取，不是拷贝引擎 DMA；独立的搬运流用于表达依赖关系，本身并不保证与 attention 重叠执行。

设计参考：[vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)、[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 和 [Quest](https://github.com/mit-han-lab/Quest)。本仓库未直接收录上游实现代码。

采用 [MIT](LICENSE) 许可证。
