"""CUDA row gather from mapped pinned host memory and physical-row attention."""

import ctypes

import torch
import triton
import triton.language as tl


class MappedHost:
    """Keep pinned backing alive; expose its CUDA alias, without a second pool.

    PyTorch already owns/registers the pinned allocation. Query the CUDA driver
    for its device alias; never register/unregister memory owned by PyTorch.
    """

    def __init__(self, tensor):
        if not tensor.is_pinned():
            raise ValueError("row gather requires pinned host backing")
        driver = ctypes.CDLL("libcuda.so.1")
        query = driver.cuPointerGetAttribute
        query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
        query.restype = ctypes.c_int
        pointer = ctypes.c_uint64()
        # CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3 (CUDA Driver API).
        status = query(ctypes.byref(pointer), 3, tensor.data_ptr())
        if status:
            raise RuntimeError(f"cannot map pinned host allocation to CUDA: {status}")
        self.tensor, self.dtype, self.pointer = tensor, tensor.dtype, pointer.value

    def data_ptr(self):
        return self.pointer


@triton.jit
def _gather(
    Host,
    Pool,
    HostRows,
    DeviceRows,
    N: tl.constexpr,
    PAGE: tl.constexpr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    elements = tl.arange(0, BLOCK)
    plane, within = elements // ROW, elements % ROW
    for i in range(tl.program_id(0), N, tl.num_programs(0)):
        source = tl.load(HostRows + i)
        destination = tl.load(DeviceRows + i)
        valid = (source >= 0) & (destination >= 0) & (elements < 2 * ROW)
        src = (source // PAGE * 2 * PAGE + plane * PAGE + source % PAGE) * ROW + within
        dst = (destination // PAGE * 2 * PAGE + plane * PAGE + destination % PAGE) * ROW + within
        value = tl.load(Host + src, valid, other=0)
        tl.store(Pool + dst, value, valid)


def gather_rows(host, pool, plan, config):
    n = plan.miss_host_rows.numel()
    row = config.kv_heads * config.head_dim
    _gather[(min(n, 64),)](
        host,
        pool,
        plan.miss_host_rows,
        plan.miss_device_rows,
        n,
        config.page_size,
        row,
        triton.next_power_of_2(2 * row),
    )


@triton.jit
def _attend(
    Q,
    Pool,
    Rows,
    Out,
    QSTRIDE: tl.constexpr,
    RSTRIDE: tl.constexpr,
    CSTRIDE: tl.constexpr,
    N: tl.constexpr,
    PAGE: tl.constexpr,
    HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    DIM: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
):
    head = tl.program_id(0)
    kh = head // GROUPS
    d = tl.arange(0, D)
    query = tl.load(Q + head * QSTRIDE + d, d < DIM, other=0).to(tl.float32)
    maximum = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0, tl.float32)
    numerator = tl.full((D,), 0, tl.float32)
    for start in range(tl.cdiv(N, T)):
        i = start * T + tl.arange(0, T)
        rows = tl.load(Rows + head * RSTRIDE + i * CSTRIDE, i < N, other=-1)
        base = ((rows // PAGE * 2 * PAGE + rows % PAGE) * HEADS + kh) * DIM
        offset = base[:, None] + d[None, :]
        valid = (i < N) & (rows >= 0)
        mask = valid[:, None] & (d[None, :] < DIM)
        key = tl.load(Pool + offset, mask, other=0).to(tl.float32)
        value = tl.load(Pool + offset + PAGE * HEADS * DIM, mask, other=0).to(tl.float32)
        score = tl.where(valid, tl.sum(key * query[None, :], 1) * DIM**-0.5, -float("inf"))
        updated = tl.maximum(maximum, tl.max(score, 0))
        alpha = tl.where(maximum == -float("inf"), 0.0, tl.exp(maximum - updated))
        prob = tl.exp(score - tl.where(updated == -float("inf"), 0.0, updated))
        numerator = numerator * alpha + tl.sum(prob[:, None] * value, 0)
        denom = denom * alpha + tl.sum(prob, 0)
        maximum = updated
    tl.store(Out + head * DIM + d, tl.where(denom > 0, numerator / denom, 0.0), d < DIM)


def attend_rows(query, pool, rows, config):
    query = query.contiguous()
    output = torch.empty_like(query)
    _attend[(query.shape[0],)](
        query,
        pool,
        rows,
        output,
        query.stride(0),
        rows.stride(0),
        rows.stride(1),
        rows.shape[1],
        config.page_size,
        config.kv_heads,
        query.shape[0] // config.kv_heads,
        config.head_dim,
        triton.next_power_of_2(config.head_dim),
        64,
    )
    return output
