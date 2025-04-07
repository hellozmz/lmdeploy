# Copyright (c) OpenMMLab. All rights reserved.
# modify from: https://github.com/vllm-project/vllm
import torch
import triton
import triton.language as tl

from .activation import silu_and_mul
from .blocked_gemm_fp8 import quant_fp8
from .fused_moe import _get_sorted_idx, _make_intermediate, _renormalize


from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

def get_cuda_autotune_config():
    return [
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 64,
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
        }, num_stages=4, num_warps=4),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['N', 'K', 'M_NP2'],
    warmup=10,
    rep=25,
)
@triton.jit
def fused_moe_blocked_f8_kernel(
    A,
    A_scale,
    B,
    B_scale,
    C,
    SortedIdx,
    ExpStart,
    ExpEnd,
    Weights,
    N: tl.constexpr,
    K: tl.constexpr,
    group_ak: tl.constexpr,
    group_bk: tl.constexpr,
    group_bn: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_asm,
    stride_ask: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    stride_cm,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    M_NP2: tl.constexpr,
    ENABLE_WEIGHTS: tl.constexpr,
    top_k: tl.constexpr,
    expert_offset: tl.constexpr,
    reindex_a: tl.constexpr,
    reindex_c: tl.constexpr,
):
    """fused moe kernel."""
    exp_id = tl.program_id(1)
    pid = tl.program_id(0)

    exp_start = tl.load(ExpStart + exp_id + expert_offset)
    exp_end = tl.load(ExpEnd + exp_id + expert_offset)
    M = exp_end - exp_start
    
    # 新增调试日志
    # tl.device_print("[KernelDebug] exp_id=%d, exp_start=%d, exp_end=%d, M=%d", 
    #          exp_id, exp_start, exp_end, M)
    if M <= 0:                          # 当传递的topk_id=-1的时候，作差之后的个数会是0，会在这里被跳过。代码逻辑是符合预期的。
        return

    num_pid_m = tl.cdiv(M_NP2, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= M or pid_n * BLOCK_SIZE_N >= N:
        return

    # 原始索引数组中的位置（exp_start当前专家起始位置 + 块内偏移）
    offs_sid = exp_start + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_sid = offs_sid < exp_end
    # 加载排序后的全局token索引（来自预排序的SortedIdx数组）
    sid = tl.load(SortedIdx + offs_sid, mask=mask_sid, other=0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if reindex_a:
        # 将排序索引转换为原始token索引（每个token有top_k个专家选择）
        offs_am = sid // top_k
    else:
        # 直接使用原始索引（专家维度顺序）
        offs_am = offs_sid
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # deepseek has 160 experts, exp index would overflow int32
    exp_id = exp_id.to(tl.int64)
    exp_off = stride_be * exp_id
    b_ptrs = B + exp_off + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = pid_n * BLOCK_SIZE_N // group_bn
    as_ptrs = A_scale + offs_am * stride_asm
    bs_ptrs = B_scale + stride_bse * exp_id + offs_bsn * stride_bsn

    acc_scale = tl.load(as_ptrs, mask=mask_sid, other=1.0) * tl.load(bs_ptrs)
    acc_ratio = 1 / acc_scale
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # load scales
        k_start = (k + 1) * BLOCK_SIZE_K
        offs_ksa = k_start // group_ak
        offs_ksb = k_start // group_bk
        a_scale = tl.load(as_ptrs + offs_ksa * stride_ask, mask=mask_sid and k_start < K, other=1.0)
        b_scale = tl.load(bs_ptrs + offs_ksb * stride_bsk, mask=k_start < K, other=1.0)

        # load ab
        a = tl.load(a_ptrs, mask=mask_sid[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # mma
        accumulator = tl.dot(a, b, acc=accumulator * acc_ratio[:, None])

        # update scales and ratio
        new_acc_scale = a_scale * b_scale
        acc_ratio = acc_scale / new_acc_scale
        acc_scale = new_acc_scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator * (acc_ratio * acc_scale)[:, None]

    if ENABLE_WEIGHTS:
        weight = tl.load(Weights + sid, mask=mask_sid)
        c = c * weight[:, None].to(c.dtype)

    c = c.to(C.dtype.element_ty)

    if reindex_c:
        offs_cm = sid
    else:
        offs_cm = offs_sid
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, c, mask=mask_sid[:, None])


def fused_moe_blocked_fp8_kernel_launcher(
    A: torch.Tensor,
    A_scale: torch.Tensor,
    B: torch.Tensor,
    B_scale: torch.Tensor,
    C: torch.Tensor,
    sorted_idx: torch.Tensor,
    exp_start: torch.Tensor,
    exp_end: torch.Tensor,
    weights: torch.Tensor,
    enable_weights: bool = False,
    top_k: int = 1,
    num_tokens: int = None,
    expert_offset: int = 0,
    reindex_a: bool = True,
    reindex_c: bool = True,
):
    """fused moe kernel launcher."""
    
    # 新增调试日志
    # if A.shape[0] != -1234:
    #     logger.error(f"[MoeKernel] Input A shape: {A.shape}, B shape: {B.shape}")
    #     logger.error(f"[MoeKernel] Expert range: {expert_offset}-{expert_offset+B.shape[0]}")
    #     logger.error(f"[MoeKernel] Sorted indices: {sorted_idx.unique()}")

    if num_tokens is None:
        num_tokens = A.size(0)
    M_NP2 = triton.next_power_of_2(num_tokens)
    M_NP2 = max(64, M_NP2)
    E, N, K = B.shape

    # if A.shape[0] != -1234:
    #     unique_experts = torch.unique(sorted_idx % E)
    #     logger.error(f"[MoeKernel] Active experts: {unique_experts.tolist()}")
    #     logger.error(f"[MoeKernel] ExpStart: {exp_start.tolist()}...")
    #     logger.error(f"[MoeKernel] ExpEnd: {exp_end.tolist()}...")

    assert A.dim() == 2
    assert A_scale.dim() == 2
    assert B.dim() == 3
    assert B_scale.dim() == 3

    assert K % A_scale.size(1) == 0
    assert K % B_scale.size(2) == 0
    assert N % B_scale.size(1) == 0

    group_ak = K // A_scale.size(1)
    group_bk = K // B_scale.size(2)
    group_bn = N // B_scale.size(1)

    def _grid_fn(META):
        grid = (triton.cdiv(M_NP2, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), E)
        # logger.error(f"[MoeKernel] Launching kernel with grid: {grid}") 
        # logger.error(f"[MoeKernel] M_NP2:{M_NP2}, N: {N}, E:{E},BLOCK_SIZE_M: {META['BLOCK_SIZE_M']}, BLOCK_SIZE_N: {META['BLOCK_SIZE_N']}")
        return grid

    # 数据被打平了，传递到 kernel
    A = A.flatten(0, -2)        # [M, K] -> [M*K]
    C = C.flatten(0, -2)        # [M, N] -> [M*N]

    BLOCK_SIZE_K = group_bk
    GROUP_SIZE_M = 8
    grid = _grid_fn
    fused_moe_blocked_f8_kernel[grid](
        A,
        A_scale,
        B,
        B_scale,
        C,
        sorted_idx,
        exp_start,
        exp_end,
        weights,
        N=N,
        K=K,
        group_ak=group_ak,
        group_bk=group_bk,
        group_bn=group_bn,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_asm=A_scale.stride(0),
        stride_ask=A_scale.stride(1),
        stride_be=B.stride(0),
        stride_bn=B.stride(1),
        stride_bk=B.stride(2),
        stride_bse=B_scale.stride(0),
        stride_bsn=B_scale.stride(1),
        stride_bsk=B_scale.stride(2),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        ENABLE_WEIGHTS=enable_weights,
        top_k=top_k,
        expert_offset=expert_offset,
        reindex_a=reindex_a,
        reindex_c=reindex_c,
        M_NP2=M_NP2,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    # 新增内核参数日志
    # if A.shape[0] != -1234:
    #     logger.error(f"[MoeKernel] reindex_a: {reindex_a}, reindex_c: {reindex_c}")


def fused_moe_blocked_fp8(input: torch.Tensor,
                          input_scale: torch.Tensor,
                          w1: torch.Tensor,
                          w1_scale: torch.Tensor,
                          w2: torch.Tensor,
                          w2_scale: torch.Tensor,
                          topk_weights: torch.Tensor,
                          topk_ids: torch.Tensor,
                          topk: int,
                          out_dtype: torch.dtype = torch.float16,
                          expert_offset: int = 0,
                          num_experts: int = None,
                          renormalize: bool = False) -> torch.Tensor:
    """fused moe."""
    device = input.device
    M = input.size(0)
    E, N, _ = w1.shape
    if num_experts is None:
        num_experts = E
    full_exp = num_experts == E
    group_size = input.size(-1) // input_scale.size(-1)
    
    
    # 新增输入维度分析日志
    # if M != -1234:
    #     logger.error(f"[MoE结构分析] 输入特征形状: {input.shape} → [batch_size*seq_len={M}, hidden_dim={input.shape[-1]}]")
    #     logger.error(f"[MoE结构分析] 输入缩放因子形状: {input_scale.shape} → [batch_size*seq_len={M}, 输入维度={input_scale.shape[-1]}]")
    #     logger.error(f"[MoE结构分析] 专家权重w1形状: {w1.shape} → [专家数={E}, 合并维度(gate+up)={N}, 输入维度={w1.shape[-1]}]")
    #     logger.error(f"[MoE结构分析] 专家权重w2形状: {w2.shape} → [专家数={E}, 输出维度={w2.shape[-1]}, 合并维度(down)=1]")
    #     logger.error(f"[MoE结构分析] w1_scale形状: {w1_scale.shape} → [专家数={E}, 合并维度(gate+up)={N//group_size}]")
    #     logger.error(f"[MoE结构分析] w2_scale形状: {w2_scale.shape} → [专家数={E}, 合并维度(down)=1]")
    #     logger.error(f"[MoE结构分析] 激活前中间缓存形状: (M={M}, topk={topk}, N={N}) → 每个token保留{topk}个专家，每个专家输出{N}维")
    #     logger.error(f"[MoE结构分析] topk_weights形状: {topk_weights.shape} → [token数={M}, topk={topk}]")
    #     logger.error(f"[MoE结构分析] topk_ids形状: {topk_ids.shape} → [token数={M}, topk={topk}]")
    #     # logger.error(f"[MoE结构分析] topk_ids: {topk_ids}")

    topk_weights = _renormalize(topk_weights, renormalize)
    sorted_idx, exp_start, exp_end = _get_sorted_idx(topk_ids, num_experts)

    intermediate_cache1 = _make_intermediate((M, topk, N), dtype=out_dtype, device=device, zeros=not full_exp)
    # 新增专家选择分析日志
    # if M != -1234:
    #     logger.error(f"[MoE专家选择] num_experts: {num_experts}")
    #     logger.error(f"[MoE专家选择] 选择前topk_ids: {topk_ids}")
    #     logger.error(f"[MoE专家选择] 选择前sorted_idx: {sorted_idx}")
    #     logger.error(f"[MoE专家选择] 选择前exp_start: {exp_start}")
    #     logger.error(f"[MoE专家选择] 选择前exp_end: {exp_end}")
    #     unique_experts = torch.unique(topk_ids)
    #     logger.error(f"[MoE专家选择] 当前批次选中的专家ID示例: {unique_experts.tolist()}... (共{len(unique_experts)}个专家)")
    #     # 考虑专家偏移量
    #     current_experts = (topk_ids - expert_offset).clamp(min=0)
    #     unique_experts = torch.unique(current_experts[current_experts >= 0])
    #     logger.error(f"[MoE专家选择] 当前rank处理的专家ID范围: [{expert_offset}-{expert_offset+num_experts}]")
    #     logger.error(f"[MoE专家选择] 实际激活的本地专家ID: {unique_experts.tolist()}")
    #     logger.error(f"[MoE专家选择] 激活前中间缓存形状: {intermediate_cache1.shape} → [token数={M}, topk={topk}, 单个专家中间维度={N}]")

    # gate and up
    fused_moe_blocked_fp8_kernel_launcher(
        input,
        input_scale,
        w1,
        w1_scale,
        intermediate_cache1,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        weights=topk_weights,
        enable_weights=False,
        top_k=topk,
        num_tokens=M,
        expert_offset=expert_offset,
        reindex_a=True,
        reindex_c=False,
    )

    # activate
    intermediate_cache1 = intermediate_cache1.flatten(0, -2)
    gate_cache = silu_and_mul(intermediate_cache1)
    del intermediate_cache1

    # 激活后维度分析
    # if M != -1234:
    #     logger.error(f"[MoE激活分析] 激活后特征形状: {gate_cache.shape} → [总token数*topk={M*topk}, 单个专家中间维度={gate_cache.shape[-1]}]")


    gate_cache, gate_scale = quant_fp8(gate_cache, group_size, dtype=input.dtype)

    intermediate_cache2 = _make_intermediate((M, topk, w2.shape[1]), dtype=out_dtype, device=device, zeros=not full_exp)
    # 新增输出维度日志
    # if M != -1234:
    #     logger.error(f"[MoE输出分析] 输出形状: {intermediate_cache2.shape} → [token数={M}, topk={topk}, 单个专家输出维度={w2.shape[1]}")
    # down
    fused_moe_blocked_fp8_kernel_launcher(
        gate_cache,
        gate_scale,
        w2,
        w2_scale,
        intermediate_cache2,
        sorted_idx=sorted_idx,
        exp_start=exp_start,
        exp_end=exp_end,
        weights=topk_weights,
        enable_weights=True,
        top_k=1,
        num_tokens=M,
        expert_offset=expert_offset,
        reindex_a=False,
        reindex_c=True,
    )

    ret = intermediate_cache2.sum(dim=1)
    # 新增输出维度日志
    # if M != -1234:
    #     logger.error(f"[MoE输出分析] 最终输出形状: {ret.shape} → [token数={M}, 隐藏层维度={ret.shape[-1]}]")
    return ret
