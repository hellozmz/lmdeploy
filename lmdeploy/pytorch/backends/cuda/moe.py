# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch
import torch.distributed as dist

import os

from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPDispatcher
from lmdeploy.pytorch.kernels.cuda import fused_moe, fused_moe_w8a8
from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import fused_moe_blocked_fp8
from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
from lmdeploy.pytorch.kernels.cuda.ep_moe import grouped_gemm_triton, silu_and_mul_triton_kernel
from lmdeploy.pytorch.kernels.cuda.fused_moe import _renormalize
from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import per_token_quant_int8
from lmdeploy.pytorch.models.q_modules import QTensor

from ..moe import (FusedMoEBlockedF8Builder, FusedMoEBlockedF8Impl, FusedMoEBuilder, FusedMoEImpl, FusedMoEW8A8Builder,
                   FusedMoEW8A8Impl)
from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank, get_tp_world_rank

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')

ep, ep_rank = get_ep_world_rank()

class TritonFusedMoEImpl(FusedMoEImpl):
    """triton fused moe implementation."""

    def __init__(self, top_k: int, num_experts: int, renormalize: bool = False):
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor):
        gate_up_weights = gate_up_weights.transpose(1, 2).contiguous().transpose(1, 2)
        down_weights = down_weights.transpose(1, 2).contiguous().transpose(1, 2)
        return gate_up_weights, down_weights

    def support_ep(self):
        """support expert parallelism."""
        return True

    def ep_expert_list(self, world_size: int, rank: int):
        """experts list of current rank."""
        num_experts = self.num_experts
        expert_per_rank = (num_experts + world_size - 1) // world_size
        first_expert = rank * expert_per_rank
        last_expert = min(first_expert + expert_per_rank, num_experts)
        return list(range(first_expert, last_expert))

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        expert_offset = 0
        num_experts = None
        if expert_list is not None and len(expert_list) != self.num_experts:
            expert_offset = expert_list[0]
            num_experts = self.num_experts
        return fused_moe(hidden_states,
                         gate_up_weights,
                         down_weights,
                         topk_weights=topk_weights,
                         topk_ids=topk_ids,
                         topk=self.top_k,
                         expert_offset=expert_offset,
                         num_experts=num_experts,
                         renormalize=self.renormalize)


class TritonFusedMoEBuilder(FusedMoEBuilder):
    """triton fused moe builder."""

    @staticmethod
    def build(top_k: int, num_experts: int, renormalize: bool = False):
        """build from mlp."""
        return TritonFusedMoEImpl(top_k=top_k, num_experts=num_experts, renormalize=renormalize)


class TritonFusedMoEW8A8Impl(FusedMoEW8A8Impl):
    """triton fused moe w8a8 implementation."""

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        renormalize: bool = False,
        out_dtype: torch.dtype = torch.float16,
        quant_dtype: torch.dtype = torch.int8,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize
        self.out_dtype = out_dtype
        self.quant_dtype = quant_dtype

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor, gate_up_scale: torch.Tensor,
                       down_scale: torch.Tensor):
        # do not transpose weight for int8/fp8
        return gate_up_weights, down_weights, gate_up_scale, down_scale

    def support_ep(self):
        """support expert parallelism."""
        return True

    def ep_expert_list(self, world_size: int, rank: int):
        """experts list of current rank."""
        num_experts = self.num_experts
        expert_per_rank = (num_experts + world_size - 1) // world_size
        first_expert = rank * expert_per_rank
        last_expert = min(first_expert + expert_per_rank, num_experts)
        return list(range(first_expert, last_expert))

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""

        if isinstance(hidden_states, torch.Tensor):
            hidden_states = hidden_states.contiguous()
            input_quant, input_scale = per_token_quant_int8(hidden_states, 1e-7, quant_dtype=self.quant_dtype)
        else:
            assert isinstance(hidden_states, QTensor)
            input_quant, input_scale = (hidden_states.tensor, hidden_states.scale)

        expert_offset = 0
        num_experts = None
        if expert_list is not None and len(expert_list) != self.num_experts:
            expert_offset = expert_list[0]
            num_experts = self.num_experts
        return fused_moe_w8a8(input_quant,
                              input_scale,
                              gate_up_weights,
                              gate_up_scale,
                              down_weights,
                              down_scale,
                              topk_weights=topk_weights,
                              topk_ids=topk_ids,
                              topk=self.top_k,
                              out_dtype=self.out_dtype,
                              quant_dtype=self.quant_dtype,
                              expert_offset=expert_offset,
                              num_experts=num_experts,
                              renormalize=self.renormalize)


class TritonFusedMoEW8A8Builder(FusedMoEW8A8Builder):
    """triton fused moe w8a8 builder."""

    @staticmethod
    def build(
        top_k: int,
        num_experts: int,
        renormalize: bool = False,
        out_dtype: torch.dtype = torch.float16,
        quant_dtype: torch.dtype = torch.int8,
    ):
        """build from mlp."""
        return TritonFusedMoEW8A8Impl(top_k=top_k,
                                      num_experts=num_experts,
                                      renormalize=renormalize,
                                      out_dtype=out_dtype,
                                      quant_dtype=quant_dtype)


class TritonFusedMoEBlockedF8Impl(FusedMoEBlockedF8Impl):
    """triton fused moe blocked f8 implementation."""

    def __init__(self,
                 top_k: int,
                 num_experts: int,
                 renormalize: bool = False,
                 block_size: int = 128,
                 out_dtype: torch.dtype = torch.float16):
        self.num_experts = num_experts
        self.top_k = top_k
        self.renormalize = renormalize
        self.block_size = block_size
        self.out_dtype = out_dtype

    def support_ep(self):
        """support expert parallelism."""
        return True

    def ep_expert_list(self, world_size: int, rank: int):
        """experts list of current rank."""
        num_experts = self.num_experts
        expert_per_rank = (num_experts + world_size - 1) // world_size
        first_expert = rank * expert_per_rank
        last_expert = min(first_expert + expert_per_rank, num_experts)
        return list(range(first_expert, last_expert))

    def forward(self,
                hidden_states: torch.Tensor,            # [batch_size*seq_len, hidden_dim] 输入特征             [11, 7168]
                topk_weights: torch.Tensor,             # [batch_size*seq_len, topk] 前k专家的权重系数           [11, 8]
                topk_ids: torch.LongTensor,             # [batch_size*seq_len, topk] 前k个专家的索引             [11, 8]
                gate_up_weights: torch.Tensor,          # [num_experts, intermediate_dim*2, hidden_dim] 输入特征到门控网络的权重                                 [256, 4096, 7168]
                gate_up_scale: torch.Tensor,            # [num_experts, intermediate_dim*2//block_size, hidden_dim//block_size] 输入特征到门控网络的缩放因子     [256, 32, 56]
                down_weights: torch.Tensor,             # [num_experts, hidden_dim, intermediate_dim] 门控网络到输出的权重                                       [256, 7168, 2048]
                down_scale: torch.Tensor,               # [num_experts, hidden_dim//block_size, intermediate_dim//block_size] 门控网络到输出的缩放因子           [256, 56, 16]
                expert_list: List[int] = None):         # [num_experts] 专家的索引列表                                                      None
        """forward."""
        """
        hidden_states 7168
        intermediate_dim 2048
        """
        
        # logger.error(f"ep_rank {ep_rank} zmz hidden_states.shape: {hidden_states.shape}, topk_ids.shape: {topk_ids.shape}, topk_weights.shape: {topk_weights.shape}, gate_up_weights.shape: {gate_up_weights.shape}, gate_up_scale.shape: {gate_up_scale.shape}, down_weights.shape: {down_weights.shape}, down_scale.shape: {down_scale.shape}")
        use_triton = os.getenv('ZMZ_USE_TRITON_IMPL', '0') == '1'
        if not use_triton:
            pass

        # assert False, "zmz debug"
        # if hidden_states.shape[0] != -1234:
        #     logger.error(f"ep_rank {ep_rank} zmz hidden_states: {hidden_states.shape}, topk_ids: {topk_ids.shape}, topk_weights: {topk_weights.shape}, gate_up_weights: {gate_up_weights.shape}, gate_up_scale: {gate_up_scale.shape}, down_weights: {down_weights.shape}, down_scale: {down_scale.shape}")
        #     logger.error(f"ep_rank {ep_rank} zmz expert_list: {expert_list}, topk_ids: {topk_ids}, topk_weights: {topk_weights}")

        input_size = hidden_states.shape                # [batch_size*seq_len, hidden_dim]
        hidden_states = hidden_states.flatten(0, -2)    # [batch_size*seq_len, hidden_dim]
        # 量化成fp8
        input_quant, input_scale = quant_fp8(hidden_states, self.block_size, dtype=gate_up_weights.dtype)

        expert_offset = 0
        num_experts = None
        if expert_list is not None and len(expert_list) != self.num_experts:
            expert_offset = expert_list[0]
            num_experts = self.num_experts
        logger.error(f"ep_rank {ep_rank} zmz debug expert_offset: {expert_offset}, num_experts: {num_experts}, topk_ids: {topk_ids}")
        output = fused_moe_blocked_fp8(input_quant,
                                       input_scale,
                                       gate_up_weights,
                                       gate_up_scale,
                                       down_weights,
                                       down_scale,
                                       topk_weights=topk_weights,
                                       topk_ids=topk_ids,
                                       topk=self.top_k,
                                       out_dtype=hidden_states.dtype,
                                       expert_offset=expert_offset,
                                       num_experts=num_experts,
                                       renormalize=self.renormalize)
        output = output.unflatten(0, input_size[:-1])
        # out_states = output
        # if hidden_states.shape[0] != -1234:
        #     logger.error(f"ep_rank {ep_rank} zmz super_out_states.shape: {out_states.shape},")
        #     # torch.save(out_states, "ep2_base_out_states.pt", cpu=True).to(device)
        #     logger.error(f"ep_rank {ep_rank} zmz super_out_states: {out_states}")

        if not use_triton:
            # raise Exception("zmz debug")
            pass
        return output


class DeepEPMoE:
    """MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-
    ai/DeepEP/tree/main)"""

    def __init__(
        self,
        num_experts: int,
        ep_size: int,
        block_shape: list[int],
    ):
        self.num_experts = num_experts
        self.ep_size = ep_size
        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.block_shape = block_shape
        self.use_fp8_w8a8 = True

    def forward(self, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor, gate_up_weight: torch.Tensor,
                gate_up_scale: torch.Tensor, gate_down_weight: torch.Tensor, gate_down_scale: torch.Tensor):
        seg_indptr_cur_rank = torch.cat([
            torch.zeros(1, device=tokens_per_expert.device, dtype=tokens_per_expert.dtype),
            torch.cumsum(tokens_per_expert, dim=0),
        ])
        reorder_topk_ids = torch.repeat_interleave(tokens_per_expert)
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        # GroupGemm-0
        gateup_output = torch.empty(
            hidden_states.shape[0],
            gate_up_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if hidden_states.shape[0] > 0:
            input, input_scale = quant_fp8(hidden_states, 128, dtype=gate_up_weight.dtype)
            gateup_output = grouped_gemm_triton(
                a=input,
                b=gate_up_weight,
                c=gateup_output,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr_cur_rank,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=input_scale,
                scale_b=gate_up_scale,
                block_shape=self.block_shape,
            )

        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=hidden_states.dtype,
        )
        silu_and_mul_triton_kernel[(gateup_output.shape[0], )](
            gateup_output,
            down_input,
            gateup_output.shape[1],
            reorder_topk_ids,
            None,
            0,
            self.num_experts_per_partition - 1,
            BLOCK_SIZE=512,
        )

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            gate_down_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if down_input.shape[0] > 0:
            down_input, down_input_scale = quant_fp8(down_input, 128, dtype=gate_down_weight.dtype)
            down_output = grouped_gemm_triton(
                a=down_input,
                b=gate_down_weight,
                c=down_output,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr_cur_rank,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=down_input_scale,
                scale_b=gate_down_scale,
                block_shape=self.block_shape,
            )
        return down_output

def _log_tensor_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, name1: str, name2: str):
    """记录张量差异的辅助函数"""
    logger.error(f"{name1} shape: {tensor1.shape if tensor1 is not None else None} "
                 f"vs {name2} shape: {tensor2.shape if tensor2 is not None else None}")
    logger.error(f"{name1} device: {tensor1.device if tensor1 is not None else None} "
                 f"vs {name2} device: {tensor2.device if tensor2 is not None else None}")
    if tensor1 is not None and tensor2 is not None:
        logger.error(f"{name1}[:3]: {tensor1[:3].cpu().detach()}")
        logger.error(f"{name2}[:3]: {tensor2[:3].cpu().detach()}")
class FusedDeepEpMoEBlockedF8Impl(TritonFusedMoEBlockedF8Impl):

    def __init__(self,
                 ep_size: int,
                 ep_group: dist.ProcessGroup,
                 top_k: int,
                 num_experts: int,
                 hidden_dim: int,
                 renormalize: bool = False,
                 block_size: int = 128,
                 out_dtype: torch.dtype = torch.bfloat16):
        super().__init__(top_k, num_experts, renormalize, block_size, out_dtype)
        
        # 新增Triton实现实例
        self.triton_impl = TritonFusedMoEBlockedF8Impl(
            top_k=top_k,
            num_experts=num_experts,
            renormalize=renormalize,
            block_size=block_size,
            out_dtype=out_dtype
        )
        self.token_dispatcher = DeepEPDispatcher(
            group=ep_group,
            num_experts=self.num_experts,
            num_local_experts=self.num_experts // ep_size,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
        )
        self.experts = DeepEPMoE(num_experts, ep_size, [block_size, block_size])

    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        # 在ep=2场景下，输入了所有的token进来，但是experts只有一半。
        # 对于不存在的experts，topk_ids当前是所有，经过下面的dispatch后，会变成一半。
        # 所以，这里需要对topk_ids进行处理，只保留当前rank的experts。
        # 这里的expert_list是当前rank的experts。

        # ep, ep_rank = get_ep_world_rank()
        # expert is first
        use_triton = os.getenv('ZMZ_USE_TRITON_IMPL', '0') == '1'
        if use_triton:
            # logger.error(f"ep_rank {ep_rank} zmz use triton impl")
            # if not topk_weights.is_contiguous():
            #     topk_weights = topk_weights.contiguous()
            pass
        else:
            topk_weights = _renormalize(topk_weights, self.renormalize)

        # 数据收集
        recv_hidden_states, recv_topk_ids, recv_topk_weights, tokens_per_expert = (self.token_dispatcher.dispatch(
            hidden_states,
            topk_ids,
            topk_weights.to(torch.float32),
            self.num_experts,
        ))

        logger.error(f"ep_rank {ep_rank} zmz debug recv_hidden_states.shape: {recv_hidden_states.shape}, recv_topk_ids.shape: {recv_topk_ids.shape}, recv_topk_weights.shape: {recv_topk_weights.shape}, tokens_per_expert.shape: {tokens_per_expert.shape}, gate_up_weights.shape: {gate_up_weights.shape}, gate_up_scale.shape: {gate_up_scale.shape}, down_weights.shape: {down_weights.shape}, down_scale.shape: {down_scale.shape}")
        logger.error(f"ep_rank {ep_rank} zmz debug recv_topk_ids: {recv_topk_ids}, recv_topk_weights: {recv_topk_weights}")
        device = hidden_states.device
        
        if use_triton:
            
            if ep_rank == -1:
                load_hidden_state = torch.load("ep2_base_hidden_states.pt", map_location='cpu').to(device)
                load_topk_weights = torch.load("ep2_base_topk_weights.pt", map_location='cpu').to(device)
                load_topk_ids = torch.load("ep2_base_topk_ids.pt", map_location='cpu').to(device)
                load_gate_up_weights = torch.load("ep2_base_gate_up_weights.pt", map_location='cpu').to(device)
                # load_gate_up_scale = torch.load("ep2_base_gate_up_scale.pt", map_location='cpu').to(device)
                # load_down_weights = torch.load("ep2_base_down_weights.pt", map_location='cpu').to(device)
                # load_down_scale = torch.load("ep2_base_down_scale.pt", map_location='cpu').to(device)
                load_expert_list = torch.load("ep2_base_expert_list.pt")
                # 调试建议：在断言前添加维度/数据类型/设备检查
                logger.error(f"load_hidden_shape: {load_hidden_state.shape} vs recv_shape: {recv_hidden_states.shape}")
                logger.error(f"load_hidden_device: {load_hidden_state.device} vs recv_device: {recv_hidden_states.device}")
                logger.error(f"load_hidden_dtype: {load_hidden_state.dtype} vs recv_dtype: {recv_hidden_states.dtype}")
                
                # 数值对比建议：打印前几个元素的差异
                _log_tensor_diff(load_hidden_state, recv_hidden_states, "load_hidden", "recv_hidden")
                # logger.error(f"load_hidden[:3]: {load_hidden_state[:3].cpu()}")
                # logger.error(f"recv_hidden[:3]: {recv_hidden_states[:3].cpu()}")
                assert torch.allclose(load_hidden_state, recv_hidden_states, atol=1e-3, rtol=1e-3)
                _log_tensor_diff(load_topk_weights, recv_topk_weights, "load_topk_weights", "recv_topk_weights")
                assert torch.allclose(load_topk_weights, recv_topk_weights, atol=1e-3, rtol=1e-3)
                # 有问题1，需要都指定ep0去比较
                _log_tensor_diff(load_topk_ids, recv_topk_ids, "load_topk_ids", "recv_topk_ids")
                assert torch.allclose(load_topk_ids, recv_topk_ids, atol=1e-3, rtol=1e-3)
                # 有问题2：RuntimeError: "mul_cuda" not implemented for 'Float8_e4m3fn'
                # _log_tensor_diff(load_gate_up_weights, gate_up_weights, "load_gate_up_weights", "gate_up_weights")
                # assert torch.allclose(load_gate_up_weights.to(torch.float16), gate_up_weights.to(torch.float16), atol=1e-3, rtol=1e-3)
                # _log_tensor_diff(load_gate_up_scale, gate_up_scale, "load_gate_up_scale", "gate_up_scale")
                # assert torch.allclose(load_gate_up_scale.to(torch.float16), gate_up_scale.to(torch.float16), atol=1e-1, rtol=1e-1)
                # _log_tensor_diff(load_down_weights, down_weights, "load_down_weights", "down_weights")
                # assert torch.allclose(load_down_weights.to(torch.float16), down_weights.to(torch.float16), atol=1e-1, rtol=1e-1)
                # _log_tensor_diff(load_down_scale, down_scale, "load_down_scale", "down_scale")
                # assert torch.allclose(load_down_scale.to(torch.float16), down_scale.to(torch.float16), atol=1e-1, rtol=1e-1)
                # _log_tensor_diff(load_expert_list, expert_list, "load_expert_list", "expert_list")
                logger.error(f"load_expert_list: {load_expert_list} vs expert_list: {expert_list}")
                assert all(a == b for a, b in zip(load_expert_list, expert_list)), "List content mismatch"

            # logger.error(f"ep_rank {ep_rank} zmz debug before expert_list: {expert_list}")
            out_states0 = self.triton_impl.forward(recv_hidden_states, recv_topk_weights, recv_topk_ids, gate_up_weights,
                                                   gate_up_scale, down_weights, down_scale, expert_list=expert_list)
            if ep_rank == -1:
                load_out_states0 = torch.load("ep2_base_output.pt", map_location='cpu').to(device)
                _log_tensor_diff(out_states0, load_out_states0, "out_states0", "load_out_states0")
                assert torch.allclose(out_states0, load_out_states0, atol=0.05, rtol=0.05)
                max_diff = torch.max(torch.abs(out_states0 - load_out_states0))
                logger.error(f"ep_rank {ep_rank} zmz debug atol 0.05 ok, max_diff: {max_diff}")
                # assert torch.allclose(out_states0, load_out_states0, atol=1e-3, rtol=1e-3)
            logger.error(f"ep_rank {ep_rank} zmz debug ok")
        else:
            if ep_rank == -1:
                torch.save(recv_hidden_states, "ep2_base_hidden_states.pt")
                torch.save(recv_topk_weights, "ep2_base_topk_weights.pt")
                torch.save(recv_topk_ids, "ep2_base_topk_ids.pt")
                torch.save(gate_up_weights, "ep2_base_gate_up_weights.pt")
                torch.save(gate_up_scale, "ep2_base_gate_up_scale.pt")
                torch.save(down_weights, "ep2_base_down_weights.pt")
                torch.save(down_scale, "ep2_base_down_scale.pt")
                # torch.save(tokens_per_expert, "ep2_base_tokens_per_expert.pt")
                torch.save(expert_list, "ep2_base_expert_list.pt")
                logger.error(f"ep_rank {ep_rank} zmz debug save ok")
            if recv_hidden_states.shape[0] > 0:
                recv_hidden_states = self.token_dispatcher.get_permuted_hidden_states_by_experts(recv_hidden_states)
            # else:
            #     logger.error(f"ep_rank {ep_rank} zmz debug shape[0] == 0, recv_hidden_states.shape: {recv_hidden_states.shape}")
            out_states0 = self.experts.forward(recv_hidden_states, tokens_per_expert, gate_up_weights, gate_up_scale,
                                               down_weights, down_scale)
            if out_states0.shape[0] > 0:
                out_states0 = self.token_dispatcher.get_restored_hidden_states_by_experts(out_states0)
            if ep_rank == -1:
                torch.save(out_states0, "ep2_base_output.pt")
            # else:
            #     logger.error(f"ep_rank {ep_rank} zmz debug shape[0] == 0, out_states0.shape: {out_states0.shape}")

        # 数据合并
        out_states = self.token_dispatcher.combine(out_states0)
        # if use_triton:
        #     load_out_states = torch.load("ep2_base_output.pt", cpu=True).to(device)
        #     assert torch.allclose(out_states, load_out_states)
        #     logger.error(f"ep_rank {ep_rank} zmz debug ok")
        # raise Exception("zmz debug")

        return out_states


class TritonFusedMoEBlockedF8Builder(FusedMoEBlockedF8Builder):
    """triton fused moe blocked f8 builder."""

    @staticmethod
    def build(top_k: int,
              num_experts: int,
              hidden_dim: int = 1,
              renormalize: bool = False,
              block_size: int = 128,
              ep_size: int = 1,
              ep_group: dist.ProcessGroup = None,
              out_dtype: torch.dtype = torch.float16):
        """build from mlp."""
        if ep_size > 1:
        # if ep_size == -100:
            logger.error(f"zmz FusedDeepEpMoEBlockedF8Impl")
            return FusedDeepEpMoEBlockedF8Impl(ep_size=ep_size,
                                               ep_group=ep_group,
                                               top_k=top_k,
                                               num_experts=num_experts,
                                               hidden_dim=hidden_dim,
                                               renormalize=renormalize,
                                               block_size=block_size,
                                               out_dtype=out_dtype)
        else:
            logger.error(f"zmz TritonFusedMoEBlockedF8Impl")
            return TritonFusedMoEBlockedF8Impl(top_k=top_k,
                                               num_experts=num_experts,
                                               renormalize=renormalize,
                                               block_size=block_size,
                                               out_dtype=out_dtype)
