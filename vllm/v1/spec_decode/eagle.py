# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import eagle_step_update_slot_mapping_and_metadata


class EagleProposer(SpecDecodeBaseProposer):
    """EAGLE/MTP proposer with optional DDTree speculative decoding.

    Inherits all baseline MTP chain-drafting logic from
    SpecDecodeBaseProposer and overrides propose() to add DDTree
    (Draft-Decision Tree) support when enabled in the speculative config.

    DDTree builds a multi-branch tree from top-K logits at each MTP step,
    improving acceptance rates under stochastic sampling compared to
    single-chain drafting.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        # DDTree: pre-allocate fixed-size GPU buffer for ancestor-only
        # attention bias.  Shape: (1 + max_spec, 1 + max_spec), always
        # the same tensor identity so CUDA graph replay sees a stable
        # data_ptr.  Content is updated in-place via copy_() each step.
        # During non-DDTree decode steps the buffer stays zero (no effect).
        if self.speculative_config.use_ddtree():
            sz = 1 + self.num_speculative_tokens
            self._ddtree_qq_buf = torch.zeros(
                (sz, sz), dtype=torch.float32, device=device
            )
            self._ddtree_child_maps: list = []  # per-request child maps
        else:
            self._ddtree_qq_buf = None

    # ------------------------------------------------------------------ #
    # DDTree propose() override
    # ------------------------------------------------------------------ #
    # Only overrides propose() when DDTree is enabled; otherwise the
    # base class chain-drafting propose() is used unchanged.
    # ------------------------------------------------------------------ #

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        # Fall back to base class for non-DDTree mode.
        if not self.speculative_config.use_ddtree():
            return super().propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                sampling_metadata=sampling_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
            )

        # ---- DDTree propose ---- #
        batch_size = common_attn_metadata.batch_size()
        spec_config = self.speculative_config
        ddtree_topk = spec_config.ddtree_topk
        mtp_loop_limit = (
            spec_config.ddtree_mtp_depth - 1
            if spec_config.ddtree_mtp_depth > 0
            else self.num_speculative_tokens - 1
        )

        # Collect top-K log-probs at each MTP step for tree building.
        ddtree_top_log_probs: list[torch.Tensor] = []
        ddtree_top_token_ids: list[torch.Tensor] = []

        # --- First pass: run draft model on target hidden states --- #
        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )

        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        sample_hidden_states = last_hidden_states[token_indices_to_sample]

        # Compute logits once for both greedy sample and top-K extraction.
        logits = self.model.compute_logits(sample_hidden_states)
        draft_token_ids = logits.argmax(dim=-1)
        topk_logits, topk_ids = torch.topk(
            logits.float(), k=ddtree_topk, dim=-1
        )
        ddtree_top_log_probs.append(topk_logits[:batch_size])
        ddtree_top_token_ids.append(topk_ids[:batch_size])

        if self.uses_mrope:
            positions = self.mrope_positions[:, token_indices_to_sample]
        else:
            positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]

        draft_token_ids_list = [draft_token_ids]

        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(batch_size)
        )

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens_gpu
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        block_size = self.block_size
        assert block_size > 0, "block_size has not been initialized."

        # --- MTP loop: limited to mtp_loop_limit (decoupled from tree budget) --- #
        for token_index in range(mtp_loop_limit):
            input_ids = draft_token_ids_list[-1].int()
            positions_1d = positions[0] if self.uses_mrope else positions
            if self.uses_mrope:
                out_pos = self.mrope_positions[0, :batch_size]
            elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                out_pos = self.xdrope_positions[0, :batch_size]
            else:
                out_pos = self.positions[:batch_size]
            eagle_step_update_slot_mapping_and_metadata(
                positions_1d=positions_1d,
                block_table_tensor=common_attn_metadata.block_table_tensor,
                seq_lens=common_attn_metadata.seq_lens,
                block_size=block_size,
                max_model_len=self.max_model_len,
                out_clamped_positions=out_pos,
                out_slot_mapping=self._slot_mapping_buffer[:input_batch_size],
                input_batch_size=input_batch_size,
            )
            common_attn_metadata.slot_mapping = (
                self._slot_mapping_buffer[:batch_size]
            )
            if self.uses_mrope:
                self.mrope_positions[1:, :batch_size] = self.mrope_positions[
                    0, :batch_size
                ]
                positions = self.mrope_positions[:, :batch_size]
            elif self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim > 0:
                self.xdrope_positions[1:, :batch_size] = self.xdrope_positions[
                    0, :batch_size
                ]
                positions = self.xdrope_positions[0, :batch_size]
            else:
                positions = self.positions[:batch_size]

            common_attn_metadata.max_seq_len = min(
                common_attn_metadata.max_seq_len + 1, self.max_model_len
            )
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu += 1
            if common_attn_metadata._num_computed_tokens_cpu is not None:
                common_attn_metadata._num_computed_tokens_cpu += 1
            if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
                common_attn_metadata.seq_lens_cpu_upper_bound += 1

            _, per_layer_attn_metadata = (
                self.build_per_group_and_layer_attn_metadata(
                    common_attn_metadata, draft_index=token_index + 1
                )
            )

            self.input_ids[:batch_size] = input_ids
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = (
                    self.model.embed_input_ids(input_ids)
                )
                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            model_kwargs = {
                "input_ids": input_ids,
                "positions": self._get_positions(input_batch_size),
                "inputs_embeds": inputs_embeds,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = (
                    self.hidden_states[:input_batch_size]
                )

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),
            ):
                ret_hidden_states = self.model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states[:batch_size]

            # DDTree: compute logits for both greedy and top-K.
            logits = self.model.compute_logits(last_hidden_states[:batch_size])
            draft_token_ids = logits.argmax(dim=-1)
            topk_logits, topk_ids = torch.topk(
                logits.float(), k=ddtree_topk, dim=-1
            )
            ddtree_top_log_probs.append(topk_logits)
            ddtree_top_token_ids.append(topk_ids)
            draft_token_ids_list.append(draft_token_ids)

        # --- Build DDTree from collected top-K log-probs --- #
        from vllm.v1.spec_decode.ddtree import build_ddtree, ddtree_to_qq_bias

        all_tree_token_ids = []
        self._ddtree_child_maps = []
        self._ddtree_qq_buf.zero_()

        for req_idx in range(batch_size):
            tp = torch.stack(
                [t[req_idx] for t in ddtree_top_log_probs], dim=0
            )
            ti = torch.stack(
                [t[req_idx] for t in ddtree_top_token_ids], dim=0
            )
            tree = build_ddtree(
                tp,
                ti,
                budget=spec_config.ddtree_budget,
                chain_seed=spec_config.ddtree_chain_seed,
            )
            all_tree_token_ids.append(tree.token_ids)
            self._ddtree_child_maps.append(tree.child_maps)
            if req_idx == 0:
                N = tree.total_slots
                qq = ddtree_to_qq_bias(tree, device=str(self.device))
                self._ddtree_qq_buf[:N, :N].copy_(qq)

        # Pad to same length and create tensor.
        max_tree_len = max(len(t) for t in all_tree_token_ids)
        if max_tree_len > 0:
            padded = torch.zeros(
                batch_size,
                max_tree_len,
                dtype=torch.int32,
                device=self.device,
            )
            for i, toks in enumerate(all_tree_token_ids):
                for j, tok in enumerate(toks):
                    padded[i, j] = tok
            return padded

        # Fallback: return chain tokens if tree is empty.
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids
