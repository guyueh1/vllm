# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import itertools
from dataclasses import dataclass, field

from vllm.forward_context import MoEMetadata
from vllm.logger import init_logger
from vllm.logprobs import (
    PromptLogprobs,
    SampleLogprobs,
    append_logprobs_for_next_position,
    create_prompt_logprobs,
    create_sample_logprobs,
)
from vllm.tokenizers.detokenizer_utils import (
    TokenizerLike,
    convert_ids_list_to_tokens,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

logger = init_logger(__name__)

NONES = itertools.repeat(None)


@dataclass
class LogprobsProcessor:
    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: TokenizerLike | None

    # Logprobs for this request
    logprobs: SampleLogprobs | None
    prompt_logprobs: PromptLogprobs | None
    cumulative_logprob: float | None
    num_logprobs: int | None
    num_prompt_logprobs: int | None

    sample_moe_topk_indices: list[str] | list[list[list[int]]] | None = field(default_factory=list)
    prompt_moe_topk_indices: list[str] | list[list[list[int]]] | None = field(default_factory=list)
    moe_metadata: MoEMetadata | None = None

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        sampling_params = request.sampling_params
        assert sampling_params is not None
        num_logprobs = sampling_params.logprobs
        num_prompt_logprobs = sampling_params.prompt_logprobs
        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.0),
            logprobs=(
                None
                if num_logprobs is None
                else create_sample_logprobs(sampling_params.flat_logprobs)
            ),
            prompt_logprobs=(
                None
                if num_prompt_logprobs is None
                else create_prompt_logprobs(sampling_params.flat_logprobs)
            ),
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
        )

    def _postproc_topk_indices(
        self, topk_indices: list[list[int]] | str
    ) -> list[list[int]] | str:
        if isinstance(topk_indices, bytes) or isinstance(topk_indices, str):
            return topk_indices

        assert self.moe_metadata is not None
        expert_bits = self.moe_metadata.calculate_expert_bits()

        bitmask = np.zeros((bits_per_pos + 7) // 8, dtype=np.uint8)
        bit_start = 0
        for exp_lst in topk_indices:
            for k in exp_lst:
                byte_mid = (bit_start + 7) >> 3
                bit_mid = byte_mid << 3
                lo_bits = bit_mid - bit_start
                bitmask[bit_start >> 3] |= (k << (bit_start & 7)) & 0xff
                bitmask[byte_mid] |= (k >> lo_bits) & 0xff
                bit_start += expert_bits
        return base64.b64encode(bitmask.data)

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

        """

        assert self.num_logprobs is not None
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        token_ids_lst = logprobs_lists.logprob_token_ids
        logprobs_lst = logprobs_lists.logprobs
        ranks_lst = logprobs_lists.sampled_token_ranks
        moe_topk_indices_lst = logprobs_lists.moe_topk_indices

        for rank_np, logprobs_np, token_ids_np, moe_topk_indices_np in zip(
            ranks_lst, logprobs_lst, token_ids_lst, moe_topk_indices_lst
        ):
            rank = rank_np.tolist()
            logprobs = logprobs_np.tolist()
            token_ids = token_ids_np.tolist()
            moe_topk_indices = self._postproc_topk_indices(
                moe_topk_indices_np.tolist()
            )
            # Detokenize (non-incrementally).
            decoded_tokens = (
                NONES
                if self.tokenizer is None
                else (convert_ids_list_to_tokens(self.tokenizer, token_ids))
            )

            # Sampler puts the sampled logprob in first.
            sampled_token_logprob = logprobs[0]
            self.cumulative_logprob += sampled_token_logprob

            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(
                self.logprobs,
                token_ids,
                logprobs,
                decoded_tokens,
                rank,
                self.num_logprobs,
            )

            self.sample_moe_topk_indices.append(moe_topk_indices)

    def _update_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
    ) -> None:
        """Update with prompt logprobs from EngineCore.

        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.

        """

        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None

        token_ids = prompt_logprobs_tensors.logprob_token_ids
        logprobs = prompt_logprobs_tensors.logprobs
        ranks = prompt_logprobs_tensors.selected_token_ranks
        moe_topk_indices = prompt_logprobs_tensors.moe_topk_indices

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        decoded_tokens = (
            None
            if self.tokenizer is None
            else (
                convert_ids_list_to_tokens(self.tokenizer, token_ids.flatten().tolist())
            )
        )

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Pythonize the torch tensors.
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids = token_ids.tolist()
        prompt_moe_topk_indices = moe_topk_indices.tolist()

        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening.
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs
            decoded_tokens_for_pos = (
                NONES if decoded_tokens is None else decoded_tokens[offset:offset_end]
            )

            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(
                self.prompt_logprobs,
                token_ids[pos],
                prompt_logprobs[pos],
                decoded_tokens_for_pos,
                prompt_token_ranks[pos],
                self.num_prompt_logprobs,
            )

            self.prompt_moe_topk_indices.append(
                self._postproc_topk_indices(prompt_moe_topk_indices[pos])
            )

    def pop_prompt_logprobs(self) -> PromptLogprobs | None:
        """Pop and return all request prompt logprobs

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.
        """
        plp = self.prompt_logprobs
        if plp:
            self.prompt_logprobs = []
        return plp

    def pop_prompt_moe_topk_indices(
        self,
    ) -> list[str] | list[list[list[int]]] | None:
        """Pop and return all request prompt logprobs

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.
        """
        plp = self.prompt_moe_topk_indices
        if plp:
            self.prompt_moe_topk_indices = []
        return plp

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if self.moe_metadata is None and output.moe_metadata is not None:
            self.moe_metadata = output.moe_metadata
        # EngineCoreOutput carries per-request logprob slices from the scheduler.
        # This conversion is identical across eager and compiled/cudagraph modes.
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            # Prompt logprobs arrive as tensors from prefill and are
            # accumulated into the request's prompt_logprobs list.
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
