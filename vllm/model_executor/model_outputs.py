# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model forward output containers."""

from typing import NamedTuple

import torch

from vllm.sequence import IntermediateTensors


class ModelForwardOutput(NamedTuple):
    """Normalized model forward output for spec/non-spec decode paths."""

    hidden_states: torch.Tensor | IntermediateTensors
    aux_hidden_states: list[torch.Tensor] | None
    moe_topk_indices: list[torch.Tensor] | None
