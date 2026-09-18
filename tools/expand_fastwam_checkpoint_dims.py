# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Expand the FastWAM base checkpoint to a new action / proprio dim.

``FastWAMVLA`` loads its ``.safetensors`` checkpoint with
``load_state_dict(strict=True)``, so a config with a different
``action_dim`` / ``proprio_dim`` (e.g. RoboTwin's 14-D dual-arm joint
space vs. the LIBERO 7-D / 8-D base) cannot load the released base
weights directly. Only three parameter groups depend on those dims:

* ``vla_head.mot.mixtures.action.action_encoder.weight``  [H, action_dim]
* ``vla_head.mot.mixtures.action.head.{weight,bias}``     [action_dim, H]
* ``vla_head.proprio_encoder.weight``                     [H, proprio_dim]

This tool widens them: new input columns and new output rows/bias
entries are zero-initialized, so the expanded checkpoint behaves
identically to the original on the first ``min(old, new)`` dims and is
finetuned from zero on the added dims.

Usage:
    python tools/expand_fastwam_checkpoint_dims.py \\
      checkpoints/fastwam_base_full/fastwam_base_full.safetensors \\
      checkpoints/fastwam_base_full/fastwam_base_full_robotwin14.safetensors \\
      --action-dim 14 --proprio-dim 14
"""

import argparse

import torch
from safetensors.torch import load_file, save_file

ACTION_ENCODER_WEIGHT = ('vla_head.mot.mixtures.action.action_encoder.weight')
ACTION_HEAD_WEIGHT = 'vla_head.mot.mixtures.action.head.weight'
ACTION_HEAD_BIAS = 'vla_head.mot.mixtures.action.head.bias'
PROPRIO_ENCODER_WEIGHT = 'vla_head.proprio_encoder.weight'


def _pad_columns(weight: torch.Tensor, new_dim: int,
                 name: str) -> torch.Tensor:
    old_dim = weight.shape[1]
    if new_dim < old_dim:
        raise ValueError(f'{name}: cannot shrink dim {old_dim} -> {new_dim}')
    if new_dim == old_dim:
        return weight
    pad = torch.zeros(weight.shape[0], new_dim - old_dim, dtype=weight.dtype)
    return torch.cat([weight, pad], dim=1)


def _pad_rows(weight: torch.Tensor, new_dim: int, name: str) -> torch.Tensor:
    old_dim = weight.shape[0]
    if new_dim < old_dim:
        raise ValueError(f'{name}: cannot shrink dim {old_dim} -> {new_dim}')
    if new_dim == old_dim:
        return weight
    pad_shape = (new_dim - old_dim, ) + tuple(weight.shape[1:])
    pad = torch.zeros(pad_shape, dtype=weight.dtype)
    return torch.cat([weight, pad], dim=0)


def expand_checkpoint(in_path: str, out_path: str, action_dim: int,
                      proprio_dim: int) -> None:
    state_dict = load_file(in_path, device='cpu')

    for key in (ACTION_ENCODER_WEIGHT, ACTION_HEAD_WEIGHT, ACTION_HEAD_BIAS,
                PROPRIO_ENCODER_WEIGHT):
        if key not in state_dict:
            raise KeyError(f'Expected parameter not found: {key}')

    old = {key: tuple(state_dict[key].shape) for key in state_dict}
    state_dict[ACTION_ENCODER_WEIGHT] = _pad_columns(
        state_dict[ACTION_ENCODER_WEIGHT], action_dim, ACTION_ENCODER_WEIGHT)
    state_dict[ACTION_HEAD_WEIGHT] = _pad_rows(state_dict[ACTION_HEAD_WEIGHT],
                                               action_dim, ACTION_HEAD_WEIGHT)
    state_dict[ACTION_HEAD_BIAS] = _pad_rows(state_dict[ACTION_HEAD_BIAS],
                                             action_dim, ACTION_HEAD_BIAS)
    state_dict[PROPRIO_ENCODER_WEIGHT] = _pad_columns(
        state_dict[PROPRIO_ENCODER_WEIGHT], proprio_dim,
        PROPRIO_ENCODER_WEIGHT)

    for key in (ACTION_ENCODER_WEIGHT, ACTION_HEAD_WEIGHT, ACTION_HEAD_BIAS,
                PROPRIO_ENCODER_WEIGHT):
        print(f'{key}: {old[key]} -> {tuple(state_dict[key].shape)}')

    save_file(state_dict, out_path)
    print(f'Saved expanded checkpoint to: {out_path}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='Path to the source .safetensors')
    parser.add_argument('output', help='Path for the expanded .safetensors')
    parser.add_argument(
        '--action-dim', type=int, required=True, help='Target action dim')
    parser.add_argument(
        '--proprio-dim', type=int, required=True, help='Target proprio dim')
    args = parser.parse_args()
    expand_checkpoint(args.input, args.output, args.action_dim,
                      args.proprio_dim)


if __name__ == '__main__':
    main()
