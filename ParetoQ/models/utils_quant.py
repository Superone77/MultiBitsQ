# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, List


class LsqBinaryTernaryExtension(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()

        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle * (-q_w + q_w.round())
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle * (-q_w + q_w.round())
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class StretchedElasticQuant(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5
        Qp = (n_levels - shift) / n_levels
        Qn = -Qp
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (
                torch.round(
                    torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift
                )
                + shift
            ) / n_levels
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        clip_val = 1 - 1e-2
        if ctx.num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (ctx.num_bits - 1)
            shift = 0.5
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle
                            * (
                                -q_w
                                + (
                                    torch.round(
                                        torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                        - shift
                                    )
                                    + shift
                                )
                                / n_levels
                            )
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle
                        * (
                            -q_w
                            + (
                                torch.round(
                                    torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                    - shift
                                )
                                + shift
                            )
                            / n_levels
                        )
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None



class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        symmetric=True,
        bias=False,
        weight_layerwise=False,
        # Multi-bit training parameters
        w_bits_list: Optional[List[int]] = None,
        prob_list: Optional[List[float]] = None,
        multiple_bits_random_assign: bool = False,
        multiple_bits_random_assign_prob: float = 0.5,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=False)
        # w_bits_list is required
        if w_bits_list is None or len(w_bits_list) == 0:
            raise ValueError("w_bits_list must be provided. For single-bit training, use w_bits_list with one element, e.g., [2]")
        self.w_bits_list = w_bits_list
        self.cur_w_bits = w_bits_list[0]  # Default to first bit width
        
        self.weight_layerwise = weight_layerwise
        
        # Multi-bit training parameters
        self.multiple_bits_random_assign = multiple_bits_random_assign
        self.multiple_bits_random_assign_prob = multiple_bits_random_assign_prob
        
        # Store prob_list for weighted selection
        # If prob_list is None or all values are equal, use uniform distribution
        if prob_list is not None and len(prob_list) == len(self.w_bits_list):
            # Check if all probabilities are equal
            if len(set(prob_list)) == 1:
                self.prob_list = None  # Uniform distribution
            else:
                # Normalize probabilities to sum to 1
                prob_sum = sum(prob_list)
                self.prob_list = [p / prob_sum for p in prob_list] if prob_sum > 0 else None
        else:
            self.prob_list = None
        
        # Initialize weight clip values
        # Check if any bit width is less than 16
        if any(w_bits < 16 for w_bits in self.w_bits_list):
            # Separate clip values for each bit width
            self.weight_clip_val_list = {}
            for w_bits in self.w_bits_list:
                if w_bits >= 16:
                    continue
                if w_bits > 4:
                    # For higher bit widths, use fixed tensor
                    self.weight_clip_val_list[str(int(w_bits))] = torch.tensor([-5.0, 5.0])
                else:
                    param = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
                    # Initialize with zeros, will be set during training
                    with torch.no_grad():
                        param.zero_()
                    self.weight_clip_val_list[str(int(w_bits))] = param
            if len(self.weight_clip_val_list) > 0:
                self.weight_clip_val_list = nn.ParameterDict(self.weight_clip_val_list)
            # No need for weight_clip_val, use weight_clip_val_list directly
            self.weight_clip_val = None
        else:
            # For 16-bit or higher, no clip value needed
            self.weight_clip_val = None
            self.weight_clip_val_list = None
    
    def set_bits(self, w_bits: int):
        """Set the current quantization bit width."""
        if w_bits in self.w_bits_list:
            self.cur_w_bits = w_bits
        else:
            raise ValueError(f"w_bits {w_bits} not in w_bits_list {self.w_bits_list}")

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight
        
        # Select bit width for this forward pass
        if (
            self.multiple_bits_random_assign
            and len(self.w_bits_list) > 1
            and np.random.rand() < self.multiple_bits_random_assign_prob
        ):
            # Use weighted probabilities if prob_list is provided, otherwise uniform
            if self.prob_list is not None:
                w_bits = np.random.choice(self.w_bits_list, p=self.prob_list)
            else:
                w_bits = np.random.choice(self.w_bits_list)
        else:
            w_bits = self.cur_w_bits
        
        # Get clip value for current bit width
        if w_bits >= 16:
            weight_clip_val = None
        else:
            # Get from the list
            if hasattr(self, 'weight_clip_val_list') and self.weight_clip_val_list is not None:
                if str(int(w_bits)) in self.weight_clip_val_list:
                    weight_clip_val = self.weight_clip_val_list[str(int(w_bits))]
                else:
                    # Fallback: create a default clip value
                    raise NotImplementedError
                    # weight_clip_val = torch.tensor([1.0], device=real_weights.device, dtype=real_weights.dtype).expand(real_weights.shape[0], 1)
            else:
                # Fallback: create a default clip value
                raise NotImplementedError
                # weight_clip_val = torch.tensor([1.0], device=real_weights.device, dtype=real_weights.dtype).expand(real_weights.shape[0], 1)
        
        # Quantize weights
        # This logic matches quant_linear.py to ensure compatibility
        if w_bits >= 16:
            weight = self.weight
        else:
            # if weight_clip_val is None:
            #     weight_clip_val = torch.tensor(
            #         [-1.0, 1.0],
            #         device=real_weights.device,
            #         dtype=real_weights.dtype,
            #     )
            # elif isinstance(weight_clip_val, torch.Tensor) and (
            #     weight_clip_val.device != real_weights.device
            # ):
            #     weight_clip_val = weight_clip_val.to(real_weights.device)

            quantizer = (
                StretchedElasticQuant
                if w_bits in (0, 2)
                else LsqBinaryTernaryExtension
            )
            weight = quantizer.apply(
                real_weights,
                weight_clip_val,
                w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out