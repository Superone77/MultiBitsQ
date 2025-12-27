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
import logging
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


class ElasticQuantBinarizerSigned(torch.autograd.Function):
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
        if (alpha - 1).abs().sum() == 0.0:
            if layerwise:
                alpha = 2 * input.abs().mean() / math.sqrt(Qp)
            else:
                alpha = (
                    2
                    * torch.mean(input.abs().mean(0), dim=-1, keepdim=True)
                    / math.sqrt(Qp)
                )

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


class ElasticQuantBinarizerSignedStretched(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise, alpha_stretch):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :param alpha_stretch: stretch parameter
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

        eps = 1e-5
        if (alpha - 1).abs().sum() == 0.0:
            if layerwise:
                alpha = 2 * input.abs().mean() / math.sqrt(Qp)
            else:
                alpha = (
                    2
                    * torch.mean(input.abs().mean(0), dim=-1, keepdim=True)
                    / math.sqrt(Qp)
                )

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
        w_q = input + (w_q - input) * (alpha_stretch)
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None, None

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
        return grad_input, grad_alpha, None, None, None


class ElasticQuantN2UQ(torch.autograd.Function):
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


class ElasticQuantN2UQStreched(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise, alpha_stretch):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :param alpha_stretch: stretch parameter
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

        eps = 1e-5
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
        w_q = input + (w_q - input) * (alpha_stretch)
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None, None

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
        return grad_input, grad_alpha, None, None, None


class ElasticQuantBinarizerUnsigned(torch.autograd.Function):
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
        Qn = 0
        Qp = 2 ** (num_bits) - 1

        if num_bits == 1:
            input_ = input
            min_val = None
        else:
            min_val = input.min().item()
            input_ = input - min_val

        eps = torch.tensor(0.00001, device=alpha.device).float()
        if (alpha - 1).abs().sum() == 0.0:
            if layerwise:
                alpha = 4 * input.abs().mean() / math.sqrt(Qp)
            else:
                alpha = (
                    4
                    * torch.mean(input.abs().mean(0), dim=-1, keepdim=True)
                    / math.sqrt(Qp)
                )

        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)

        ctx.save_for_backward(input_, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise

        q_w = (input_ / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha

        if num_bits != 1:
            w_q = w_q + min_val

        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)

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
        noise_injection: bool = False,
        noise_sigma_clipvals: float = 0.001,
        noise_sigma_weights: float = 0.001,
        random_init: bool = False,
        debug: bool = False,
        layer_name: Optional[str] = None,
        gradient_accumulation_steps: int = 1,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=False)
        # w_bits_list is required
        if w_bits_list is None or len(w_bits_list) == 0:
            raise ValueError("w_bits_list must be provided. For single-bit training, use w_bits_list with one element, e.g., [2]")
        self.w_bits_list = w_bits_list
        self.cur_w_bits = w_bits_list[0]  # Default to first bit width
        self.prob_list = prob_list
        
        
        
        self.weight_layerwise = weight_layerwise
        
        # Multi-bit training parameters
        self.multiple_bits_random_assign = multiple_bits_random_assign
        self.multiple_bits_random_assign_prob = multiple_bits_random_assign_prob
        self.noise_injection = noise_injection
        self.noise_sigma_clipvals = noise_sigma_clipvals
        self.noise_sigma_weights = noise_sigma_weights
        self.random_init = random_init
        
        
        # Gradient accumulation parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._last_accumulation_step = -1  # Track which accumulation step we're in
        self._forward_counter = 0
        
        # Debug option
        self.debug = debug
        self.layer_name = layer_name
        self.logger = logging.getLogger("clm") if debug else None

        # Initialize bit usage counter
        self.bit_usage_count = {w_bits: 0 for w_bits in self.w_bits_list}

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

        if (
            self.random_init
            and len(self.w_bits_list) > 1
        ):
            # Use weighted probabilities if prob_list is provided, otherwise uniform
            if self.prob_list is not None:
                self.cur_w_bits = np.random.choice(self.w_bits_list, p=self.prob_list)
            else:
                self.cur_w_bits = np.random.choice(self.w_bits_list)
        else:
            self.cur_w_bits = self.cur_w_bits
        
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

    def get_bit_usage_stats(self):
        """Get statistics about bit width usage.
        
        Returns:
            dict: A dictionary mapping bit width to usage count
        """
        return self.bit_usage_count.copy()

    def reset_bit_usage_stats(self):
        """Reset bit usage statistics."""
        self.bit_usage_count = {w_bits: 0 for w_bits in self.w_bits_list}

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight
        
        # Track forward passes globally to maintain consistent bit width within accumulation steps
        _global_gradient_accumulation_steps = 0
        # Update global gradient accumulation steps if this layer has a different value
        if self.gradient_accumulation_steps > 1:
            _global_gradient_accumulation_steps = self.gradient_accumulation_steps
        
        # Calculate current accumulation step using current counter value
        # We increment the counter at the end, so this forward pass belongs to the current accumulation step
        
        if _global_gradient_accumulation_steps > 1 and self.multiple_bits_random_assign:
            current_accumulation_step = self._forward_counter % _global_gradient_accumulation_steps
            # Check if we're in a new accumulation step
            if current_accumulation_step == 0:
                self._forward_counter = 0
                # Select bit width for this accumulation step
                if (
                    self.multiple_bits_random_assign
                    and len(self.w_bits_list) > 1
                    and np.random.rand() < self.multiple_bits_random_assign_prob
                ):
                    # Use weighted probabilities if prob_list is provided, otherwise uniform
                    if self.prob_list is not None:
                        self.cur_w_bits = np.random.choice(self.w_bits_list, p=self.prob_list)
                    else:
                        self.cur_w_bits = np.random.choice(self.w_bits_list)
                else:
                    self.cur_w_bits = self.cur_w_bits
            
            # Use the bit width selected for this accumulation step
            w_bits = self.cur_w_bits
            self._forward_counter += 1
        else:
            # No gradient accumulation or gradient_accumulation_steps == 1: select bit width per forward pass
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
            self.cur_w_bits = w_bits
        
        # Update bit usage count
        self.bit_usage_count[w_bits] = self.bit_usage_count.get(w_bits, 0) + 1
        
        # Increment global forward counter at the end of forward pass
        # This ensures that the next forward pass will use the correct accumulation step
        
        
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
        
        
        # Apply pre-quantization noise if enabled
        if self.noise_injection and weight_clip_val is not None:
            # Ensure weight_clip_val is on the correct device
            if isinstance(weight_clip_val, torch.Tensor):
                if weight_clip_val.device != real_weights.device:
                    weight_clip_val = weight_clip_val.to(real_weights.device)
            
            if isinstance(weight_clip_val, torch.Tensor) and not isinstance(weight_clip_val, nn.Parameter):
                # For fixed tensors (like [-2.0, 2.0] or [-5.0, 5.0])
                if weight_clip_val.dim() == 0 or (weight_clip_val.dim() == 1 and len(weight_clip_val) <= 2):
                    # Scalar or small tensor, create noise and add mean
                    noise_clip_vals = torch.randn_like(real_weights[:, :1]) * self.noise_sigma_clipvals
                    weight_clip_val = weight_clip_val + noise_clip_vals.mean()
                else:
                    # Parameter-like shape (out_features, 1)
                    noise_clip_vals = (
                        torch.randn_like(weight_clip_val) * self.noise_sigma_clipvals
                    )
                    weight_clip_val = weight_clip_val + noise_clip_vals
            else:
                # For Parameters, create noise with matching shape
                noise_clip_vals = (
                    torch.randn_like(weight_clip_val) * self.noise_sigma_clipvals
                )
                weight_clip_val = weight_clip_val + noise_clip_vals
            
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
        # Apply post-quantization noise if enabled
        if self.noise_injection:
            noise_weights = (
                torch.randn_like(self.weight) * self.noise_sigma_weights
            )
            weight = weight + noise_weights



        # Debug information printing
        if self.debug and self.logger is not None:
            with torch.no_grad():
                # Calculate weight statistics
                weight_min = weight.min().item()
                weight_max = weight.max().item()
                weight_mean = weight.mean().item()
                
                # Calculate MSE between weight and real_weights
                mse = torch.nn.functional.mse_loss(weight, real_weights).item()
                
                # Get weight_clip_val info
                if weight_clip_val is not None:
                    if isinstance(weight_clip_val, torch.Tensor):
                        if weight_clip_val.numel() == 2:
                            # Fixed tensor like [-5.0, 5.0]
                            clip_val_str = f"[{weight_clip_val[0].item():.4f}, {weight_clip_val[1].item():.4f}]"
                        else:
                            # Parameter tensor
                            clip_val_min = weight_clip_val.min().item()
                            clip_val_max = weight_clip_val.max().item()
                            clip_val_mean = weight_clip_val.mean().item()
                            clip_val_str = f"min={clip_val_min:.4f}, max={clip_val_max:.4f}, mean={clip_val_mean:.4f}"
                    else:
                        clip_val_str = str(weight_clip_val)
                else:
                    clip_val_str = "None"
                
                layer_name_str = self.layer_name if self.layer_name else "Unknown"
                self.logger.info(
                    f"[DEBUG {layer_name_str}] "
                    f"weight: min={weight_min:.6f}, max={weight_max:.6f}, mean={weight_mean:.6f}, "
                    f"MSE(weight, real_weights)={mse:.6f}, "
                    f"w_bits={w_bits}, weight_clip_val={clip_val_str}"
                )
        
        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
