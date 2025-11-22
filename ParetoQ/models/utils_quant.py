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


class AsymQuantizer(torch.autograd.Function):
    """
    min-max quantization
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `_SingleLevelFunction`
    #  inconsistently.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(ctx, input, clip_val, num_bits, layerwise) -> Tensor:
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)

        # input = torch.where(input < clip_val[1], input, clip_val[1])
        # input = torch.where(input > clip_val[0], input, clip_val[0])
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # NOTE: dynamic scaling gives better performance than static
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                alpha = (
                    (
                        input.max(dim=-1, keepdim=True)[0]
                        - input.min(dim=-1, keepdim=True)[0]
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (
                    (
                        tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
                        - tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
                    )
                    .expand_as(input)
                    .detach()
                )
                beta = (
                    tmp.min(dim=-1, keepdim=True)[0]
                    .unsqueeze(-1)
                    .expand_as(input)
                    .detach()
                )
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = 2**num_bits - 1
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `_SingleLevelFunction`
    #  inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        symmetric=True,
        bias=False,
        w_bits=16,
        weight_layerwise=False,
        # Noise injection parameters
        noise_injection: bool = False,
        noise_sigma_weights: float = 0.001,
        noise_sigma_clipvals: float = 0.001,
        initialize_noise: bool = False,
        pre_quantization_noise: bool = False,
        post_quantization_noise: bool = False,
        trainable_noise_scale: bool = False,
        # Multi-bit training parameters
        w_bits_list: Optional[List[int]] = None,
        prob_list: Optional[List[float]] = None,
        multiple_bits_random_assign: bool = False,
        multiple_bits_random_assign_prob: float = 0.5,
        multiple_bits_share_clipvals: bool = False,
        multiple_bits_disable_clipvals: bool = False,
        # Stretch quantization parameters (Still unused)    
        use_stretch: bool = False,
        stretch_alpha: float = 1.0,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=False)
        # Support both single w_bits and w_bits_list for backward compatibility
        if w_bits_list is not None:
            self.w_bits_list = w_bits_list
            self.w_bits = w_bits_list[0]  # Default to first bit width
            self.cur_w_bits = w_bits_list[0]
        else:
            self.w_bits_list = [w_bits]
            self.w_bits = w_bits
            self.cur_w_bits = w_bits
        
        self.weight_layerwise = weight_layerwise
        
        # Noise injection parameters
        self.noise_injection = noise_injection
        self.noise_sigma_weights = noise_sigma_weights
        self.noise_sigma_clipvals = noise_sigma_clipvals
        self.initialize_noise = initialize_noise
        self.pre_quantization_noise = pre_quantization_noise
        self.post_quantization_noise = post_quantization_noise
        self.trainable_noise_scale = trainable_noise_scale
        
        # Multi-bit training parameters
        self.multiple_bits_random_assign = multiple_bits_random_assign
        self.multiple_bits_random_assign_prob = multiple_bits_random_assign_prob
        self.multiple_bits_share_clipvals = multiple_bits_share_clipvals
        self.multiple_bits_disable_clipvals = multiple_bits_disable_clipvals
        
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
        
        # Stretch quantization parameters (Still unused)
        self.use_stretch = use_stretch
        self.stretch_alpha = stretch_alpha
        
        # Initialize noise parameters if needed
        if self.initialize_noise:
            self.weight_noise = nn.Parameter(
                torch.Tensor(self.weight.shape[0], self.weight.shape[1])
            )
            self.weight_noise.data.fill_(0)
            self.weight_noise.requires_grad = False
            if self.trainable_noise_scale:
                self.noise_scale = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
                self.noise_scale.data.fill_(1)
        
        # Initialize weight clip values
        if self.w_bits < 16:
            if self.multiple_bits_share_clipvals and len(self.w_bits_list) > 1:
                # Share clip values across all bit widths
                if any(w_bits > 4 for w_bits in self.w_bits_list):
                    # For higher bit widths, use fixed tensor
                    self.weight_clip_val = torch.tensor([-2.0, 2.0])
                else:
                    self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
                    # Initialize with zeros, will be set during training
                    with torch.no_grad():
                        self.weight_clip_val.zero_()
            else:
                # Separate clip values for each bit width
                self.weight_clip_val_list = {}
                for w_bits in self.w_bits_list:
                    if w_bits >= 16:
                        continue
                    if w_bits > 4 or self.multiple_bits_disable_clipvals:
                        # For higher bit widths or when disabled, use fixed tensor
                        self.weight_clip_val_list[str(int(w_bits))] = torch.tensor([-5.0, 5.0])
                    else:
                        param = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
                        with torch.no_grad():
                            param.zero_()
                        self.weight_clip_val_list[str(int(w_bits))] = param
                if not self.multiple_bits_disable_clipvals and len(self.weight_clip_val_list) > 0:
                    self.weight_clip_val_list = nn.ParameterDict(self.weight_clip_val_list)
                # For backward compatibility, if only one bit width, use weight_clip_val
                if len(self.w_bits_list) == 1 and self.w_bits < 16:
                    if self.w_bits > 4 or self.multiple_bits_disable_clipvals:
                        self.weight_clip_val = torch.tensor([-5.0, 5.0])
                    else:
                        self.weight_clip_val = self.weight_clip_val_list[str(int(self.w_bits))]
        else:
            # For 16-bit or higher, no clip value needed
            self.weight_clip_val = None
    
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
        # This logic must match quant_linear.py
        if w_bits >= 16:
            weight_clip_val = None
        elif self.multiple_bits_share_clipvals and len(self.w_bits_list) > 1:
            weight_clip_val = self.weight_clip_val
        elif len(self.w_bits_list) == 1:
            weight_clip_val = self.weight_clip_val
        else:
            # For multi-bit without shared clipvals, get from the list
            if hasattr(self, 'weight_clip_val_list') and str(int(w_bits)) in self.weight_clip_val_list:
                weight_clip_val = self.weight_clip_val_list[str(int(w_bits))]
            else:
                # Fallback: use the first clip value or create a default one
                if hasattr(self, 'weight_clip_val') and self.weight_clip_val is not None:
                    weight_clip_val = self.weight_clip_val
                else:
                    # Create a default clip value on the same device as weights
                    weight_clip_val = torch.tensor([1.0], device=real_weights.device, dtype=real_weights.dtype).expand(real_weights.shape[0], 1)
        
        # Apply noise injection to clip values if enabled
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
        
        # Apply pre-quantization noise if enabled
        if self.noise_injection and self.pre_quantization_noise:
            if self.initialize_noise:
                self.weight_noise.data = torch.randn_like(self.weight_noise.data)
                noise_weights = (
                    self.weight_noise.detach() * self.noise_sigma_weights
                )
            else:
                noise_weights = (
                    torch.randn_like(self.weight) * self.noise_sigma_weights
                )
            if self.trainable_noise_scale:
                real_weights = real_weights + noise_weights * self.noise_scale
            else:
                real_weights = real_weights + noise_weights
        
        # Quantize weights
        # This logic matches quant_linear.py to ensure compatibility
        if w_bits >= 16:
            weight = self.weight
        else:
            if weight_clip_val is None:
                weight_clip_val = torch.tensor(
                    [-1.0, 1.0],
                    device=real_weights.device,
                    dtype=real_weights.dtype,
                )
            elif isinstance(weight_clip_val, torch.Tensor) and (
                weight_clip_val.device != real_weights.device
            ):
                weight_clip_val = weight_clip_val.to(real_weights.device)

            if self.multiple_bits_disable_clipvals:
                quantizer = AsymQuantizer
            else:
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
        if self.noise_injection and self.post_quantization_noise:
            if self.initialize_noise:
                self.weight_noise.data = torch.randn_like(self.weight_noise.data)
                noise_weights = self.weight_noise.detach() * self.noise_sigma_weights
            else:
                noise_weights = torch.randn_like(self.weight) * self.noise_sigma_weights
            if self.trainable_noise_scale:
                weight = weight + noise_weights * self.noise_scale
            else:
                weight = weight + noise_weights
        
        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
