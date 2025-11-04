# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn

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
        w_bits=16,
        w_bits_list=None,  # Support for multiple bits
        weight_layerwise=False,
        noise_injection=False,
        noise_sigma_weights=0.001,
        noise_sigma_clipvals=0.001,
        initialize_noise=False,
        pre_quantization_noise=False,
        post_quantization_noise=False,
        trainable_noise_scale=False,
        use_stretch=False,
        stretch_alpha=1.0,
        multiple_bits_random_assign=False,
        multiple_bits_random_assign_prob=0.5,
        multiple_bits_share_clipvals=False,
        multiple_bits_disable_clipvals=False,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=False)
        # Support both single w_bits and w_bits_list
        if w_bits_list is not None:
            self.w_bits_list = w_bits_list if isinstance(w_bits_list, list) else [w_bits_list]
            self.w_bits = self.w_bits_list[0]  # Default to first bit
        else:
            self.w_bits_list = [w_bits]
            self.w_bits = w_bits
        
        self.weight_layerwise = weight_layerwise
        self.noise_injection = noise_injection
        self.noise_sigma_weights = noise_sigma_weights
        self.noise_sigma_clipvals = noise_sigma_clipvals
        self.initialize_noise = initialize_noise
        self.pre_quantization_noise = pre_quantization_noise
        self.post_quantization_noise = post_quantization_noise
        self.trainable_noise_scale = trainable_noise_scale
        self.use_stretch = use_stretch
        self.stretch_alpha = stretch_alpha
        self.multiple_bits_random_assign = multiple_bits_random_assign
        self.multiple_bits_random_assign_prob = multiple_bits_random_assign_prob
        self.multiple_bits_share_clipvals = multiple_bits_share_clipvals
        self.multiple_bits_disable_clipvals = multiple_bits_disable_clipvals
        
        # Initialize noise parameters if needed
        if self.noise_injection and self.initialize_noise:
            self.weight_noise = nn.Parameter(
                torch.Tensor(self.weight.shape[0], self.weight.shape[1])
            )
            self.weight_noise.data.fill_(0)
            self.weight_noise.requires_grad = False
            if self.trainable_noise_scale:
                self.noise_scale = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
                self.noise_scale.data.fill_(1)
        
        # params for weight quant
        # Only create clip_val if we have bits < 16
        # Note: ParetoQ's quantizers (LsqBinaryTernaryExtension, StretchedElasticQuant) 
        # work best with <= 4 bits. For > 4 bits, we'll use them as fallback.
        if any(w_bits < 16 for w_bits in self.w_bits_list):
            if self.multiple_bits_share_clipvals:
                # Share clip_val across all bits
                # For > 4 bits, we still create a Parameter since quantizers expect it
                self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
                # Initialize based on whether we have > 4 bits
                if any(w_bits > 4 for w_bits in self.w_bits_list):
                    self.weight_clip_val.data.fill_(2.0)  # Use larger initial value for > 4 bits
                else:
                    self.weight_clip_val.data.fill_(1.0)
            else:
                # Separate clip_val for each bit width
                self.weight_clip_val_list = {}
                for w_bits in self.w_bits_list:
                    if w_bits >= 16:
                        continue  # Skip bits >= 16
                    # Always create a Parameter for compatibility with quantizers
                    param = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
                    if w_bits > 4 or self.multiple_bits_disable_clipvals:
                        param.data.fill_(5.0)  # Larger initial value
                    else:
                        param.data.fill_(1.0)  # Standard initial value
                    self.weight_clip_val_list[str(int(w_bits))] = param
                if len(self.weight_clip_val_list) > 0:
                    self.weight_clip_val_list = nn.ParameterDict(self.weight_clip_val_list)
        
        self.cur_w_bits = self.w_bits_list[0]

    def set_bits(self, w_bits: int):
        """Set current quantization bits"""
        self.cur_w_bits = w_bits

    def forward(self, input_):
        # quantize weight
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        # Select which bit width to use
        if (
            self.multiple_bits_random_assign
            and np.random.rand() < self.multiple_bits_random_assign_prob
        ):
            w_bits = np.random.choice(self.w_bits_list)
        else:
            w_bits = self.cur_w_bits

        # Get clip_val for current bit width
        if w_bits < 16:
            if self.multiple_bits_share_clipvals and hasattr(self, 'weight_clip_val'):
                weight_clip_val = self.weight_clip_val
            elif not self.multiple_bits_share_clipvals and hasattr(self, 'weight_clip_val_list'):
                weight_clip_val = self.weight_clip_val_list.get(str(int(w_bits)), None)
            else:
                weight_clip_val = None
        else:
            weight_clip_val = None

        # Apply noise injection if enabled
        if self.noise_injection:
            # Add noise to clip_vals
            if weight_clip_val is not None:
                if isinstance(weight_clip_val, nn.Parameter) or isinstance(weight_clip_val, torch.Tensor):
                    noise_clip_vals = (
                        torch.randn_like(weight_clip_val) * self.noise_sigma_clipvals
                    )
                    weight_clip_val = weight_clip_val + noise_clip_vals
            
            # Pre-quantization noise
            if self.pre_quantization_noise:
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
        # Note: weight_clip_val should always be a Parameter after initialization
        if w_bits >= 16:
            weight = self.weight
        elif w_bits == 2 or w_bits == 0:
            # Use StretchedElasticQuant for 2-bit or 0-bit
            if weight_clip_val is None:
                # Fallback: create a default if somehow None
                if not hasattr(self, '_default_clip_val'):
                    self._default_clip_val = nn.Parameter(torch.tensor([1.0], device=real_weights.device))
                weight_clip_val = self._default_clip_val
            weight = StretchedElasticQuant.apply(
                real_weights,
                weight_clip_val,
                w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
        elif w_bits <= 4:
            # Use LsqBinaryTernaryExtension for <= 4-bit (recommended for ParetoQ)
            if weight_clip_val is None:
                if not hasattr(self, '_default_clip_val'):
                    self._default_clip_val = nn.Parameter(torch.tensor([1.0], device=real_weights.device))
                weight_clip_val = self._default_clip_val
            weight = LsqBinaryTernaryExtension.apply(
                real_weights,
                weight_clip_val,
                w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
        else:
            # For > 4 bits, ParetoQ doesn't have a specific quantizer
            # Use LsqBinaryTernaryExtension as fallback (may not be optimal)
            if weight_clip_val is None:
                if not hasattr(self, '_default_clip_val'):
                    self._default_clip_val = nn.Parameter(torch.tensor([1.0], device=real_weights.device))
                weight_clip_val = self._default_clip_val
            weight = LsqBinaryTernaryExtension.apply(
                real_weights,
                weight_clip_val,
                w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)

        # Post-quantization noise
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
