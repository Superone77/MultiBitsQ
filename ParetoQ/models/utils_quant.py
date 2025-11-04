# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

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
        # Debug: Check for NaN in inputs
        if torch.isnan(input).any():
            logger.error(f"[StretchedElasticQuant.forward] NaN detected in input! num_bits={num_bits}, shape={input.shape}")
            logger.error(f"  Input stats: min={input.min().item():.6f}, max={input.max().item():.6f}, mean={input.mean().item():.6f}")
        if torch.isnan(alpha).any():
            logger.error(f"[StretchedElasticQuant.forward] NaN detected in alpha! num_bits={num_bits}, shape={alpha.shape}")
            logger.error(f"  Alpha stats: min={alpha.min().item():.6f}, max={alpha.max().item():.6f}, mean={alpha.mean().item():.6f}")
        
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
        alpha_clamped = torch.where(alpha > eps, alpha, eps)
        
        # Debug: Check if clamping changed alpha significantly
        if num_bits == 2 and (alpha_clamped != alpha).any():
            clamped_count = (alpha_clamped != alpha).sum().item()
            logger.warning(f"[StretchedElasticQuant.forward] w_bits=2: Clamped {clamped_count} alpha values from < {eps.item()} to {eps.item()}")
            logger.warning(f"  Alpha before clamp: min={alpha.min().item():.6f}, max={alpha.max().item():.6f}")
            logger.warning(f"  Alpha after clamp: min={alpha_clamped.min().item():.6f}, max={alpha_clamped.max().item():.6f}")
        
        alpha = alpha_clamped

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
        
        # Debug: Check for NaN in output
        if torch.isnan(w_q).any():
            logger.error(f"[StretchedElasticQuant.forward] NaN detected in output w_q! num_bits={num_bits}")
            logger.error(f"  q_w stats: min={q_w.min().item():.6f}, max={q_w.max().item():.6f}, mean={q_w.mean().item():.6f}")
            logger.error(f"  alpha stats: min={alpha.min().item():.6f}, max={alpha.max().item():.6f}, mean={alpha.mean().item():.6f}")
            logger.error(f"  input/alpha max: {(input / alpha).max().item():.6f}")
            logger.error(f"  n_levels={n_levels}, shift={shift}, clip_val={clip_val}")
        
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        # Debug: Check for NaN in grad_output
        if torch.isnan(grad_output).any():
            logger.error(f"[StretchedElasticQuant.backward] NaN detected in grad_output! num_bits={ctx.num_bits}")
            logger.error(f"  grad_output stats: min={grad_output.min().item():.6f}, max={grad_output.max().item():.6f}")
            nan_count = torch.isnan(grad_output).sum().item()
            logger.error(f"  NaN count: {nan_count} / {grad_output.numel()}")

        input_, alpha = ctx.saved_tensors
        
        # Debug: Check saved tensors
        if torch.isnan(input_).any():
            logger.error(f"[StretchedElasticQuant.backward] NaN detected in saved input_! num_bits={ctx.num_bits}")
        if torch.isnan(alpha).any():
            logger.error(f"[StretchedElasticQuant.backward] NaN detected in saved alpha! num_bits={ctx.num_bits}")
            logger.error(f"  Alpha stats: min={alpha.min().item():.6f}, max={alpha.max().item():.6f}, mean={alpha.mean().item():.6f}")
        
        # Add numerical stability protection - handle NaN/Inf first
        if torch.isnan(alpha).any() or torch.isinf(alpha).any():
            logger.error(f"[StretchedElasticQuant.backward] NaN/Inf in saved alpha! Replacing with safe values.")
            alpha = torch.where(torch.isnan(alpha) | torch.isinf(alpha),
                               torch.ones_like(alpha) * 0.1,
                               alpha)
        
        eps = torch.tensor(0.00001, device=alpha.device, dtype=alpha.dtype)
        alpha_safe = torch.clamp(alpha, min=eps.item(), max=10.0)  # Add max clamp too
        
        if ctx.num_bits == 2 and (alpha_safe != alpha).any():
            clamped_count = (alpha_safe != alpha).sum().item()
            logger.warning(f"[StretchedElasticQuant.backward] w_bits=2: Clamped {clamped_count} alpha values in backward")
            logger.warning(f"  Alpha before clamp: min={alpha.min().item():.6f}, max={alpha.max().item():.6f}")
            logger.warning(f"  Alpha after clamp: min={alpha_safe.min().item():.6f}, max={alpha_safe.max().item():.6f}")
        
        alpha = alpha_safe
        
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        
        # Debug: Check q_w for extreme values or NaN
        if torch.isnan(q_w).any() or torch.isinf(q_w).any():
            logger.error(f"[StretchedElasticQuant.backward] NaN/Inf detected in q_w = input_ / alpha! num_bits={ctx.num_bits}")
            logger.error(f"  input_ stats: min={input_.min().item():.6f}, max={input_.max().item():.6f}")
            logger.error(f"  alpha stats: min={alpha.min().item():.6f}, max={alpha.max().item():.6f}")
            logger.error(f"  q_w stats: min={q_w.min().item():.6f}, max={q_w.max().item():.6f}")
            logger.error(f"  input_/alpha max: {(input_ / alpha).max().item():.6f}")
            if torch.isnan(q_w).any():
                nan_count = torch.isnan(q_w).sum().item()
                logger.error(f"  NaN count in q_w: {nan_count} / {q_w.numel()}")
            if torch.isinf(q_w).any():
                inf_count = torch.isinf(q_w).sum().item()
                logger.error(f"  Inf count in q_w: {inf_count} / {q_w.numel()}")
        
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
        
        # Debug: Check grad_alpha for NaN
        if torch.isnan(grad_alpha).any():
            logger.error(f"[StretchedElasticQuant.backward] NaN detected in grad_alpha! num_bits={ctx.num_bits}, layerwise={layerwise}")
            logger.error(f"  grad_alpha stats: min={grad_alpha.min().item():.6f}, max={grad_alpha.max().item():.6f}")
            logger.error(f"  indicate_small sum: {indicate_small.sum().item()}, indicate_big sum: {indicate_big.sum().item()}")
            logger.error(f"  indicate_middle sum: {indicate_middle.sum().item()}")

        grad_input = indicate_middle * grad_output
        
        # Debug: Check grad_input for NaN
        if torch.isnan(grad_input).any():
            logger.error(f"[StretchedElasticQuant.backward] NaN detected in grad_input! num_bits={ctx.num_bits}")
        
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
        
        # Debug: Check weight_clip_val for w_bits=2
        if w_bits == 2 and weight_clip_val is not None:
            if torch.isnan(weight_clip_val).any() or torch.isinf(weight_clip_val).any():
                logger.error(f"[QuantizeLinear.forward] NaN/Inf detected in weight_clip_val for w_bits=2!")
                logger.error(f"  weight_clip_val stats: min={weight_clip_val.min().item():.6f}, max={weight_clip_val.max().item():.6f}")
                # Emergency fix: replace NaN/Inf with safe values
                nan_inf_mask = torch.isnan(weight_clip_val) | torch.isinf(weight_clip_val)
                nan_inf_count = nan_inf_mask.sum().item()
                logger.error(f"  NaN/Inf count: {nan_inf_count}")
                weight_clip_val = torch.where(nan_inf_mask,
                                             torch.ones_like(weight_clip_val) * 0.1,
                                             weight_clip_val)
                weight_clip_val = torch.clamp(weight_clip_val, min=0.01, max=10.0)
                logger.warning(f"[QuantizeLinear.forward] Fixed weight_clip_val by replacing NaN/Inf with safe values")
            if (weight_clip_val < 0.01).any():
                small_count = (weight_clip_val < 0.01).sum().item()
                logger.warning(f"[QuantizeLinear.forward] w_bits=2: Found {small_count} weight_clip_val values < 0.01")
                logger.warning(f"  min weight_clip_val: {weight_clip_val.min().item():.6f}")
                # Fix small values
                weight_clip_val = torch.clamp(weight_clip_val, min=0.01)

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
                logger.warning(f"[QuantizeLinear.forward] w_bits={w_bits}: Using default clip_val")
            
            # Debug: Check real_weights before quantization
            if w_bits == 2 and torch.isnan(real_weights).any():
                logger.error(f"[QuantizeLinear.forward] NaN detected in real_weights before StretchedElasticQuant! w_bits=2")
                logger.error(f"  real_weights stats: min={real_weights.min().item():.6f}, max={real_weights.max().item():.6f}")
            
            weight = StretchedElasticQuant.apply(
                real_weights,
                weight_clip_val,
                w_bits,
                self.weight_layerwise,
            ).to(input_.dtype)
            
            # Debug: Check weight after quantization
            if w_bits == 2 and torch.isnan(weight).any():
                logger.error(f"[QuantizeLinear.forward] NaN detected in weight after StretchedElasticQuant! w_bits=2")
                logger.error(f"  weight stats: min={weight.min().item():.6f}, max={weight.max().item():.6f}")
                logger.error(f"  weight_clip_val used: min={weight_clip_val.min().item():.6f}, max={weight_clip_val.max().item():.6f}")
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
        
        # Debug: Check output for NaN
        if any(w_bits == 2 for w_bits in self.w_bits_list) and torch.isnan(out).any():
            logger.error(f"[QuantizeLinear.forward] NaN detected in output! w_bits={w_bits}")
            logger.error(f"  input_ stats: min={input_.min().item():.6f}, max={input_.max().item():.6f}")
            logger.error(f"  weight stats: min={weight.min().item():.6f}, max={weight.max().item():.6f}")
            logger.error(f"  output stats: min={out.min().item():.6f}, max={out.max().item():.6f}")
            nan_count = torch.isnan(out).sum().item()
            logger.error(f"  NaN count in output: {nan_count} / {out.numel()}")
        
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
