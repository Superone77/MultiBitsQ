class MultiBitsQuantizedColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.
    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        w_bits_list: List[int] = [32],
        a_bits: int = 32,
        seq_len: int = 2048,
        act_layerwise: bool = True,
        weight_layerwise: bool = False,
        symmetric: bool = True,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        parallel_impl: bool = True,
        noise_injection: bool = False,
        noise_sigma_weights: float = 0.001,
        noise_sigma_clipvals: float = 0.001,
        initialize_noise: bool = False,
        pre_quantization_noise: bool = False,
        post_quantization_noise: bool = False,
        trainable_noise_scale: bool = False,
        apply_codebook: bool = False,  # removed codebook for now
        codebook_dim: int = 4,
        codebook_group: int = 32,
        codebook_rot: bool = False,
        codebook_trainable_scales: bool = False,
        use_stretch: bool = False,
        stretch_alpha: float = 1.0,
        multiple_bits_random_assign=False,
        multiple_bits_random_assign_prob=0.5,
        multiple_bits_share_clipvals=False,
        multiple_bits_disable_clipvals=False,
    ) -> None:
        super(MultiBitsQuantizedColumnParallelLinear, self).__init__()
        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.seq_len = seq_len
        self.w_bits_list = w_bits_list
        self.a_bits = a_bits
        self.act_layerwise = act_layerwise
        self.weight_layerwise = weight_layerwise
        self.parallel_impl = parallel_impl
        self.multiple_bits_random_assign = multiple_bits_random_assign
        self.multiple_bits_random_assign_prob = multiple_bits_random_assign_prob
        self.multiple_bits_share_clipvals = multiple_bits_share_clipvals
        self.multiple_bits_disable_clipvals = multiple_bits_disable_clipvals
        # For activation quantization
        if self.a_bits < 16:
            bit_thre = 3
            if self.a_bits < bit_thre and symmetric:
                self.act_layerwise = True
                # pyre-fixme[4]: Attribute must be annotated.
                self.act_clip_val = torch.tensor([1.0])
                # pyre-fixme[4]: Attribute must be annotated.
                self.act_quantizer = ElasticQuantBinarizerSigned
            elif self.a_bits < bit_thre and not symmetric:
                self.act_layerwise = True
                self.act_clip_val = torch.tensor([1.0])
                self.act_quantizer = ElasticQuantBinarizerUnsigned
            elif self.a_bits >= bit_thre and symmetric:
                self.act_clip_val = torch.tensor([-2.0, 2.0])
                self.act_quantizer = SymQuantizer
            elif self.a_bits >= bit_thre and not symmetric:
                self.act_clip_val = torch.tensor([-2.0, 2.0])
                self.act_quantizer = AsymQuantizer
            else:
                raise NotImplementedError
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size() if self.parallel_impl else 1
        # pyre-fixme[4]: Attribute must be annotated.
        self.output_size_per_partition = divide_and_check_no_remainder(
            out_features, world_size
        )
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(
            torch.Tensor(self.output_size_per_partition, self.in_features)
        )
        if bias:
            # pyre-fixme[4]: Attribute must be annotated.
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)
        # Initialize weight.
        # pyre-fixme[4]: Attribute must be annotated.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.output_size_per_partition,
            0,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )
        self.noise_injection = noise_injection
        self.noise_sigma_weights = noise_sigma_weights
        self.noise_sigma_clipvals = noise_sigma_clipvals
        self.initialize_noise = initialize_noise
        self.pre_quantization_noise = pre_quantization_noise
        self.post_quantization_noise = post_quantization_noise
        self.trainable_noise_scale = trainable_noise_scale
        if self.initialize_noise:
            self.weight_noise = Parameter(
                torch.Tensor(self.output_size_per_partition, self.in_features)
            )
            _initialize_affine_weight(
                self.weight_noise,
                self.out_features,
                self.in_features,
                self.output_size_per_partition,
                0,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
            )
            self.weight_noise.data.fill_(0)
            self.weight_noise.requires_grad = False
            if self.trainable_noise_scale:
                self.noise_scale = Parameter(torch.Tensor(self.out_features, 1))
                self.noise_scale.data.fill_(1)
        if self.w_bits_list is not None and self.w_bits_list[0] < 16:
            self.weight_clip_val_list = {}
            if self.multiple_bits_share_clipvals:
                if any(w_bits > 4 for w_bits in self.w_bits_list):
                    # pyre-fixme[4]: Attribute must be annotated.
                    self.weight_clip_val = torch.tensor([-2.0, 2.0])
                else:
                    self.weight_clip_val = Parameter(torch.Tensor(self.out_features, 1))
            else:
                for w_bits in self.w_bits_list:
                    if w_bits > 4 or self.multiple_bits_disable_clipvals:
                        # pyre-fixme[4]: Attribute must be annotated.
                        self.weight_clip_val_list[str(int(w_bits))] = torch.tensor(
                            [-5.0, 5.0]
                        )
                    else:
                        self.weight_clip_val_list[str(int(w_bits))] = Parameter(
                            torch.Tensor(self.out_features, 1)
                        )
                if not self.multiple_bits_disable_clipvals:
                    self.weight_clip_val_list = torch.nn.ParameterDict(
                        self.weight_clip_val_list
                    )
        self.use_stretch = use_stretch
        self.stretch_alpha = stretch_alpha
        self.codebook_rot = codebook_rot
        self.cur_w_bits = self.w_bits_list[0]
    @staticmethod
    def get_very_efficient_rotation(u, q, e) -> torch.Tensor:
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
        e = (
            e
            - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1))
            + 2
            * torch.bmm(torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        )
        return e
    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(
            self.weight.data.transpose(0, 1)
        ).transpose_(0, 1)
    def set_bits(self, w_bits: int):
        self.cur_w_bits = w_bits
    def forward(
        self,
        input_: torch.Tensor,
        # pyre-fixme[9]: scale has type `Optional[Tensor]`; used as `float`.
        scale: Optional[torch.Tensor] = 1.0,
        online_smoothquant: Optional[bool] = False,
    ) -> torch.Tensor:  # type: ignore
        # Quantize weights
        assert len(self.weight.size()) == 2
        if scale is None:
            if online_smoothquant:
                max_input = (
                    torch.max(torch.abs(input_), dim=-2, keepdim=True)[0]
                    .detach()
                    .mean(0)
                )
                max_weight = torch.max(torch.abs(self.weight), dim=-2, keepdim=True)[
                    0
                ].detach()
                scale = torch.sqrt(max_weight / (max_input + 1e-6)) + 1e-6
            else:
                # pyre-fixme[9]: scale has type `Optional[Tensor]`; used as `float`.
                scale = 1.0
        # pyre-fixme[58]: `*` is not supported for operand types `Tensor` and
        #  `Optional[Tensor]`.
        input_ = input_ * scale
        # pyre-fixme[58]: `/` is not supported for operand types `Parameter` and
        #  `Optional[Tensor]`.
        real_weights = self.weight / scale
        # Note that this is sub-optimal to repeat "self.w_bit is not None"
        # However, this is coming from typing Optional[int] to match typing in ModelArgs
        # which is to match typing in ModelArguments.
        # TODO: Remove unnecessary optional typing in ModelArguments
        if (
            self.multiple_bits_random_assign
            and np.random.rand() < self.multiple_bits_random_assign_prob
        ):
            w_bits = np.random.choice(self.w_bits_list)
        else:
            w_bits = self.cur_w_bits
        if self.multiple_bits_share_clipvals:
            weight_clip_val = self.weight_clip_val
        else:
            weight_clip_val = self.weight_clip_val_list[str(int(w_bits))]
        if self.noise_injection:
            noise_clip_vals = (
                torch.randn_like(weight_clip_val) * self.noise_sigma_clipvals
            )
            weight_clip_val = weight_clip_val + noise_clip_vals
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
        if w_bits is None or w_bits >= 16:
            weight = self.weight
        elif w_bits is not None and w_bits > 4:
            weight = SymQuantizer.apply(
                real_weights,
                weight_clip_val,
                w_bits,
                self.weight_layerwise,
            )
        elif self.multiple_bits_disable_clipvals:
            weight = AsymQuantizer.apply(
                real_weights,
                weight_clip_val,
                w_bits,
                self.weight_layerwise,
            )
        elif w_bits is not None and (
            w_bits == 2
            or w_bits == 0
            or w_bits == 1
            or (self.multiple_bits_share_clipvals and w_bits == 4)
            or (self.multiple_bits_share_clipvals and w_bits == 3)
        ):
            if self.use_stretch:
                weight = ElasticQuantN2UQStreched.apply(
                    real_weights,
                    weight_clip_val,
                    w_bits,
                    self.weight_layerwise,
                    self.stretch_alpha,
                ).to(input_.dtype)
            else:
                weight = ElasticQuantN2UQ.apply(
                    real_weights,
                    weight_clip_val,
                    w_bits,
                    self.weight_layerwise,
                ).to(input_.dtype)
        else:
            if self.use_stretch:
                weight = ElasticQuantBinarizerSignedStretched.apply(
                    real_weights,
                    weight_clip_val,
                    w_bits,
                    self.weight_layerwise,
                    self.stretch_alpha,
                ).to(input_.dtype)
            else:
                weight = ElasticQuantBinarizerSigned.apply(
                    real_weights,
                    weight_clip_val,
                    w_bits,
                    self.weight_layerwise,
                ).to(input_.dtype)
        if self.codebook_rot and (
            w_bits is not None and w_bits < 16
        ):  # We can also do a rotation for uniform quantization
            x = real_weights
            quantized = weight
            pre_norm_q = self.get_very_efficient_rotation(
                x / (torch.norm(x, dim=1, keepdim=True) + 1e-6),
                quantized / (torch.norm(quantized, dim=1, keepdim=True) + 1e-6),
                x.unsqueeze(1),
            ).squeeze()
            quantized = (
                pre_norm_q
                * (
                    torch.norm(quantized, dim=1, keepdim=True)
                    / (torch.norm(x, dim=1, keepdim=True) + 1e-6)
                ).detach()
            )
            weight = quantized
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
        # Quantize inputs
        if self.a_bits < 16:
            input_ = self.act_quantizer.apply(
                input_, self.act_clip_val, self.a_bits, self.act_layerwise
            )
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output

