# MobileLLM 模型支持说明

本文档说明如何在 ParetoQ 中使用 MobileLLM 模型进行训练和推理。

## 修改内容

### 1. 添加了 MobileLLM 相关参数支持

在 `utils/process_args.py` 中添加了两个新参数：
- `share_embedding`: 是否共享输入和输出 embedding（MobileLLM 特性）
- `layer_sharing`: 是否启用层共享（重复使用 decoder layer，MobileLLM 特性）

### 2. 更新了训练脚本

在 `train.py` 中，从命令行参数读取这两个参数并设置到 config 中，确保模型正确加载 MobileLLM 的特性。

## 使用方法

### 从 HuggingFace 加载 MobileLLM 模型

MobileLLM 模型已经在 HuggingFace Hub 上发布，你可以直接使用模型名称加载：

```bash
# 示例：MobileLLM-125M
--input_model_filename "facebook/MobileLLM-125M"

# 其他可用的模型：
# - facebook/MobileLLM-350M
# - facebook/MobileLLM-600M
# - facebook/MobileLLM-1B
# - facebook/MobileLLM-1.5B
```

### 设置 MobileLLM 参数

根据你加载的 MobileLLM 模型，需要设置相应的参数：

```bash
# 对于 MobileLLM 模型，通常需要：
--share_embedding True
--layer_sharing False  # 大部分 MobileLLM 模型不使用 layer_sharing

# 如果模型配置中已经设置了这些参数，你也可以不设置，代码会使用配置中的值
# 但为了明确，建议显式设置
```

### 完整训练示例

参考 `run_mobilellm_training.sh` 脚本，这是一个完整的训练示例：

```bash
torchrun --nnodes=1 --nproc_per_node=1 train.py \
--local_dir "/tmp/llama/" \
--input_model_filename "facebook/MobileLLM-125M" \
--output_model_filename "MobileLLM-125M-multi-bit-trained" \
--share_embedding True \
--layer_sharing False \
--w_bits_list "1,2,4" \
# ... 其他训练参数
```

### 推理/评估示例

对于推理和评估，使用相同的参数：

```bash
torchrun --nnodes=1 --nproc_per_node=1 train.py \
--local_dir "/tmp/llama/" \
--input_model_filename "facebook/MobileLLM-125M" \
--share_embedding True \
--layer_sharing False \
--do_train False \
--do_eval True \
--contain_weight_clip_val True \
# ... 其他评估参数
```

## 注意事项

1. **参数检查**：如果 HuggingFace 上的模型 config.json 中已经包含了 `share_embedding` 和 `layer_sharing` 参数，命令行参数会覆盖它们。建议查看模型配置并显式设置这些参数。

2. **模型兼容性**：代码已经支持 MobileLLM 的所有特性：
   - ✅ `share_embedding`: 在 `LlamaForCausalLM` 中实现
   - ✅ `layer_sharing`: 在 `LlamaModel.forward` 中实现

3. **权重加载**：当从 HuggingFace 加载预训练模型时，`from_pretrained` 会自动加载权重。如果模型使用了 `share_embedding`，确保 config 中设置了正确的参数，否则可能导致权重加载错误。

## 验证

加载模型后，可以通过以下方式验证 MobileLLM 特性是否生效：

```python
# 检查 config
print(model.config.share_embedding)  # 应该为 True
print(model.config.layer_sharing)    # 应该为 False（或 True，取决于模型）

# 检查模型结构
if model.config.share_embedding:
    # lm_head 应该不存在，使用 embed_tokens 作为输出
    assert not hasattr(model, 'lm_head') or model.lm_head is None
```

## 参考

- MobileLLM 论文: https://arxiv.org/abs/2402.14905
- HuggingFace 模型库: https://huggingface.co/collections/facebook/mobilellm-6722be18cb86c20ebe113e95

