# 关于 weight_clip_val 警告的说明

## 警告内容

当使用 `2_run_eval.sh` 加载多比特训练后的 MobileLLM 模型时，可能会看到以下警告：

```
Some weights of the model checkpoint at ... were not used when initializing LlamaForCausalLM: 
['model.layers.0.mlp.down_proj.weight_clip_val', ...]

Some weights of LlamaForCausalLM were not initialized from the model checkpoint ... and are newly initialized:
['model.layers.0.mlp.down_proj.weight_clip_val_list.2', 'model.layers.0.mlp.down_proj.weight_clip_val_list.3', ...]
```

## 原因分析

这个警告是**正常的**，原因如下：

1. **保存时的格式**：
   - 如果模型在训练时使用了 `multiple_bits_share_clipvals=True`，所有比特宽度共享同一个 `weight_clip_val`
   - 或者模型是单比特训练的，只有 `weight_clip_val`（单数形式）

2. **加载时的格式**：
   - 如果加载时配置了 `w_bits_list=[2,3,4]` 且 `multiple_bits_share_clipvals=False`（默认值）
   - 模型会期望 `weight_clip_val_list.2`、`weight_clip_val_list.3`、`weight_clip_val_list.4`（列表形式）

3. **结构不匹配**：
   - HuggingFace 的 `from_pretrained` 无法自动将 `weight_clip_val` 映射到 `weight_clip_val_list`
   - 因此会忽略 `weight_clip_val`，并重新初始化 `weight_clip_val_list`

## 是否影响使用？

**通常情况下，这个警告不会影响模型的正常使用**，因为：

1. 代码中有自动迁移逻辑（如果设置了 `contain_weight_clip_val=True`），会尝试将 `weight_clip_val` 迁移到 `weight_clip_val_list`
2. 即使迁移失败，`weight_clip_val_list` 会被重新初始化，模型仍然可以正常运行（只是可能丢失了一些训练好的 clip value 信息）

## 解决方案

### 方案 1：忽略警告（推荐）

如果模型能正常运行且评估结果正常，可以**直接忽略这个警告**。这是最简单的方法。

### 方案 2：确保配置一致

如果原来的模型是用 `multiple_bits_share_clipvals=True` 保存的，在加载时也设置相同的参数：

```bash
--multiple_bits_share_clipvals True
```

这样模型结构就会匹配，不会出现警告。

### 方案 3：使用迁移功能

代码已经添加了自动迁移功能。如果设置了 `contain_weight_clip_val=True`，代码会尝试从 checkpoint 中加载 `weight_clip_val` 并迁移到 `weight_clip_val_list`。

迁移过程：
1. 加载原始 checkpoint 的 state_dict
2. 查找所有 `weight_clip_val` 参数
3. 将它们复制到对应的 `weight_clip_val_list.2`、`weight_clip_val_list.3`、`weight_clip_val_list.4` 等

如果迁移成功，你会看到日志：
```
Migrated X weight_clip_val parameters to weight_clip_val_list
```

## 总结

- **这个警告是正常的**，不会影响模型的基本功能
- 如果模型能正常运行，**可以安全地忽略这个警告**
- 如果想要消除警告，确保加载时的 `multiple_bits_share_clipvals` 参数与保存时一致
- 代码已经包含了自动迁移逻辑，会在可能的情况下自动处理权重迁移

