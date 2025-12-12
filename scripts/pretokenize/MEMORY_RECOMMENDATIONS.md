# Pretokenize 内存需求分析与建议

## ⚠️ 当前实现的内存问题

**重要发现**：当前 `pretokenize.py` 的实现会在内存中累积**所有**处理过的 chunks（`all_input_ids` 和 `all_labels`），直到最后一次性写入磁盘。这意味着：

- **不是真正的流式处理**：内存使用会随着处理的样本数线性增长
- **对于大数据集**：内存需求接近理论最大值

## 内存需求计算

### 理论最大值（一次性加载所有数据）

对于 **100B tokens**：
- Input IDs: 100B × 4 bytes = **400 GB**
- Labels: 100B × 4 bytes = **400 GB**
- Python对象开销: ~80 GB
- **总计: ~880 GB - 1 TB**

### 实际内存使用（当前实现）

由于代码会累积所有chunks，实际内存使用为：

```
实际内存 ≈ (已处理的tokens数 / 总tokens数) × 理论最大值
```

**关键点**：
- 处理过程中内存持续增长
- 写入磁盘时达到峰值（接近理论最大值）
- 写入完成后内存释放

### 不同数据集规模的内存需求

| 数据集规模 | Tokens | 理论最大值 | 推荐内存 | 说明 |
|-----------|--------|-----------|---------|------|
| 小数据集 | 1B | ~9 GB | **32 GB** | 安全余量 |
| 中等数据集 | 10B | ~88 GB | **128 GB** | 2倍安全余量 |
| 大数据集 | 100B | ~880 GB | **640-960 GB** | 1.2-1.5倍安全余量 |
| 超大数据集 | 1T | ~8.8 TB | **6-8 TB** | 需要分批处理 |

## 推荐配置

### 方案1：保守配置（推荐用于生产环境）

```ini
# pretokenize.sub
request_memory = 640000  # 640 GB
request_cpus = 8
request_gpus = 0
```

**适用场景**：
- 100B tokens以下的数据集
- 需要稳定运行
- 有足够资源

### 方案2：中等配置（平衡性能和资源）

```ini
# pretokenize.sub
request_memory = 320000  # 320 GB
request_cpus = 8
request_gpus = 0
```

**适用场景**：
- 10-50B tokens的数据集
- 资源有限
- 可以接受偶尔的内存不足风险

### 方案3：最小配置（仅用于测试）

```ini
# pretokenize.sub
request_memory = 128000  # 128 GB
request_cpus = 8
request_gpus = 0
```

**适用场景**：
- 1-10B tokens的小数据集
- 测试和开发
- 快速迭代

## 根据 batch_size_samples 调整

当前默认 `batch_size_samples=1000`，这个参数主要影响：
- **处理速度**：更大的batch = 更快的处理
- **峰值内存**：影响较小（因为所有chunks都会累积）

### 建议的 batch_size_samples 设置

| 可用内存 | batch_size_samples | 说明 |
|---------|-------------------|------|
| < 128 GB | 500 | 保守设置 |
| 128-320 GB | 1000 | 默认值，平衡 |
| 320-640 GB | 2000-5000 | 可以加快处理速度 |
| > 640 GB | 5000+ | 最大化处理速度 |

## 优化建议

### 1. 立即优化：根据数据集大小动态调整内存

在 `pretokenize.sh` 中添加内存估算：

```bash
# 估算数据集大小（示例）
DATASET_SIZE=$(wc -l < "$TRAIN_DATA_PATH")
ESTIMATED_TOKENS=$((DATASET_SIZE * 2000))  # 假设每行平均2000 tokens

# 根据估算调整内存请求
if [ $ESTIMATED_TOKENS -gt 50000000000 ]; then  # > 50B
    REQUEST_MEMORY=640000  # 640 GB
elif [ $ESTIMATED_TOKENS -gt 10000000000 ]; then  # > 10B
    REQUEST_MEMORY=320000  # 320 GB
else
    REQUEST_MEMORY=128000  # 128 GB
fi
```

### 2. 长期优化：真正的流式写入

修改 `pretokenize.py` 实现真正的流式处理：
- 每处理一定数量的chunks就写入磁盘
- 使用多个缓存文件，最后合并
- 或者使用数据库/文件系统直接写入

### 3. 分批处理超大数据集

对于 > 100B tokens的数据集，建议：
- 将数据文件分割成多个小文件
- 分别处理每个文件
- 最后合并缓存（如果需要）

## 监控内存使用

在运行过程中监控内存：

```bash
# 在pretokenize.sh中添加
watch -n 5 'free -h && ps aux | grep pretokenize | head -1'
```

或者使用HTCondor的监控工具。

## 总结

**当前配置（320GB）的建议**：
- ✅ **适合**：10-30B tokens的数据集
- ⚠️ **可能不足**：50B+ tokens的数据集
- ❌ **不适合**：100B+ tokens的数据集

**推荐调整**：
- 对于100B tokens：使用 **640GB-960GB**
- 对于10B tokens：**320GB** 足够
- 对于1B tokens：**128GB** 足够

**最佳实践**：
1. 先用 `--estimate_memory` 估算
2. 根据估算结果调整 `request_memory`
3. 监控实际运行时的内存使用
4. 根据实际情况调整

