import json
from datasets import load_dataset
from tqdm import tqdm

def save_samples_to_jsonl(dataset, output_path, num_samples=20000):
    """从流式数据集中提取指定数量样本并保存为JSONL"""
    with open(output_path, 'w', encoding='utf-8') as f_out:
        # 遍历流式数据集，提取text字段
        for i, sample in tqdm(enumerate(dataset), total=num_samples, desc="提取样本"):
            if i >= num_samples:
                break  # 达到目标数量则停止
            # 过滤空文本
            text = sample.get('text', '').strip()
            if text:
                # 按要求格式写入JSONL
                json.dump({"text": text}, f_out, ensure_ascii=False)
                f_out.write('\n')

if __name__ == "__main__":
    # 配置参数
    DATASET_NAME = "HuggingFaceFW/fineweb-edu"
    SUBSET_NAME = "CC-MAIN-2024-10"  # 可替换为其他子集（如"sample-10BT"）
    OUTPUT_JSONL = "finewebedu_50k_samples.jsonl"
    TARGET_COUNT = 50000

    # 流式加载数据集（无需下载完整数据）
    print(f"1/2 开始流式加载数据集：{DATASET_NAME}（{SUBSET_NAME}）")
    fw_dataset = load_dataset(
        DATASET_NAME,
        name=SUBSET_NAME,
        split="train",
        streaming=True  # 关键参数：流式加载，避免内存占用过大
    )

    # 提取并保存样本
    print(f"2/2 开始提取{TARGET_COUNT}个样本...")
    save_samples_to_jsonl(fw_dataset, OUTPUT_JSONL, TARGET_COUNT)

    print(f"完成！样本已保存至：{OUTPUT_JSONL}")