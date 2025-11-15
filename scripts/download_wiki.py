import json
from datasets import load_dataset
from tqdm import tqdm

def save_wikitext_to_jsonl(dataset, output_path, num_samples=10000):
    """从Wikitext数据集中提取指定数量样本并保存为JSONL"""
    with open(output_path, 'w', encoding='utf-8') as f_out:
        count = 0
        # 遍历数据集，处理每个样本
        for sample in tqdm(dataset, desc="提取样本"):
            if count >= num_samples:
                break  # 达到目标数量则停止
            
            # Wikitext的文本字段为"text"，但可能包含空行或短文本，需要过滤
            text = sample.get('text', '').strip()
            # 过滤过短或无效的文本（根据需要调整长度阈值）
            if text and len(text) > 50:  # 忽略过短文本（如标题、空行）
                # 按要求格式写入JSONL
                json.dump({"text": text}, f_out, ensure_ascii=False)
                f_out.write('\n')
                count += 1

if __name__ == "__main__":
    # 配置参数
    DATASET_NAME = "wikitext"  # Wikitext数据集名称
    SUBSET_NAME = "wikitext-103-raw-v1"  # 常用子集（可替换为其他版本）
    OUTPUT_JSONL = "wikitext_10k_samples.jsonl"
    TARGET_COUNT = 10000  # 需要提取的样本数

    # 加载Wikitext数据集（流式加载，节省内存）
    print(f"1/2 开始加载数据集：{DATASET_NAME}（{SUBSET_NAME}）")
    wikitext_dataset = load_dataset(
        DATASET_NAME,
        name=SUBSET_NAME,
        split="train",
        streaming=True  # 流式加载，无需下载完整数据集
    )

    # 提取并保存样本
    print(f"2/2 开始提取{TARGET_COUNT}个样本...")
    save_wikitext_to_jsonl(wikitext_dataset, OUTPUT_JSONL, TARGET_COUNT)

    print(f"完成！样本已保存至：{OUTPUT_JSONL}")