import json
import argparse
from pathlib import Path
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
    parser = argparse.ArgumentParser(description='从HuggingFace下载Wikitext数据集并保存为JSONL')
    parser.add_argument('--output_dir', type=str, default='./', 
                        help='保存目录（默认：当前目录）')
    parser.add_argument('--dataset_name', type=str, default='wikitext',
                        help='数据集名称（默认：wikitext）')
    parser.add_argument('--subset_name', type=str, default='wikitext-103-raw-v1',
                        help='子集名称（默认：wikitext-103-raw-v1）')
    parser.add_argument('--output_filename', type=str, default='wikitext_10k_samples.jsonl',
                        help='输出文件名（默认：wikitext_10k_samples.jsonl）')
    parser.add_argument('--target_count', type=int, default=10000,
                        help='目标样本数量（默认：10000）')
    
    args = parser.parse_args()
    
    # 创建输出目录（如果不存在）
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建完整输出路径
    OUTPUT_JSONL = str(output_dir / args.output_filename)

    # 加载Wikitext数据集（流式加载，节省内存）
    print(f"1/2 开始加载数据集：{args.dataset_name}（{args.subset_name}）")
    wikitext_dataset = load_dataset(
        args.dataset_name,
        name=args.subset_name,
        split="train",
        streaming=True  # 流式加载，无需下载完整数据集
    )

    # 提取并保存样本
    print(f"2/2 开始提取{args.target_count}个样本...")
    save_wikitext_to_jsonl(wikitext_dataset, OUTPUT_JSONL, args.target_count)

    print(f"完成！样本已保存至：{OUTPUT_JSONL}")