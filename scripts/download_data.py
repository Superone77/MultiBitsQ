import json
import os
import argparse
from pathlib import Path
from modelscope import MsDataset
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
    parser = argparse.ArgumentParser(description='从ModelScope下载数据集并保存为JSONL')
    parser.add_argument('--output_dir', type=str, default='./', 
                        help='保存目录（默认：当前目录）')
    parser.add_argument('--dataset_name', type=str, default='HuggingFaceFW/fineweb-edu',
                        help='数据集名称（默认：HuggingFaceFW/fineweb-edu）')
    parser.add_argument('--subset_name', type=str, default='CC-MAIN-2024-10',
                        help='子集名称（默认：CC-MAIN-2024-10）')
    parser.add_argument('--output_filename', type=str, default='finewebedu_4000k_samples.jsonl',
                        help='输出文件名（默认：finewebedu_6000k_samples.jsonl）')
    parser.add_argument('--target_count', type=int, default=6000000,
                        help='目标样本数量（默认：6000000）')
    
    args = parser.parse_args()
    
    # 创建输出目录（如果不存在）
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建完整输出路径
    OUTPUT_JSONL = str(output_dir / args.output_filename)
    
    # 流式加载数据集（无需下载完整数据）
    print(f"1/2 开始流式加载数据集：{args.dataset_name}（{args.subset_name}）")
    fw_dataset = MsDataset.load(
        args.dataset_name,
        subset_name=args.subset_name,
        split="train",
        streaming=True  # 关键参数：流式加载，避免内存占用过大
    )

    # 提取并保存样本
    print(f"2/2 开始提取{args.target_count}个样本...")
    save_samples_to_jsonl(fw_dataset, OUTPUT_JSONL, args.target_count)

    print(f"完成！样本已保存至：{OUTPUT_JSONL}")