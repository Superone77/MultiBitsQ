import json
import os
import argparse
from pathlib import Path
from modelscope import MsDataset
from tqdm import tqdm

def save_samples_to_jsonl(dataset, output_path, num_samples=20000):
    """从流式数据集中提取指定数量样本并保存为JSONL
    
    Args:
        dataset: 流式数据集
        output_path: 输出文件路径
        num_samples: 目标样本数量，如果为None或-1则保存所有数据
    """
    with open(output_path, 'w', encoding='utf-8') as f_out:
        count = 0
        # 遍历流式数据集，提取text字段
        if num_samples is None or num_samples == -1:
            # 保存所有数据，不设置total（tqdm会显示进度但不显示总数）
            for sample in tqdm(dataset, desc="提取样本"):
                # 过滤空文本
                text = sample.get('text', '').strip()
                if text:
                    # 按要求格式写入JSONL
                    json.dump({"text": text}, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    count += 1
        else:
            # 保存指定数量的样本
            for i, sample in tqdm(enumerate(dataset), total=num_samples, desc="提取样本"):
                if i >= num_samples:
                    break  # 达到目标数量则停止
                # 过滤空文本
                text = sample.get('text', '').strip()
                if text:
                    # 按要求格式写入JSONL
                    json.dump({"text": text}, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    count += 1
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从ModelScope下载数据集并保存为JSONL')
    parser.add_argument('--output_dir', type=str, default='./', 
                        help='保存目录（默认：当前目录）')
    parser.add_argument('--dataset_name', type=str, default='HuggingFaceFW/fineweb-edu',
                        help='数据集名称（默认：HuggingFaceFW/fineweb-edu）')
    parser.add_argument('--subset_name', type=str, default='sample-100BT',
                        help='子集名称（默认：sample-100BT）')
    parser.add_argument('--output_filename', type=str, default='finewebedu_train_samples.jsonl',
                        help='输出文件名（默认：finewebedu_train_samples.jsonl）')
    parser.add_argument('--target_count', type=int, default=100000000,
                        help='目标样本数量（默认：100000000，使用-1表示保存所有数据）')
    
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
    if args.target_count == -1:
        print(f"2/2 开始提取所有样本...")
        num_samples = None
    else:
        print(f"2/2 开始提取{args.target_count}个样本...")
        num_samples = args.target_count
    
    saved_count = save_samples_to_jsonl(fw_dataset, OUTPUT_JSONL, num_samples)

    print(f"完成！共保存 {saved_count} 个样本至：{OUTPUT_JSONL}")