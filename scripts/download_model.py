#SDK模型下载
import argparse
from pathlib import Path
from modelscope import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从ModelScope下载模型')
    parser.add_argument('--output_dir', type=str, default='./', 
                        help='保存目录（默认：当前目录）')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['LLM-Research/MobileLLM-125M', 'LLM-Research/Llama-3.2-1B'],
                        help='要下载的模型列表（默认：MobileLLM-125M和Llama-3.2-1B）')
    
    args = parser.parse_args()
    
    # 创建输出目录（如果不存在）
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载每个模型
    for model_name in args.models:
        print(f"正在下载模型：{model_name}")
        model_dir = snapshot_download(model_name, cache_dir=str(output_dir))
        print(f"模型已保存至：{model_dir}")
    
    print("所有模型下载完成！")
