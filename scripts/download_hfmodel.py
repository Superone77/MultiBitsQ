# HuggingFace模型下载
import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从HuggingFace下载模型')
    parser.add_argument('--output_dir', type=str, default='./', 
                        help='保存目录（默认：当前目录）')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['facebook/MobileLLM-ParetoQ-125M-BF16'],
                        help='要下载的模型列表（默认：facebook/MobileLLM-ParetoQ-125M-BF16）')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace API token（可选，也可通过环境变量HF_TOKEN设置）')
    
    args = parser.parse_args()
    
    # 获取token：优先使用参数，其次环境变量，最后使用文件中的key
    token = args.token
    if not token:
        token = os.environ.get('HF_TOKEN')
    if not token:
        # 尝试从文件读取（如果存在）
        key_file = Path(__file__).parent / 'download_hfmodel.py'
        # 这里可以读取文件中的key，但为了安全，建议使用环境变量
    
    # 清理和验证token：去除首尾空白，只传递非空字符串
    if token:
        token = token.strip()
        if not token:
            token = None
    
    # 创建输出目录（如果不存在）
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载每个模型
    for model_name in args.models:
        print(f"正在下载模型：{model_name}")
        try:
            # 只传递token如果它是有效的非空字符串
            download_kwargs = {
                'repo_id': model_name,
                'cache_dir': str(output_dir),
            }
            if token:
                download_kwargs['token'] = token
            
            model_dir = snapshot_download(**download_kwargs)
            print(f"模型已保存至：{model_dir}")
        except Exception as e:
            print(f"下载模型 {model_name} 时出错：{e}")
            raise
    
    print("所有模型下载完成！")
