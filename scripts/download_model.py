#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('LLM-Research/MobileLLM-125M', cache_dir = './')
model_dir = snapshot_download('LLM-Research/Llama-3.2-1B', cache_dir = './')
