#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('LLM-Research/MobileLLM-125M', cache_dir = './')
