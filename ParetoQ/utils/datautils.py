import logging
from typing import Optional, Dict, Any, Iterator
import torch
import datasets
import datasets.distributed

from datasets import load_dataset, IterableDataset as HFIterableDataset

log = logging.getLogger(__name__)

class StreamingJsonlDataset(torch.utils.data.IterableDataset):
    """
    基于 Hugging Face datasets 库重写的流式 Dataset。
    Stream jsonl using HF datasets, tokenize efficiently in batches, and pack into block_size chunks.
    保留了实时 Token 统计功能。
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        block_size: int,
        rank: int = 0,
        world_size: int = 1,
        max_samples: Optional[int] = None,
        print_interval: int = 100000,
        batch_size: int = 1000, # 新增：每次分词处理的文本批大小
    ):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.rank = rank
        self.world_size = max(world_size, 1)
        self.max_samples = max_samples if max_samples and max_samples > 0 else None
        self.print_interval = print_interval
        self.batch_size = batch_size

        # Token统计变量
        self.total_processed_tokens = 0
        self.total_yielded_tokens = 0
        self._last_print_tokens = 0 

        # 1. 加载流式数据集 (Lazy Load)
        # split="train" 是必须的，即使你的文件不叫 train
        try:
            self.hf_dataset = load_dataset(
                "json", 
                data_files=self.path, 
                split="train", 
                streaming=True
            )
        except Exception as e:
            log.error(f"Failed to load dataset from {self.path}: {e}")
            raise e

        # 2. DDP 分片 (Sharding)
        # HF datasets 的 streaming 模式支持 shard，它会自动处理跳过逻辑
        if self.world_size > 1:
            self.hf_dataset = datasets.distributed.split_dataset_by_node(
                self.hf_dataset, world_size=self.world_size, 
                rank=self.rank
            )
        # 3. 定义分词映射 (利用 batched=True 加速)
        self.tokenized_dataset = self.hf_dataset.map(
            self._batch_tokenize,
            batched=True,
            batch_size=self.batch_size,
        )

    def _batch_tokenize(self, examples: Dict[str, list]) -> Dict[str, list]:
        """批量分词回调函数"""
        return self.tokenizer(
            examples["text"],
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer_ids = []
        yielded_count = 0
        
        # 重置统计
        self.total_processed_tokens = 0
        self.total_yielded_tokens = 0
        self._last_print_tokens = 0

        # 创建迭代器
        iterator = iter(self.tokenized_dataset)

        try:
            for batch in iterator:
                # 注意：由于使用了 batched=True 的 map，
                # 这里的 batch['input_ids'] 可能是一个列表（如果不被拼接）
                # 但在 streaming 模式下，map 通常会逐个或按批返回
                # input_ids 此时是一个 token list: [101, 292, ...]
                
                tokens = batch["input_ids"]
                if not tokens:
                    continue
                
                # 边界检查：检查是否有 token_id 超出 vocab_size 范围
                vocab_size = 128256
                if any(token_id >= vocab_size for token_id in tokens):
                    continue
                
                # 统计处理的 Token
                num_tokens = len(tokens)
                self.total_processed_tokens += num_tokens
                buffer_ids.extend(tokens)

                # Packing Logic: 凑够 block_size 就产出
                while len(buffer_ids) >= self.block_size:
                    chunk = buffer_ids[: self.block_size]
                    buffer_ids = buffer_ids[self.block_size :]
                    
                    yielded_count += 1
                    self.total_yielded_tokens += len(chunk)
                    

                    # 转为 Tensor 并 Yield
                    seq_tensor = torch.tensor(chunk, dtype=torch.long)
                    yield dict(
                        input_ids=seq_tensor,
                        labels=seq_tensor, # Causal LM 任务 labels = input_ids
                    )


        except StopIteration:
            pass
        except Exception as e:
            log.error(f"Error during iteration at rank {self.rank}: {e}")
            raise e
        
        

    def get_token_counts(self):
        return {
            "total_processed_tokens": self.total_processed_tokens,
            "total_yielded_tokens": self.total_yielded_tokens,
            "rank": self.rank
        }

# 辅助函数保持不变
def get_dataset_length(dataset) -> Optional[int]:
    try:
        return len(dataset)
    except (TypeError, AttributeError):
        return None