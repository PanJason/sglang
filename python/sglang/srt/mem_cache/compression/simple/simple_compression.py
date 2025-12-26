import logging

import torch

from sglang.srt.mem_cache.common_compression import AbstractCompressor
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


class RandomCompressor(AbstractCompressor):
    """A simple compressor that randomly drops half of the tokens."""

    def __init__(self):
        super().__init__()
        # NOTE[PAN]: Hardcoded for now. Randomly keep 90% of tokens
        self.random_ratio = 0.9

    def compress(
        self,
        req_to_token_pool: ReqToTokenPool,
        req_to_token_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        compressed_req_to_token_pool: ReqToTokenPool,
        compressed_req_to_token_pool_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        logger.info("Using RandomCompressor to compress tokens.")
        device = req_to_token_pool_indices.device
        batch_size = req_to_token_pool_indices.size(0)
        compressed_seq_lens = torch.ceil(
            seq_lens.to(torch.float32) * self.random_ratio
        ).to(dtype=torch.int64, device=device)
        compressed_seq_lens_sum = compressed_seq_lens.sum().item()
        seq_lens_cpu = seq_lens.to("cpu")
        compressed_seq_lens_cpu = compressed_seq_lens.to("cpu")

        for i in range(batch_size):
            original_start = req_to_token_pool_indices[i]
            compressed_start = compressed_req_to_token_pool_indices[i]
            original_len = int(seq_lens_cpu[i])
            compressed_len = int(compressed_seq_lens_cpu[i])
            if compressed_len == 0:
                continue

            # Randomly select tokens to keep
            indices = torch.randperm(original_len, device=device)[:compressed_len]
            indices, _ = torch.sort(indices)  # Sort to maintain order

            compressed_req_to_token_pool.write(
                (compressed_start, torch.arange(compressed_len, device=device)),
                req_to_token_pool.req_to_token[original_start, indices],
            )

        return compressed_seq_lens, compressed_seq_lens_sum

    def compress_async(
        self,
        req_to_token_pool: ReqToTokenPool,
        req_to_token_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        compressed_req_to_token_pool: ReqToTokenPool,
        compressed_req_to_token_pool_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        # For simplicity, we will just call the synchronous version here.
        compressed_seq_lens, compressed_seq_lens_sum = self.compress(
            req_to_token_pool,
            req_to_token_pool_indices,
            seq_lens,
            compressed_req_to_token_pool,
            compressed_req_to_token_pool_indices,
        )
        return compressed_seq_lens, compressed_seq_lens_sum


class TruncateCompressor(AbstractCompressor):
    """A simple compressor that truncates half of the tokens."""

    def __init__(self):
        super().__init__()
        self.direction = "front"
        self.truncate_ratio = 0.1

    def compress(
        self,
        req_to_token_pool,
        req_to_token_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        compressed_req_to_token_pool,
        compressed_req_to_token_pool_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        device = req_to_token_pool_indices.device
        batch_size = req_to_token_pool_indices.size(0)
        seq_lens_cpu = seq_lens.to("cpu")
        keep_lens_cpu = []
        for i in range(batch_size):
            seq_len = int(seq_lens_cpu[i])
            drop_len = int(seq_len * self.truncate_ratio)
            keep_len = max(seq_len - drop_len, 0)
            keep_lens_cpu.append(keep_len)
        compressed_seq_lens = torch.tensor(
            keep_lens_cpu, dtype=torch.int64, device=device
        )
        compressed_seq_lens_sum = compressed_seq_lens.sum().item()

        for i in range(batch_size):
            original_start = req_to_token_pool_indices[i]
            compressed_start = compressed_req_to_token_pool_indices[i]
            original_len = int(seq_lens_cpu[i])
            compressed_len = keep_lens_cpu[i]
            if compressed_len == 0:
                continue

            if self.direction == "front":
                start = original_len - compressed_len
            elif self.direction == "back":
                start = 0
            else:
                raise ValueError(
                    f"Unsupported truncate direction: {self.direction}. "
                    "Expected 'front' or 'back'."
                )

            tokens = req_to_token_pool.req_to_token[
                original_start, start : start + compressed_len
            ]
            compressed_req_to_token_pool.write(
                (compressed_start, slice(0, compressed_len)),
                tokens,
            )

        return compressed_seq_lens, compressed_seq_lens_sum

    def compress_async(
        self,
        req_to_token_pool,
        req_to_token_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        compressed_req_to_token_pool,
        compressed_req_to_token_pool_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        # For simplicity, we will just call the synchronous version here.
        compressed_seq_lens, compressed_seq_lens_sum = self.compress(
            req_to_token_pool,
            req_to_token_pool_indices,
            seq_lens,
            compressed_req_to_token_pool,
            compressed_req_to_token_pool_indices,
        )
        return compressed_seq_lens, compressed_seq_lens_sum
