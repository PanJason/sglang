import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.common_compression import AbstractCompressor
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    pass

COMPRESSORS = {}


def register_compressor(name: str):
    def decorator(fn):
        COMPRESSORS[name] = fn
        return fn

    return decorator


class DummyCompressor(AbstractCompressor):
    """A no-op compressor that keeps all tokens."""

    def compress(
        self,
        req_to_token_pool: ReqToTokenPool,
        req_to_token_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        compressed_req_to_token_pool: ReqToTokenPool,
        compressed_req_to_token_pool_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        device = req_to_token_pool_indices.device
        batch_size = req_to_token_pool_indices.size(0)
        compressed_seq_lens = seq_lens.to(device)
        compressed_seq_lens_sum = compressed_seq_lens.sum().item()
        seq_lens_cpu = seq_lens.to("cpu")

        for i in range(batch_size):
            original_start = req_to_token_pool_indices[i]
            compressed_start = compressed_req_to_token_pool_indices[i]
            seq_len = int(seq_lens_cpu[i])
            if seq_len == 0:
                continue
            tokens = req_to_token_pool.req_to_token[original_start, :seq_len]
            compressed_req_to_token_pool.write(
                (compressed_start, slice(0, seq_len)),
                tokens,
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
        return self.compress(
            req_to_token_pool,
            req_to_token_pool_indices,
            seq_lens,
            compressed_req_to_token_pool,
            compressed_req_to_token_pool_indices,
        )


@register_compressor("dummy")
def create_dummy_compressor():
    return DummyCompressor()


@register_compressor("random")
def create_random_compressor():
    from sglang.srt.mem_cache.compression.simple.simple_compression import (
        RandomCompressor,
    )

    return RandomCompressor()


@register_compressor("truncate")
def create_truncate_compressor():
    from sglang.srt.mem_cache.compression.simple.simple_compression import (
        TruncateCompressor,
    )

    return TruncateCompressor()


def get_compressor_from_name(name: str) -> AbstractCompressor:
    if name not in COMPRESSORS:
        raise ValueError(f"Invalid compressor: {name}")
    return COMPRESSORS[name]()
