from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class AbstractCompressor(ABC):
    """Abstract base class for compressors."""

    # NOTE[PAN]: This interface definitely needs to be changed later to
    # implement algo like kvzip which requires a full prefill with repeat prompt
    @abstractmethod
    def compress(
        self,
        req_to_token_pool: ReqToTokenPool,
        req_to_token_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        compressed_req_to_token_pool: ReqToTokenPool,
        compressed_req_to_token_pool_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        # NOTE[PAN]: We will simply write to where indices points to in place
        pass

    @abstractmethod
    def compress_async(
        self,
        req_to_token_pool: ReqToTokenPool,
        req_to_token_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        compressed_req_to_token_pool: ReqToTokenPool,
        compressed_req_to_token_pool_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        pass


def allocate_req_slots_for_compression(
    compressed_req_to_token_pool: ReqToTokenPool,
    num_reqs: int,
    reqs: list["Req"] | None,
):
    assert isinstance(compressed_req_to_token_pool, ReqToTokenPool)
    compressed_req_pool_indices = compressed_req_to_token_pool.alloc(num_reqs)

    if compressed_req_pool_indices is None:
        raise RuntimeError(
            "alloc_req_slots runs out of memory. "
            "Please set a smaller number for `--max-running-requests`. "
            f"{compressed_req_to_token_pool.available_size()=}, "
            f"{num_reqs=}, "
        )
    return compressed_req_pool_indices


# Used to compress for a whole batch
def compress_batch(
    batch: ScheduleBatch,
):
    # Check a bunch of cases that we did not consider yet
    assert batch.tree_cache.page_size == 1
    assert not batch.model_config.is_encoder_decoder
    assert batch.compressed_req_to_token_pool is not None
    # Only work for ReqToTokenPool for now
    assert isinstance(batch.compressed_req_to_token_pool, ReqToTokenPool)
    assert batch.compressor is not None

    # Allocate from the compressed_req_to_token_pool first
    if batch.compressed_req_pool_indices is None:
        compressed_req_pool_indices = allocate_req_slots_for_compression(
            batch.compressed_req_to_token_pool,
            len(batch.reqs),
            batch.reqs,
        )
        compressed_req_pool_indices_cpu = torch.tensor(
            compressed_req_pool_indices, dtype=torch.int64
        )
        compressed_req_pool_indices_gpu = compressed_req_pool_indices_cpu.to(
            batch.device, non_blocking=True
        )
        batch.compressed_req_pool_indices = compressed_req_pool_indices_gpu

    locs = batch.seq_lens.clone()

    # Invoke compressor to compress
    compressed_seq_lens, compressed_seq_lens_sum = batch.compressor.compress(
        batch.req_to_token_pool,
        batch.req_pool_indices,
        locs,
        batch.compressed_req_to_token_pool,
        batch.compressed_req_pool_indices,
    )

    batch.compressed_seq_lens = compressed_seq_lens
    batch.compressed_seq_lens_sum = compressed_seq_lens_sum


def compress_req(
    req: Req,
):
    pass


def compress_async(
    batch: ScheduleBatch,
):
    pass


def wait_for_compression():
    pass
