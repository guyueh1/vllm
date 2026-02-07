from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
import multiprocessing.resource_tracker

import numpy as np


@dataclass
class BlockCachePage:
    pid: int
    mem: SharedMemory
    _mem_view: np.ndarray = None

    def __post_init__(self):
        if self._mem_view is None:
            self._mem_view = np.frombuffer(bytearray(self.mem.buf), dtype=np.uint8)


class BlockCacheProducerRef:
    def __init__(self, page_max_size: int, block_size: int):
        self.page_max_size: int = page_max_size
        self.block_size: int = block_size
        self.block_aligned_size: int = ((block_size + 64 - 1) // 64) * 64
        self.num_blocks_per_page: int = self.page_max_size // self.block_aligned_size
        print(f"DEBUG: BlockCacheProducerRef: page max size       = {self.page_max_size}", flush=True)
        print(f"DEBUG: BlockCacheProducerRef: block size          = {self.block_size}", flush=True)
        print(f"DEBUG: BlockCacheProducerRef: block aligned size  = {self.block_aligned_size}", flush=True)
        print(f"DEBUG: BlockCacheProducerRef: num blocks per page = {self.num_blocks_per_page}", flush=True)
        self.pages: list = []

    def copy_to_gid(self, gid: int, data: np.ndarray):
        # NB: if a gid was returned from the disaggregated block cache instance,
        # then the corresponding page is guaranteed to already exist.
        pid = gid // self.num_blocks_per_page
        blk = gid % self.num_blocks_per_page
        # print(f"DEBUG: BlockCacheProducerRef.copy_to_gid: gid = {gid} pid = {pid} blk = {blk}", flush=True)
        while pid >= len(self.pages):
            tmp_pid = len(self.pages)
            # print(f"DEBUG: BlockCacheProducerRef.copy_to_gid: deref pid = {tmp_pid}", flush=True)
            mem = SharedMemory(
                name=f"nemo_rl.block_cache.page.{tmp_pid}",
                size=self.page_max_size,
                create=False,
                # track=True,
            )
            multiprocessing.resource_tracker.register(mem._name, "shared_memory")
            page = BlockCachePage(pid=tmp_pid, mem=mem)
            self.pages.append(page)
        # print(f"DEBUG: BlockCacheProducerRef.copy_to_gid: pid = {pid} num pages = {len(self.pages)}", flush=True)
        page = self.pages[pid]
        # print(f"DEBUG: BlockCacheProducerRef.copy_to_gid: data type = {type(data).__name__}", flush=True)
        if isinstance(data, np.ndarray):
            pass
            # print(f"DEBUG: BlockCacheProducerRef.copy_to_gid: data shape = {data.shape} dtype = {data.dtype}", flush=True)
        data_view = data.ravel().view(np.uint8)
        # print(f"DEBUG: BlockCacheProducerRef.copy_to_gid: view shape = {data_view.shape} dtype = {data_view.dtype}", flush=True)
        start = blk * self.block_aligned_size
        # print(f"DEBUG: BlockCacheProducerRef.copy_to_gid: start = {start}", flush=True)
        end = start + int(data_view.shape[0])
        # print(f"DEBUG: BlockCacheProducerRef.copy_to_gid: end = {end}", flush=True)
        assert end <= start + self.block_aligned_size
        assert end == start + self.block_size
        # print(f"DEBUG: BlockCacheProducerRef.copy_to_gid: copy...", flush=True)
        try:
            page._mem_view[start:end] = data_view[:]
            # print(f"DEBUG: BlockCacheProducerRef.copy_to_gid: done", flush=True)
        except Exception as e:
            pass
            # print(f"DEBUG: BlockCacheProducerRef.copy_to_gid: except: {type(e).__name__} {e}", flush=True)
