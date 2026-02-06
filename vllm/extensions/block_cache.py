from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
# from typing import Optional
import multiprocessing.resource_tracker

import numpy as np


@dataclass
class BlockCachePage:
    pid: int
    mem: SharedMemory
    _mem_view: np.ndarray = None

    def __post_init__(self):
        if self._mem_view is None:
            self._mem_view = np.frombuffer(bytearray(self.mem.buffer), dtype=np.uint8)


class BlockCacheRef:
    def __init__(self, page_max_size: int, block_size: int):
        self.page_max_size: int = page_max_size
        self.block_size: int = block_size
        self.block_aligned_size: int = ((block_size + 64 - 1) // 64) * 64
        self.num_blocks_per_page: int = self.page_max_size // self.block_aligned_size
        self.pages: list = []

    def copy_to_gid(self, gid: int, data: np.ndarray):
        # NB: if a gid was returned from the disaggregated block cache instance,
        # then the corresponding page is guaranteed to already exist.
        pid = gid // self.num_blocks_per_page
        blk = gid % self.num_blocks_per_page
        while pid <= len(self.pages):
            tmp_pid = len(self.pages)
            mem = SharedMemory(
                name=f"nemo_rl.block_cache.page.{tmp_pid}",
                size=self.page_max_size,
                create=False,
                # track=True,
            )
            multiprocessing.resource_tracker.register(mem._name, "shared_memory")
            page = BlockCachePage(pid=tmp_pid, mem=mem)
            self.pages.append(page)
        page = self.pages[pid]
        data = np.ascontiguousarray(data)
        data_view = data.view(np.uint8)
        start = blk * self.block_aligned_size
        end = start + len(data_view)
        assert end <= start + self.block_aligned_size
        page._mem_view[start:end] = data_view[:]
