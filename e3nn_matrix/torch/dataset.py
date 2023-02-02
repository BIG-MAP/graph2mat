import logging
from torch import multiprocessing
from pathlib import Path
from typing import Sequence, Union, Optional
import threading
import numpy as np

import torch.utils.data

from ..data.configuration import load_orbital_config_from_run, PhysicsMatrixType
from ..data.periodic_table import AtomicTableWithEdges
from .data import OrbitalMatrixData

class OrbitalMatrixDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        runpaths: Sequence[Union[Path, str]],
        z_table: AtomicTableWithEdges,
        out_matrix: Optional[PhysicsMatrixType]=None,
        symmetric_matrix: bool=False,
        sub_atomic_matrix: bool=True,
    ):
        self.runpaths = runpaths
        self.out_matrix: Optional[PhysicsMatrixType] = out_matrix
        self.sub_atomic_matrix = sub_atomic_matrix
        self.symmetric_matrix = symmetric_matrix
        self.z_table = z_table

    def __len__(self):
        return len(self.runpaths)

    def __getitem__(self, index: int):
        path = self.runpaths[index]
        config = load_orbital_config_from_run(path, out_matrix=self.out_matrix)
        return OrbitalMatrixData.from_config(
            config,
            z_table=self.z_table,
            sub_atomic_matrix=self.sub_atomic_matrix,
            symmetric_matrix = self.symmetric_matrix,
        )

class InMemoryData(torch.utils.data.Dataset):
    """
    Wrapper for a dataset. Loads all data into memory.
    """

    def __init__(self, dataset, size=None, **kwargs):
        super().__init__(**kwargs)
        size = size or len(dataset)
        self.data_objects = [dataset[i] for i in range(size)]

    def __len__(self):
        return len(self.data_objects)

    def __getitem__(self, index):
        return self.data_objects[index]

class SimpleCounter():
    def __init__(self):
        self.reset()
    def inc(self):
        self.count += 1
    def reset(self):
        self.count = 0
    def get_count(self):
        return self.count

def _rotating_pool_worker(dataset, rng, queue):
    while True:
        for index in rng.permutation(len(dataset)).tolist():
            queue.put(dataset[index])


def _transfer_thread(queue: multiprocessing.Queue, datalist: list, counter: SimpleCounter):
    while True:
        for index in range(len(datalist)):
            datalist[index] = queue.get()
            counter.inc()

class RotatingPoolData(torch.utils.data.Dataset):
    """
    Wrapper for a dataset that continously loads data into a smaller pool.
    The data loading is performed in a separate process and is assumed to be IO bound.
    """

    def __init__(self, dataset, pool_size, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.parent_data = dataset
        self.rng = np.random.default_rng()
        self.counter = SimpleCounter()
        self.manager = multiprocessing.Manager()
        logging.debug("Filling rotating data pool of size %d" % pool_size)
        data_list = [
            self.parent_data[i]
            for i in self.rng.integers(
                0, high=len(self.parent_data), size=self.pool_size, endpoint=False
            ).tolist()
        ]
        self.data_pool = self.manager.list(data_list)
        self.loader_queue = multiprocessing.Queue(2)

        # Start loaders
        self.loader_process = multiprocessing.Process(
            target=_rotating_pool_worker,
            args=(self.parent_data, self.rng, self.loader_queue),
            daemon=True,
        )
        self.transfer_thread = threading.Thread(
            target=_transfer_thread,
            args=(self.loader_queue, self.data_pool, self.counter),
            daemon=True,
        )
        self.loader_process.start()
        self.transfer_thread.start()

    def __len__(self):
        return self.pool_size

    def __getitem__(self, index):
        return self.data_pool[index]

    def get_data_pool(self):
        """
        Get the minimal dataset handle object for transfering to dataloader workers

        Returns
        -------
            Multiprocessing proxy data object

        """
        return self.data_pool
