import logging
from torch import multiprocessing
from pathlib import Path
from typing import Sequence, Union
import threading

import numpy as np
import sisl

import torch.utils.data

from .data import MatrixDataProcessor

class OrbitalMatrixDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_data: Sequence[Union[Path, str, sisl.Geometry]],
        data_processor: MatrixDataProcessor,
        load_labels: bool = True,
    ):
        self.input_data = input_data
        self.data_processor = data_processor
        self.load_labels = load_labels

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index: int):
        item = self.input_data[index]
        return self.data_processor.process_input(item, labels=self.load_labels)

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
