from itertools import islice
from typing import Optional, List, Set

import pytest
import torch.distributed as dist

from allennlp.common import util as common_util
from allennlp.data import Instance
from allennlp.data.dataset_readers import (
    DatasetReader,
    WorkerInfo,
)
from allennlp.data.fields import LabelField


TOTAL_INSTANCES = 100

"""
这段代码是一组用于测试不同数据读取器在分布式和多进程环境下的实例切片行为的测试用例。代码主要测试了四种不同的数据读取器：
MockDatasetReader、MockMmpsDatasetReader、MockMdsDatasetReader和MockMmpdsDatasetReader，
以确保它们在分布式和/或多进程情况下正确地分片读取数据，并且总数和期望的一致。
"""

class MockDatasetReader(DatasetReader):
    def _read(self, file_path):
        for i in range(TOTAL_INSTANCES):
            yield self.text_to_instance(i)

    def text_to_instance(self, index: int):  # type: ignore
        return Instance({"index": LabelField(index, skip_indexing=True)})


class MockMmpsDatasetReader(DatasetReader):
    """
    Implements manual multi-process sharding (MMPS).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(manual_multiprocess_sharding=True, **kwargs)

    def _read(self, file_path):
        start_index = 0
        step_size = 1
        worker_info = self.get_worker_info()
        if worker_info is not None:
            start_index += worker_info.id
            step_size *= worker_info.num_workers
        for i in islice(range(TOTAL_INSTANCES), start_index, None, step_size):
            yield self.text_to_instance(i)

    def text_to_instance(self, index: int):  # type: ignore
        return Instance({"index": LabelField(index, skip_indexing=True)})


class MockMdsDatasetReader(DatasetReader):
    """
    Implements manual distributed sharding (MDS).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(manual_distributed_sharding=True, **kwargs)

    def _read(self, file_path):
        start_index = 0
        step_size = 1
        if common_util.is_distributed():
            start_index += dist.get_rank()
            step_size *= dist.get_world_size()
        for i in islice(range(TOTAL_INSTANCES), start_index, None, step_size):
            yield self.text_to_instance(i)

    def text_to_instance(self, index: int):  # type: ignore
        return Instance({"index": LabelField(index, skip_indexing=True)})


class MockMmpdsDatasetReader(DatasetReader):
    """
    Implements manual multi-process and distributed sharding (MMPDS).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )

    def _read(self, file_path):
        for i in self.shard_iterable(range(TOTAL_INSTANCES)):
            yield self.text_to_instance(i)

    def text_to_instance(self, index: int):  # type: ignore
        return Instance({"index": LabelField(index, skip_indexing=True)})


@pytest.mark.parametrize(
    "world_size, num_workers, max_instances",
    [
        (None, None, 67),
    ],
)
@pytest.mark.parametrize(
    "reader_class",
    [MockDatasetReader],
)
def test_instance_slicing(
    monkeypatch,
    reader_class,
    world_size: Optional[int],
    num_workers: Optional[int],
    max_instances: Optional[int],
):
    """
    Ensure that the intances read by each worker are always unique and the total
    adds up to `max_instances`.
    """
    results: List[Set[int]] = []

    minimum_expected_result_size = max_instances or TOTAL_INSTANCES
    maximum_expected_result_size = max_instances or TOTAL_INSTANCES

    #模拟同时使用分布式和多进程的情况：
    if world_size is not None and num_workers is not None:
        minimum_expected_result_size //= world_size
        minimum_expected_result_size //= num_workers
        maximum_expected_result_size = minimum_expected_result_size + 1
        for global_rank in range(world_size):
            monkeypatch.setattr(common_util, "is_distributed", lambda: True)
            monkeypatch.setattr(dist, "get_rank", lambda: global_rank)
            monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
            for worker_id in range(num_workers):
                reader = reader_class(max_instances=max_instances)
                reader._set_worker_info(WorkerInfo(num_workers, worker_id))
                result = set(
                    x["index"].label for x in reader.read("the-path-doesnt-matter")  # type: ignore
                )
                results.append(result)
    #模拟仅使用分布式的情况：
    elif world_size is not None:
        minimum_expected_result_size //= world_size
        maximum_expected_result_size = minimum_expected_result_size + 1
        for global_rank in range(world_size):
            monkeypatch.setattr(common_util, "is_distributed", lambda: True)
            monkeypatch.setattr(dist, "get_rank", lambda: global_rank)
            monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
            reader = reader_class(max_instances=max_instances)
            result = set(
                x["index"].label for x in reader.read("the-path-doesnt-matter")  # type: ignore
            )
            results.append(result)
    #模拟仅使用多进程的情况：
    elif num_workers is not None:
        minimum_expected_result_size //= num_workers
        maximum_expected_result_size = minimum_expected_result_size + 1
        for worker_id in range(num_workers):
            reader = reader_class(max_instances=max_instances)
            reader._set_worker_info(WorkerInfo(num_workers, worker_id))
            result = set(
                x["index"].label for x in reader.read("the-path-doesnt-matter")  # type: ignore
            )
            results.append(result)
    #默认单进程的情况：
    else:
        reader = reader_class(max_instances=max_instances)
        result = set(
            x["index"].label for x in reader.read("the-path-doesnt-matter")  # type: ignore
        )
        results.append(result)

    # We need to check that all of the result sets are mutually exclusive and that they're
    # union has size `max_instances`.
    # Checking that they're mutually exclusive is equivalent to checking that the sum
    # of the size of each set is equal to the size of the union.

    union: Set[int] = set()
    total: int = 0
    for result in results:
        union |= result
        total += len(result)
        # Also make sure the size of the set is within the expected bounds.
        assert minimum_expected_result_size <= len(result)
        assert len(result) <= maximum_expected_result_size

    assert len(union) == total == (max_instances or TOTAL_INSTANCES)
