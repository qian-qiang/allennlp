from dataclasses import dataclass
import itertools
from os import PathLike
from typing import Iterable, Iterator, Optional, Union, TypeVar, Dict, List
import logging
import warnings

import torch.distributed as dist

from allennlp.data.instance import Instance
from allennlp.common import util
from allennlp.common.registrable import Registrable


logger = logging.getLogger(__name__)

"""这段代码定义了一个 DatasetReader 类及相关的辅助类，用于处理数据集的读取、分片（sharding）和分布式训练环境下的数据处理"""

@dataclass
class WorkerInfo:
    """
    当 `DatasetReader` 在多进程 `DataLoader` 中使用时，包含有关工作线程上下文的信息。

    可以通过 `DatasetReader` 中的 [`get_worker_info()`](#get_worker_info) 方法来访问这些信息。
    """

    num_workers: int
    """
    工作线程的总数量。
    """

    id: int
    """
    当前工作线程的从0开始的ID。
    """


@dataclass
class DistributedInfo:
    """
    当读取器在分布式训练中使用时，包含关于全局进程等级和总体世界大小的信息。

    可以通过 [`get_distributed_info()`](#get_distributed_info) 方法从 `DatasetReader` 中访问这些信息。
    """

    world_size: int
    """
    分布式组中进程的总数。
    """

    global_rank: int
    """
    当前进程在分布式组中的0索引ID。
    这将在0到 `world_size - 1`（含）之间。
    """


_T = TypeVar("_T")

PathOrStr = Union[PathLike, str]
DatasetReaderInput = Union[PathOrStr, List[PathOrStr], Dict[str, PathOrStr]]

class DatasetReader(Registrable):
    """
    `DatasetReader` 知道如何将包含数据集的文件转换为一组 `Instance`。要实现自己的 `DatasetReader`，
    只需覆盖 [`_read(file_path)`](#_read) 方法以返回一个 `Instance` 的可迭代对象。
    最好是一个惰性生成器，逐个生成实例。

    所有除文件路径外 `_read` 数据所需的参数应传递给 `DatasetReader` 的构造函数。

    你还应该实现 [`text_to_instance(*inputs)`](#text_to_instance) 方法，
    该方法用于将原始数据转换为 `Instance`。这个方法在使用 `Predictor` 与你的 reader 时是必需的。

    通常情况下，`_read()` 方法实现时会调用 `text_to_instance()`。

    # 参数

    max_instances: `int`，可选（默认=`None`）
        如果指定，将在读取此数目的实例后停止。这对于调试很有用。
        设置此参数会禁用缓存。

    manual_distributed_sharding: `bool`，可选（默认=`False`）
        默认情况下，在分布式设置中使用时，`DatasetReader` 确保每个训练进程只接收数据的一个子集。
        它通过在每个工作进程中读取整个数据集，但过滤掉不需要的实例来实现这一点。

        虽然这确保了每个工作进程收到唯一的实例，但这种方法效率不高，
        因为每个工作进程仍然需要处理数据集中的每个实例。

        更好的方式是在 `_read()` 方法内手动处理过滤，此时应将 `manual_distributed_sharding` 设置为 `True`，
        以便基类知道你正在处理过滤。

        参见下面关于如何实现的部分。

    manual_multiprocess_sharding: `bool`，可选（默认=`False`）
        这与 `manual_distributed_sharding` 参数类似，但适用于多进程数据加载。
        默认情况下，当此 reader 被多进程数据加载器使用（即具有 `num_workers > 1` 的 `DataLoader`）时，
        每个工作进程会过滤掉除了需要的实例之外的所有实例，以避免重复。

        但是，除非在 `_read()` 方法中实现分片，否则在你的 `DataLoader` 中使用多个工作进程没有实际好处。
        因此，当你在 `_read()` 方法中处理了分片逻辑时，应设置 `manual_multiprocess_sharding` 为 `True`，
        就像 `manual_distributed_sharding` 一样。

        参见下面关于如何实现的部分。

    serialization_dir: `str`，可选（默认=`None`）
        保存训练输出的目录，或加载模型的目录。

        !!! 注意
            这通常不会在配置文件中给出条目。在使用内置的 `allennlp` 命令时会自动设置。

    # 在多进程或分布式数据加载中使用你的 reader

    若要使你的 `DatasetReader` 在多进程或分布式数据加载上更高效，可能需要更新两个内容。

    1. `_read()` 方法应该处理仅生成每个特定工作进程所需的实例。

        这很重要，因为在分布式或多进程 `DataLoader` 设置中，用于过滤 `Instance` 的默认机制效率不高，
        因为每个工作进程仍然需要处理数据集中的每个单个 `Instance`。

        但通过在 `_read()` 方法内手动处理过滤或分片，每个工作进程只需执行创建实例所需工作的子集。

        例如，如果你正在使用 2 个 GPU 进行训练，而 `_read()` 方法按行读取文件，每行创建一个 `Instance`，
        你可以在 `_read()` 内部检查节点排名，然后从对应节点排名的行开始丢弃每一行的其他行。

        辅助方法 [`shard_iterable()`](#shard_iterable) 使这一点变得更容易。
        你可以在 `_read()` 方法中的任何可迭代对象周围包装它，并返回一个迭代器，根据分布式训练或多进程加载上下文跳过正确的项。
        无论实际上是否在使用分布式训练或多进程加载，都可以调用此方法。

        但请记住，当在 `_read()` 内部手动处理分片时，需要让 `DatasetReader` 知道这一点，
        以便它不执行任何额外的过滤。因此，你需要确保 `self.manual_distributed_sharding` 和
        `self.manual_multiprocess_sharding` 都设置为 `True`。

        如果在不设置这些为 `True` 的情况下调用辅助方法 `shard_iterable()`，会引发异常。

    2. 如果 `_read()` 生成的 `Instance` 包含 `TextField`，那么这些 `TextField` 不应该分配任何 token indexers。
        Token indexers 应该在 [`apply_token_indexers()`](#apply_token_indexers) 方法中应用。

        强烈建议这样做，因为如果 `_read()` 方法生成的实例附带了 token indexers，那么当它们在进程间传递时，
        这些 indexers 将被复制。如果你的 token indexers 包含大对象（如 `PretrainedTransformerTokenIndexer`），
        这可能会占用大量内存。

    """

    def __init__(
        self,
        max_instances: Optional[int] = None,
        manual_distributed_sharding: bool = False,
        manual_multiprocess_sharding: bool = False,
        serialization_dir: Optional[str] = None,
    ) -> None:
        # Do some validation.
        if max_instances is not None and max_instances < 0:
            raise ValueError("If specified, max_instances should be a positive int")

        self.max_instances = max_instances
        self.manual_distributed_sharding = manual_distributed_sharding
        self.manual_multiprocess_sharding = manual_multiprocess_sharding
        self.serialization_dir = serialization_dir
        self._worker_info: Optional[WorkerInfo] = None
        self._distributed_info: Optional[DistributedInfo] = None

        # 如果我们实际处于主进程中，可以使用torch工具找到信息
        if util.is_distributed():
            self._distributed_info = DistributedInfo(dist.get_world_size(), dist.get_rank())

    def read(self, file_path: DatasetReaderInput) -> Iterator[Instance]:
        """
        返回一个实例的迭代器，这些实例可以从文件路径中读取。
        """
        for instance in self._multi_worker_islice(self._read(file_path)):  # type: ignore
            if self._worker_info is None:
                # 如果不是在子进程中运行，则可以立即应用token_indexers。
                self.apply_token_indexers(instance)
            yield instance

    def _read(self, file_path) -> Iterable[Instance]:
        """
        从给定的 `file_path` 中读取实例并将它们作为 `Iterable` 返回。

        强烈建议使用生成器，以便用户可以选择懒惰地读取数据集。
        """
        # 注意：这里故意不对 `file_path` 进行类型标注。
        # 从技术上讲，类型应该是 `DatasetReaderInput`，但是许多 `DatasetReader` 的子类
        # 定义它们的 `_read()` 方法来接受更具体的类型，比如只是 `str`。但是这样做会导致
        # 类型错误，详见：https://mypy.readthedocs.io/en/stable/common_issues.html#incompatible-overrides
        raise NotImplementedError

    def text_to_instance(self, *inputs) -> Instance:
        """
        将文本输入转换为 `Instance` 实例所需的任何标记化或处理操作。这主要用于
        :class:`~allennlp.predictors.predictor.Predictor`，它将文本输入作为 JSON
        对象并需要处理它以输入模型。

        在这里的意图是在 :func:`_read` 和模型服务时共享代码，或者任何需要从新数据进行预测的时候。
        我们需要以与训练时相同的方式处理数据。允许 `DatasetReader` 处理新文本让我们能够实现这一点，
        因为我们可以在提供预测时调用 `DatasetReader.text_to_instance`。

        不幸的是，这里的输入类型描述相当模糊。`Predictor` 将不得不对其使用的 `DatasetReader` 类型
        做一些假设，以便向其传递正确的信息。
        """
        raise NotImplementedError

    def apply_token_indexers(self, instance: Instance) -> None:
        """
        如果由该读取器创建的 `Instance` 包含没有 `token_indexers` 的 `TextField`，
        则可以重写此方法来设置那些字段的 `token_indexers`。

        例如，如果您有名为 `"source"` 的 `TextField`，您可以像这样实现此方法：

        ```python
        def apply_token_indexers(self, instance: Instance) -> None:
            instance["source"].token_indexers = self._token_indexers
        ```

        如果您的 `TextField` 包装在 `ListField` 中，您可以通过 `field_list` 访问它们。
        例如，如果您有一个 `"source"` 字段包含 `ListField[TextField]` 对象，您可以：

        ```python
        for text_field in instance["source"].field_list:
            text_field.token_indexers = self._token_indexers
        ```
        """
        pass

    def get_worker_info(self) -> Optional[WorkerInfo]:
        """
        当 `DatasetReader` 在多进程 `DataLoader` 的工作器中使用时，提供一个 [`WorkerInfo`](#WorkerInfo) 对象。

        如果读取器在主进程中，则返回 `None`。

        !!! 注意
            这与分布式训练不同。如果 `DatasetReader` 在分布式训练中使用，`get_worker_info()` 只会提供
            关于其节点内的 `DataLoader` 工作器的信息。

            使用 [`get_distributed_info`](#get_distributed_info) 获取分布式训练上下文的信息。
        """
        return self._worker_info

    def get_distributed_info(self) -> Optional[DistributedInfo]:
        """
        当读取器在分布式训练中使用时，提供一个 [`DistributedInfo`](#DistributedInfo) 对象。

        如果不在分布式训练中，则返回 `None`。
        """
        return self._distributed_info

    def _set_worker_info(self, info: Optional[WorkerInfo]) -> None:
        """
        仅在内部使用。
        """
        self._worker_info = info

    def _set_distributed_info(self, info: Optional[DistributedInfo]) -> None:
        """
        仅在内部使用。
        """
        self._distributed_info = info

    def shard_iterable(self, iterable: Iterable[_T]) -> Iterator[_T]:
        """
        辅助方法，根据当前节点排名（用于分布式训练）和工作器 ID（用于多进程数据加载），确定要跳过的可迭代对象中的项。
        """
        if not self.manual_distributed_sharding or not self.manual_multiprocess_sharding:
            raise ValueError(
                "self.shard_iterable() 被调用，但 self.manual_distributed_sharding 和 "
                "self.manual_multiprocess_sharding 未设置为 True。您是否忘记在构造函数中调用 "
                "super().__init__(manual_distributed_sharding=True, manual_multiprocess_sharding=True)？"
            )

        sharded_slice: Iterator[_T] = iter(iterable)

        if util.is_distributed():
            sharded_slice = itertools.islice(
                sharded_slice, dist.get_rank(), None, dist.get_world_size()
            )

        if self._worker_info is not None:
            sharded_slice = itertools.islice(
                sharded_slice, self._worker_info.id, None, self._worker_info.num_workers
            )

        # 我们无法确定需要生成多少个实例。
        # _multi_worker_islice() 负责计算这一点。但我们确切地知道它不会超过 max_instances。
        if self.max_instances is not None:
            sharded_slice = itertools.islice(sharded_slice, self.max_instances)

        return sharded_slice

    def _multi_worker_islice(
        self,
        iterable: Iterable[_T],
    ) -> Iterator[_T]:
        """
        这个方法与 `shard_iterable` 类似，但仅供内部使用。

        它具有一些额外的逻辑，用于根据分布式或多进程上下文以及是否在 `_read()` 方法中手动处理分片来处理 `max_instances`。

        参数:
            iterable (Iterable[_T]): 要分片的可迭代对象。

        返回:
            Iterator[_T]: 分片后的迭代器。
        """
        # 这里有些复杂的逻辑，因为任何给定的读取器可能实现了或未实现多进程和分布式分片。
        # 我们必须处理所有可能性。

        sharded_slice: Iterator[_T] = iter(iterable)

        # 随着处理的进行，我们将根据进行的分片方式调整 max_instances。
        # 最后，我们希望确保在所有工作器进程收集的实例总数等于 self.max_instances。
        max_instances = self.max_instances

        if self._distributed_info is not None:
            if max_instances is not None:
                # 需要缩减 max_instances，因为否则每个节点都会读取 self.max_instances，
                # 但我们实际上希望在所有节点上总共读取 self.max_instances。
                if self._distributed_info.global_rank < (
                    max_instances % self._distributed_info.world_size
                ):
                    max_instances = max_instances // self._distributed_info.world_size + 1
                else:
                    max_instances = max_instances // self._distributed_info.world_size

            if not self.manual_distributed_sharding:
                sharded_slice = itertools.islice(
                    sharded_slice,
                    self._distributed_info.global_rank,
                    None,
                    self._distributed_info.world_size,
                )

        if self._worker_info is not None:
            if max_instances is not None:
                # 类似于上面的分布式情况，我们需要调整 max_instances。
                if self._worker_info.id < (max_instances % self._worker_info.num_workers):
                    max_instances = max_instances // self._worker_info.num_workers + 1
                else:
                    max_instances = max_instances // self._worker_info.num_workers

            if not self.manual_multiprocess_sharding:
                warnings.warn(
                    "使用多进程数据加载，但未将 DatasetReader.manual_multiprocess_sharding 设置为 True。\n"
                    "您是否忘记设置此选项？\n"
                    "如果在您的 _read() 方法中未处理多进程分片逻辑，则使用多个工作器可能没有任何好处。",
                    UserWarning,
                )
                sharded_slice = itertools.islice(
                    sharded_slice, self._worker_info.id, None, self._worker_info.num_workers
                )

        if max_instances is not None:
            sharded_slice = itertools.islice(sharded_slice, max_instances)

        return sharded_slice

