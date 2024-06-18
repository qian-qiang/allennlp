from typing import Dict, Union, Iterator

import torch

from allennlp.common.registrable import Registrable
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary

# `TensorDict`是我们用来表示批处理的类型。
TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]

"""
这段代码定义了一个抽象的 `DataLoader` 类，用于从数据源生成批处理数据。具体功能如下：

1. **`TensorDict` 类型**：
   - 定义了一个类型别名 `TensorDict`，用于表示批处理数据的字典结构。

2. **`DataLoader` 类**：
   - 这是一个抽象基类，负责生成 `Instance` 对象的批处理数据。
   - 该类继承自 `Registrable`，允许注册和实例化不同的 `DataLoader` 实现。

3. **抽象方法**：
   - `__len__()`：返回数据集的长度，如果无法实现应抛出 `TypeError`。
   - `__iter__()`：生成一个 `TensorDict` 的迭代器，用于批处理数据。
   - `iter_instances()`：生成一个 `Instance` 的迭代器，用于实例数据。
   - `index_with(vocab)`：使用词汇表 `Vocabulary` 为数据建立索引。
   - `set_target_device(device)`：设置生成批处理张量时的目标设备。

4. **默认实现**：
   - 指定了默认实现为 `MultiProcessDataLoader`，用于多进程数据加载。

### 总结
这段代码提供了一个抽象的框架，用于定义从数据源生成批处理数据的逻辑。具体的 `DataLoader` 实现需要继承该类并实现上述抽象方法，以便在训练或评估模型时使用。
"""

class DataLoader(Registrable):
    """
    `DataLoader`负责从[`DatasetReader`](https://docs.allennlp.org/main/api/data/dataset_readers/dataset_reader/#datasetreader)
    或其他数据源生成`Instance`的批次。

    这是一个纯粹的抽象基类。所有具体的子类必须提供以下方法的实现：

      - `__iter__()`：创建一个`TensorDict`的可迭代对象，
      - `iter_instances()`：创建一个`Instance`的可迭代对象，
      - `index_with(vocab)`：用词汇表为数据建立索引，
      - `set_target_device(device)`：更新生成批次张量时的目标设备。

    此外，这个类还应该在可能的情况下实现`__len__()`方法。

    默认实现是[`MultiProcessDataLoader`](https://docs.allennlp.org/main/api/data/data_loaders/multiprocess_data_loader/#multiprocessdataloader)。
    """

    default_implementation = "multiprocess"

    def __len__(self) -> int:
        """
        返回数据集的长度。
        如果子类无法实现该方法，应抛出`TypeError`。
        """
        raise TypeError

    def __iter__(self) -> Iterator[TensorDict]:
        """
        返回一个`TensorDict`的迭代器。
        子类必须实现此方法以生成批次数据。
        """
        raise NotImplementedError

    def iter_instances(self) -> Iterator[Instance]:
        """
        返回一个`Instance`的迭代器。
        子类必须实现此方法以生成实例数据。
        """
        raise NotImplementedError

    def index_with(self, vocab: Vocabulary) -> None:
        """
        用词汇表为数据建立索引。
        子类必须实现此方法以准备数据索引。

        Args:
            vocab (Vocabulary): 用于索引的数据词汇表。
        """
        raise NotImplementedError

    def set_target_device(self, device: torch.device) -> None:
        """
        更新生成批次张量时的目标设备。
        子类必须实现此方法以设置目标设备。

        Args:
            device (torch.device): 用于生成批次张量的目标设备。
        """
        raise NotImplementedError
