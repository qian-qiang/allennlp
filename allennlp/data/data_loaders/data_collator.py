from typing import List
from transformers.data.data_collator import DataCollatorForLanguageModeling
from allennlp.common import Registrable
from allennlp.data.batch import Batch
from allennlp.data.data_loaders.data_loader import TensorDict
from allennlp.data.instance import Instance

"""
这段代码主要用于将一组`Instance`对象转换为用于训练或评估的张量(tensor)批处理数据，特别是针对语言模型任务。代码分为几部分，每部分都有特定的功能:

1. **数据整理函数(`allennlp_collate`)**:
   - 将`Instance`对象的列表转换为`TensorDict`，这是一个包含批处理数据的字典。

2. **数据整理器基类(`DataCollator`)**:
   - 一个抽象基类，允许对不同批次的张量进行动态操作。
   - 它是`Registrable`的，可以通过名称进行注册和实例化。
   - 提供了一个未实现的`__call__`方法，子类需要实现该方法。

3. **默认数据整理器(`DefaultDataCollator`)**:
   - 继承自`DataCollator`，实现了将`Instance`列表转换为`TensorDict`的默认行为。

4. **语言模型数据整理器(`LanguageModelingDataCollator`)**:
   - 专门用于语言模型任务的数据整理器。
   - 初始化时需要一个预训练模型的名称、是否使用掩码语言模型(MLM)以及相关参数。
   - 使用`transformers`库中的`DataCollatorForLanguageModeling`来处理掩码语言模型的特定需求。
   - 在将`Instance`列表转换为`TensorDict`后，对令牌进行处理(如应用掩码)。

### 代码总结:
该代码定义了一个数据整理框架，用于在训练或评估自然语言处理模型时，将一组`Instance`对象转换为适合模型输入的张量批处理数据。特别地，
它提供了默认的整理器和针对语言模型任务的整理器，后者能够处理掩码语言模型的特定需求。通过注册机制，用户可以轻松地扩展和定制数据整理器以适应不同的任务和模型。
"""

def allennlp_collate(instances: List[Instance]) -> TensorDict:
    """
    将`Instance`的列表转换为`TensorDict`批处理。
    这是默认用于将一组`Instance`转换为`TensorDict`批处理的方法。

    Args:
        instances (List[Instance]): 一个`Instance`对象的列表。

    Returns:
        TensorDict: 包含批处理数据的字典。
    """
    batch = Batch(instances)
    return batch.as_tensor_dict()


class DataCollator(Registrable):
    """
    类似于[Transformers]中的`DataCollator`。
    允许对不同批次中的张量进行一些动态操作。
    因为这个方法在每个epoch之前运行，以将`List[Instance]`转换为`TensorDict`。

    参考:https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py
    """
    default_implementation = "allennlp"

    def __call__(self, instances: List[Instance]) -> TensorDict:
        """
        将`Instance`的列表转换为`TensorDict`批处理。

        Args:
            instances (List[Instance]): 一个`Instance`对象的列表。

        Returns:
            TensorDict: 包含批处理数据的字典。
        """
        raise NotImplementedError


@DataCollator.register("allennlp")
class DefaultDataCollator(DataCollator):
    """
    默认的`DataCollator`实现。
    """

    def __call__(self, instances: List[Instance]) -> TensorDict:
        """
        将`Instance`的列表转换为`TensorDict`批处理。

        Args:
            instances (List[Instance]): 一个`Instance`对象的列表。

        Returns:
            TensorDict: 包含批处理数据的字典。
        """
        return allennlp_collate(instances)


@DataCollator.register("language_model")
class LanguageModelingDataCollator(DataCollator):
    """
    用于语言模型的数据整理器。
    注册为名称`LanguageModelingDataCollator`的`DataCollator`。
    """

    def __init__(
            self,
            model_name: str,
            mlm: bool = True,
            mlm_probability: float = 0.15,
            filed_name: str = "source",
            namespace: str = "tokens",
    ):
        """
        初始化语言模型数据整理器。

        Args:
            model_name (str): 预训练模型的名称。
            mlm (bool): 是否使用掩码语言模型(MLM)。
            mlm_probability (float): 掩码概率。
            filed_name (str): 字段名称，默认为"source"。
            namespace (str): 命名空间，默认为"tokens"。
        """
        self._field_name = filed_name
        self._namespace = namespace
        from allennlp.common import cached_transformers

        tokenizer = cached_transformers.get_tokenizer(model_name)
        self._collator = DataCollatorForLanguageModeling(tokenizer, mlm, mlm_probability)
        if hasattr(self._collator, "mask_tokens"):
            # 兼容transformers版本 < 4.10
            self._mask_tokens = self._collator.mask_tokens
        else:
            self._mask_tokens = self._collator.torch_mask_tokens

    def __call__(self, instances: List[Instance]) -> TensorDict:
        tensor_dicts = allennlp_collate(instances)
        tensor_dicts = self.process_tokens(tensor_dicts)
        return tensor_dicts

    def process_tokens(self, tensor_dicts: TensorDict) -> TensorDict:
        """
        处理令牌，应用掩码语言模型(MLM)。

        Args:
            tensor_dicts (TensorDict): 包含批处理数据的字典。

        Returns:
            TensorDict: 包含处理后的令牌的字典。
        """
        inputs = tensor_dicts[self._field_name][self._namespace]["token_ids"]
        inputs, labels = self._mask_tokens(inputs)
        tensor_dicts[self._field_name][self._namespace]["token_ids"] = inputs
        tensor_dicts[self._field_name][self._namespace]["labels"] = labels
        return tensor_dicts
