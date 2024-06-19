from copy import deepcopy
from typing import Dict, Generic, List, TypeVar, Any

import torch

from allennlp.data.vocabulary import Vocabulary

"""这份代码定义了一个名为 Field 的基类，用于 AllenNLP 框架中处理数据字段，这些字段最终会转换为模型中的张量。
Field 类使用泛型 DataArray 来支持不同类型的数据结构，如 torch.Tensor 和字典。
主要功能和方法包括：
1. count_vocab_items(self, counter): 用于统计需要转换为索引的字符串，以便构建词汇表。
2. index(self, vocab): 使用词汇表将字段中的字符串转换为索引。
3. get_padding_lengths(self): 计算字段中数据的填充长度。
4. as_tensor(self, padding_lengths): 根据填充长度将数据转换为张量。
5. empty_field(self): 生成一个空的字段实例，用于处理不同长度的数据列表。
6. batch_tensors(self, tensor_list): 将多个实例的张量合并为一个批处理张量。
7. duplicate(self): 复制当前字段实例。
此外，还包括用于比较字段实例、生成人类可读表示和获取字段长度的方法。这个类主要用于数据预处理和准备阶段，以适应模型的输入需求。
"""

# 多层嵌套的好处在哪里
DataArray = TypeVar(
    "DataArray", torch.Tensor, Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]
)


# 这段代码定义了一个名为Field的类，它是一种数据实例中某些部分的表示，该类在模型中将作为一个张量（either as an input or an output）。数据实例是一组字段的集合。
# 字段经过最多两个处理步骤：
# （1）将标记化的字段转换为标记符号ID，
# （2）包含标记符号ID（或任何其他数字数据）的字段被填充（如果需要）并转换为张量。
# 字段API具有两个步骤的方法，尽管它们可能不适用于某些具体的字段类——如果您的字段没有需要索引的任何字符串，则不需要实现count_vocab_items或index。这些方法默认为“通過”。
# 一旦计算了词汇表并将所有字段索引，我们将确定填充长度，然后智能地将实例批量在一起并填充到实际张量中。
class Field(Generic[DataArray]):
    """
    A `Field` is some piece of a data instance that ends up as an tensor in a model (either as an
    input or an output).  Data instances are just collections of fields.
    Fields go through up to two steps of processing: (1) tokenized fields are converted into token
    ids, (2) fields containing token ids (or any other numeric data) are padded (if necessary) and
    converted into tensors.  The `Field` API has methods around both of these steps, though they
    may not be needed for some concrete `Field` classes - if your field doesn't have any strings
    that need indexing, you don't need to implement `count_vocab_items` or `index`.  These
    methods `pass` by default.
    Once a vocabulary is computed and all fields are indexed, we will determine padding lengths,
    then intelligently batch together instances and pad them into actual tensors.
    """
    # 以下是Field类的源代码，其中有一些注释。
    __slots__ = []  # type: ignore

    # 以下是Field类的注释。
    # 这个类没有任何实例变量，因为它使用了Python的“类属性”（class attribute）和“实例属性”（instance attribute）机制。
    # 类属性是由类定义的，而实例属性是由实例创建的。在这种情况下，我们使用类属性来存储一些常量，例如“tokens”和“labels”。
    # 这些类属性被称为“类槽”（class slots），因为它们被放入类中的__slots__元组中。
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        If there are strings in this field that need to be converted into integers through a
        :class:`Vocabulary`, here is where we count them, to determine which tokens are in or out
        of the vocabulary.
        If your `Field` does not have any strings that need to be converted into indices, you do
        not need to implement this method.
        A note on this `counter`: because `Fields` can represent conceptually different things,
        we separate the vocabulary items by `namespaces`.  This way, we can use a single shared
        mechanism to handle all mappings from strings to integers in all fields, while keeping
        words in a `TextField` from sharing the same ids with labels in a `LabelField` (e.g.,
        "entailment" or "contradiction" are labels in an entailment task)
        Additionally, a single `Field` might want to use multiple namespaces - `TextFields` can
        be represented as a combination of word ids and character ids, and you don't want words and
        characters to share the same vocabulary - "a" as a word should get a different id from "a"
        as a character, and the vocabulary sizes of words and characters are very different.
        Because of this, the first key in the `counter` object is a `namespace`, like "tokens",
        "token_characters", "tags", or "labels", and the second key is the actual vocabulary item.
        """
        pass

    # 下面是count_vocab_items()方法的注释。
    # 这个方法是可选的，因为您的字段可能不包含需要转换为整数的字符串。如果您不需要此功能，则可以删除此方法。
    # 如果您的字段包含字符串，则此方法将被调用，以确定词汇表中是否存在这些字符串，以及如果存在，则将它们映射到整数。
    # 这将通过遍历字段中的字符串，并将它们添加到计数器中来完成。计数器是一个字典，其中键是要计数的字符串，值是它们在词汇表中的索引。
    def human_readable_repr(self) -> Any:
        """
        This method should be implemented by subclasses to return a structured, yet human-readable
        representation of the field.
        !!! Note
            `human_readable_repr()` is not meant to be used as a method to serialize a `Field` since the return
            value does not necessarily contain all of the attributes of the `Field` instance. But the object
            returned should be JSON-serializable.
        """
        raise NotImplementedError

    def index(self, vocab: Vocabulary):
        """
        Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
        This `modifies` the `Field` object, it does not return anything.
        If your `Field` does not have any strings that need to be converted into indices, you do
        not need to implement this method.
        """
        pass

    # 下面是index()方法的注释。
    # 这个方法是可选的，因为您的字段可能不包含需要转换为整数的字符串。如果您不需要此功能，则可以删除此方法。
    # 如果您的字段包含字符串，则此方法将被调用，以将字符串转换为整数。这将修改字段本身，而不返回任何内容。
    # 这将通过遍历字段中的字符串，并将它们映射到词汇表中相应的索引来完成。
    def get_padding_lengths(self) -> Dict[str, int]:
        """
        If there are things in this field that need padding, note them here.  In order to pad a
        batch of instance, we get all of the lengths from the batch, take the max, and pad
        everything to that length (or use a pre-specified maximum length).  The return value is a
        dictionary mapping keys to lengths, like `{'num_tokens': 13}`.
        This is always called after :func:`index`.
        """
        raise NotImplementedError

    # 下面是get_padding_lengths()方法的注释。
    # 这个方法是可选的，因为您的字段可能不包含需要填充的东西。如果您不需要此功能，则可以删除此方法。
    # 如果您的字段包含需要填充的东西，则此方法将被调用，以获取需要填充的东西的长度。这将用于为批处理实例进行填充，以便将其转换为张量。
    # 这将返回一个字典，其中键是需要填充的东西，值是它们的长度。
    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        """
        Given a set of specified padding lengths, actually pad the data in this field and return a
        torch Tensor (or a more complex data structure) of the correct shape.  We also take a
        couple of parameters that are important when constructing torch Tensors.
        # Parameters
        padding_lengths : `Dict[str, int]`
            This dictionary will have the same keys that were produced in
            :func:`get_padding_lengths`.  The values specify the lengths to use when padding each
            relevant dimension, aggregated across all instances in a batch.
        """
        raise NotImplementedError

    # 下面是as_tensor()方法的注释。
    # 这个方法是抽象的，因为它需要由子类实现。此方法将根据指定的填充长度填充数据，并返回正确的形状的torch.Tensor（或更复杂的数据结构）。我们还接受一些重要的构造torch.Tensor的参数。
    # 具体来说，此方法将使用指定的填充长度，将数据填充到适当的维度中。然后，它将使用提供的参数将其转换为torch.Tensor。
    # 例如，如果字段是`TextField`，则此方法将使用反向索引器将字符串转换为整数，并将其组合在一起以形成一个批处理的张量。
    def empty_field(self) -> "Field":
        """
        为了让 `ListField` 能够填充列表中的字段数量（例如，答案选项 `TextFields` 的数量），
        我们需要每种类型的空字段的表示。这个方法就是为此返回一个空字段的表示。
        此方法仅在调用 :func:`as_tensor` 时才会被调用，因此您不需要担心在这个空字段上调用 `get_padding_lengths`、`count_vocab_items` 等方法。
        我们将此方法设计为实例方法而不是静态方法，以便如果字段中有任何状态，我们可以复制它（例如，在 `TextField` 中的 token indexers）。
        """
        raise NotImplementedError

    def batch_tensors(self, tensor_list: List[DataArray]) -> DataArray:  # type: ignore
        """
        将 `Field.as_tensor()` 的输出从 `Instances` 列表中合并为一个批处理张量。
        基类中的默认实现处理 `as_tensor` 为每个实例返回单个 torch 张量的情况。如果您的子类返回其他内容，则需要重写此方法。
        此操作不会修改 `self`，但在某些情况下，我们需要 `self` 中包含的信息来执行批处理，因此这是一个实例方法，而不是类方法。
        """
        return torch.stack(tensor_list)

    def __eq__(self, other) -> bool:
        """
        检查两个对象是否相等。首先检查 `other` 是否是当前实例的同一类。
        如果是，它将检查所有通过 `__slots__` 定义的属性是否相等。
        `__slots__` 只包含当前类定义的槽，不包括基类中的槽，因此需要检查所有基类的槽。
        如果某个属性不相等，则返回 False。
        如果子类没有定义为槽类（即没有使用 `__slots__`），则会检查 `__dict__` 是否相等。
        如果所有属性都相等，则返回 True，否则返回 NotImplemented。
        """
        if isinstance(self, other.__class__):
            # 检查所有基类中定义的槽
            for class_ in self.__class__.mro():
                for attr in getattr(class_, "__slots__", []):
                    if getattr(self, attr) != getattr(other, attr):
                        return False
            # 检查是否需要比较 __dict__
            if hasattr(self, "__dict__"):
                return self.__dict__ == other.__dict__
            return True
        return NotImplemented

    def __len__(self):
        raise NotImplementedError

    def duplicate(self):
        return deepcopy(self)
