from typing import Dict, MutableMapping, Mapping

from allennlp.data.fields.field import DataArray, Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.util import JsonDict


class Instance(Mapping[str, Field]):
    """
    `Instance` 是一个包含多个 :class:`~allennlp.data.fields.field.Field` 对象的集合，
    这些对象指定了某个模型的输入和输出。这里我们不区分输入和输出 - 所有操作都在所有字段上进行，
    并且当我们返回数组时，我们将它们作为以字段名为键的字典返回。模型可以决定哪些字段用作输入，哪些用作输出。

    在 `Instance` 中的 `Fields` 可以是已索引的或未索引的。在数据处理流程中，所有字段将被索引，
    之后多个实例可以组合成一个 `Batch`，然后转换成填充数组。

    # 参数

    fields : `Dict[str, Field]`
        将用于为此实例产生数据数组的 `Field` 对象。
    """

    __slots__ = ["fields", "indexed"]

    def __init__(self, fields: MutableMapping[str, Field]) -> None:
        self.fields = fields
        self.indexed = False

    # 添加 `Mapping` 的方法。注意，尽管字段是可变的，
    # 我们没有实现 `MutableMapping`，因为我们希望
    # 你使用 `add_field` 方法并提供一个词汇表。
    def __getitem__(self, key: str) -> Field:
        return self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    # 这段代码用于向现有的字段映射中添加字段
    def add_field(self, field_name: str, field: Field, vocab: Vocabulary = None) -> None:
        """
        Add the field to the existing fields mapping.
        If we have already indexed the Instance, then we also index `field`, so
        it is necessary to supply the vocab.
        """
        self.fields[field_name] = field  # 添加字段到字段映射中
        if self.indexed and vocab is not None:  # 如果我们已经索引了实例，那么我们也需要索引 `field`，因此需要提供词汇表
            field.index(vocab)

    # 这段代码用于遍历所有字段（fields）并对词汇表（vocabulary）进行计数
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        Increments counts in the given `counter` for all of the vocabulary items in all of the
        `Fields` in this `Instance`.
        """
        # 遍历所有字段
        for field in self.fields.values():
            # 对字段中的词汇表进行计数
            field.count_vocab_items(counter)

    # 这段代码用于将实例中的所有字段索引化（index），使用提供的词汇表（vocabulary）
    def index_fields(self, vocab: Vocabulary) -> None:
        """
        Indexes all fields in this `Instance` using the provided `Vocabulary`.
        This `mutates` the current object, it does not return a new `Instance`.
        A `DataLoader` will call this on each pass through a dataset; we use the `indexed`
        flag to make sure that indexing only happens once.
        This means that if for some reason you modify your vocabulary after you've
        indexed your instances, you might get unexpected behavior.
        """
        if not self.indexed:
            for field in self.fields.values():
                field.index(vocab)
            self.indexed = True

    # 这段代码用于获取填充长度，返回一个键为字段名，值为填充长度的字典。下面是加上注释后的代码：
    def get_padding_lengths(self) -> Dict[str, Dict[str, int]]:
        """
        Returns a dictionary of padding lengths, keyed by field name.  Each `Field` returns a
        mapping from padding keys to actual lengths, and we just key that dictionary by field name.
        """
        lengths = {}
        for field_name, field in self.fields.items():
            lengths[field_name] = field.get_padding_lengths()
        return lengths

    # 这段代码用于将 DataArray 实例转换为 tensor 字典，该字典包含了每个字段的 torch.Tensor。下面是加上注释后的代码：
    def as_tensor_dict(
            self, padding_lengths: Dict[str, Dict[str, int]] = None
    ) -> Dict[str, DataArray]:
        """
        Pads each `Field` in this instance to the lengths given in `padding_lengths` (which is
        keyed by field name, then by padding key, the same as the return value in
        :func:`get_padding_lengths`), returning a list of torch tensors for each field.
        If `padding_lengths` is omitted, we will call `self.get_padding_lengths()` to get the
        sizes of the tensors to create.
        """
        # 如果 padding_lengths 不存在，则从实例中获取
        padding_lengths = padding_lengths or self.get_padding_lengths()
        # 创建一个空的 tensor 字典
        tensors = {}
        # 遍历实例中的每个字段
        for field_name, field in self.fields.items():
            tensors[field_name] = field.as_tensor(padding_lengths[field_name])
        # 返回 tensor 字典
        return tensors

    # 这段代码的主要目的是实现实例的字符串表示，即输出实例中包含的字段和字段的值。下面是加上注释后的代码：
    def __str__(self) -> str:
        # 定义一个基本字符串，表示输出的开头
        base_string = "Instance with fields:\n"
        # 使用字符串格式化将实例的字段和字段的值添加到输出字符串中
        return " ".join(
            [base_string] + [f"\t {name}: {field} \n" for name, field in self.fields.items()]
        )

    # 这段代码用于复制实例，创建一个新的实例，并复制原始实例中的所有字段。
    def duplicate(self) -> "Instance":
        new = Instance({k: field.duplicate() for k, field in self.fields.items()})
        new.indexed = self.indexed
        return new

    # 这是一个帮助将实例输出为 JSON 文件或以人类可读方式打印的函数。该函数的用例包括基于示例的解释，在这种情况下，更好地具有输出文件，而不是打印或记录。以下是加上注释后的代码：
    def human_readable_dict(self) -> JsonDict:
        """
           This function help to output instances to json files or print for human readability.
           Use case includes example-based explanation, where it's better to have a output file or
           rather than printing or logging.
           """
        return {key: field.human_readable_repr() for key, field in self.fields.items()}
