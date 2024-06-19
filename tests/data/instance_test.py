import numpy
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, TensorField
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token


class TestInstance(AllenNlpTestCase):
    """这段代码用于测试实例是否实现了可变映射"""

    def test_instance_implements_mutable_mapping(self):
        """创建一个包含两个字段（文本字段和标签字段）的实例"""
        words_field = TextField([Token("hello")], {})
        label_field = LabelField(1, skip_indexing=True)
        instance = Instance({"words": words_field, "labels": label_field})

        """使用方括号语法访问实例的两个字段，并检查其值"""
        assert instance["words"] == words_field
        assert instance["labels"] == label_field
        assert len(instance) == 2

        """检查实例的键和值是否正确"""
        keys = {k for k, v in instance.items()}
        assert keys == {"words", "labels"}

        values = [v for k, v in instance.items()]
        assert words_field in values
        assert label_field in values

    # 这段代码用于测试 duplicate() 方法是否与 PretrainedTransformerIndexer 结合在 TextField 中工作。
    # 请参阅 https://github.com/allenai/allennlp/issues/4270。
    def test_duplicate(self):
        # Verify the `duplicate()` method works with a `PretrainedTransformerIndexer` in
        # a `TextField`. See https://github.com/allenai/allennlp/issues/4270.
        instance = Instance(
            {
                "words": TextField(
                    [Token("hello")], {"tokens": PretrainedTransformerIndexer("bert-base-uncased")}
                )
            }
        )

        other = instance.duplicate()
        assert other == instance
        # Adding new fields to the original instance should not affect the duplicate.
        instance.add_field("labels", LabelField("some_label"))
        assert "labels" not in other.fields
        assert other != instance  # sanity check on the '__eq__' method.

    # 这段代码用于测试 Instance 类的 human_readable_dict() 方法，该方法可将 Instance 对象转换为易于阅读的字典。
    def test_human_readable_repr(self):
        # 创建一个包含单词“hello”的文本字段和一个标签字段
        words_field = TextField([Token("hello")], {})
        label_field = LabelField(1, skip_indexing=True)
        # 创建一个实例
        instance1 = Instance({"words": words_field, "labels": label_field})
        # 使用 human_readable_dict() 方法将实例转换为易于阅读的字典
        assert type(instance1.human_readable_dict()) is dict
        assert instance1.human_readable_dict() == {"words": ["hello"], "labels": 1}
        # 接着，我们将实例转换为易于阅读的字典，并将其存储在变量 instance1_human_readable_dict 中。
        # 然后，我们将一个形状为 [3] 的张量字段添加到实例中，并使用 human_readable_dict() 方法将其转换为易于阅读的字典。
        # 最后，我们检查 instance1_human_readable_dict 和 instance2.human_readable_dict() 的结果是否相同。
        # 这段代码展示了如何使用 human_readable_dict() 方法将 Instance 对象转换为易于阅读的字典，并将其与原始实例进行比较。
        instance1_human_readable_dict = instance1.human_readable_dict()
        array = TensorField(numpy.asarray([1.0, 1, 1]))
        array_human_readable_dict = {
            "shape": [3],
            "element_mean": 1.0,
            "element_std": 0,
            "type": "float64",
        }
        instance2 = Instance({"words": words_field, "labels": label_field, "tensor": array})
        instance1_human_readable_dict["tensor"] = array_human_readable_dict
        assert instance1_human_readable_dict == instance2.human_readable_dict()


"""
Q:TextField入参是什么：

A:TextField类的构造函数接受两个参数：

1. tokens: 这是一个 `List[Token]` 类型的列表，代表分词后的文本数据。
2. token_indexers这是一个可选的 `Optional[Dict[str, TokenIndexer]]` 类型的字典，用于将 tokens 转换为索引。
每个 `TokenIndexer` 可以以不同的方式表示 token，例如单个 ID 或字符 ID 列表等。

构造函数的定义如下：
```python
def __init__(
    self, tokens: List[Token], token_indexers: Optional[Dict[str, TokenIndexer]] = None
) -> None:
    ...
```
在这个构造函数中，还包括了对 `tokens` 中每个元素是否为 `Token` 或 `SpacyToken` 类型的检查。如果不是，将抛出 `ConfigurationError`。


Q:LabelField入参是什么：

A:`LabelField` 类用于处理标签或分类输出，如在分类任务中的目标标签。它通常用于模型的监督学习，以指定每个样本的正确输出。

### `LabelField` 的构造函数接受以下参数：

1. label: 这是标签的值，可以是字符串（如在文本分类任务中的类别名）或整数（如在某些类型的分类任务中的类别索引）。

2. label_namespace` (`str`, 可选): 这个参数用于指定标签所属的命名空间。在使用 `Vocabulary` 对象管理不同类型的标签时，命名空间可以帮助区分不同类型的标签集合。默认值通常是 `'labels'`。

3. skip_indexing` (`bool`, 可选): 如果设置为 `True`，则不会对标签进行索引操作，这通常用于标签已经是整数形式的情况。默认值是 `False`，即默认对字符串标签进行索引。



`TextField` 和 `LabelField` 是 AllenNLP 中用于处理不同类型数据的类：

### TextField
- **用途**: TextField主要用于处理输入文本数据。它接受一系列的 Token对象，这些对象代表了分词后的文本。
- **功能**: 它与 TokenIndexer结合使用，将文本转换为模型可以处理的数值数据（如词索引或字符索引）。这使得TextField能够支持多种不同的文本表示方法，适应不同的模型需求。
- **数据类型**: 接受 `List[Token]`，即一系列经过分词的标记。
- **索引**: 需要通过 TokenIndexer将文本转换为数值索引。

### LabelField
- **用途**: `LabelField` 用于处理输出标签或分类结果，常见于分类任务中。它用于指定每个样本的目标输出，如文本的类别标签。
- **功能**: 它可以处理字符串或整数形式的标签，并且可以选择是否对这些标签进行索引。如果标签是字符串，通常会被索引为整数，以便模型处理。
- **数据类型**: 接受字符串或整数，表示标签。
- **索引**: 可以选择是否对标签进行索引，取决于 `skip_indexing` 参数的设置。

### 主要区别
- **目的和用途**: TextField用于输入处理，而 `LabelField` 用于输出标签处理。
- **数据处理**: TextField需要复杂的索引机制来处理文本数据，而 `LabelField` 的处理相对简单，主要是标签的索引（如果需要）。
- **在模型中的角色**: TextField通常用于模型的输入层，处理特征数据；`LabelField` 用于监督学习的目标输出，如分类任务中的类别标签。

这两个字段类型在自然语言处理框架中是互补的，TextField处理输入数据，`LabelField` 处理输出数据，共同支持模型的训练和预测过程。


Q:Instance入参是什么：

A:Instance类的构造函数接受以下入参：
类型为 `MutableMapping[str, Field]`，这是一个映射，其中键是字符串（字段名），值是 `Field` 对象。这些 `Field` 对象用于为此实例生成数据数组。
"""
