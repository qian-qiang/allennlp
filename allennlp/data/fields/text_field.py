"""
A `TextField` represents a string of text, the kind that you might want to represent with
standard word vectors, or pass through an LSTM.
"""
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Iterator
import textwrap


from spacy.tokens import Token as SpacyToken
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util

# 这里有两层字典：顶层是*key*，它将TokenIndexers与其对应的TokenEmbedders对齐。
# 底层是由给定TokenIndexer生成的*objects*，这些对象将作为输入传递给特定TokenEmbedder的forward()方法。
# 我们将这些对象标记为张量，因为它们通常是张量，尽管实际上它们可以具有任意类型。
TextFieldTensors = Dict[str, Dict[str, torch.Tensor]]


class TextField(SequenceField[TextFieldTensors]):
    """
    此 `Field` 表示一个字符串令牌列表。在构建此对象之前，您需要使用 :class:`~allennlp.data.tokenizers.tokenizer.Tokenizer` 对原始字符串进行分词。

    由于字符串令牌可以通过多种方式表示为索引数组，我们还使用了一个 :class:`~allennlp.data.token_indexers.token_indexer.TokenIndexer` 对象的字典，这些对象将用于将令牌转换为索引。
    每个 `TokenIndexer` 可以将每个令牌表示为单个 ID,或者字符 ID 的列表，或者其他形式。

    这个字段将被转换为一个数组的字典，每个 `TokenIndexer` 对应一个。`SingleIdTokenIndexer` 生成的数组形状为 (num_tokens,)，而 `TokenCharactersIndexer` 生成的数组形状为 (num_tokens, num_characters)。
    """

    __slots__ = ["tokens", "_token_indexers", "_indexed_tokens"]
    # tokens: 表示文本的 Token 对象列表。
    # _token_indexers: 字典，将索引器名称映射到用于索引 tokens 的 TokenIndexer 实例。
    # _indexed_tokens: 字典，存储索引 tokens 的结果，通常用于模型输入。

    def __init__(
        self, tokens: List[Token], token_indexers: Optional[Dict[str, TokenIndexer]] = None
    ) -> None:
        self.tokens = tokens
        self._token_indexers = token_indexers
        self._indexed_tokens: Optional[Dict[str, IndexedTokenList]] = None

        if not all(isinstance(x, (Token, SpacyToken)) for x in tokens):
            raise ConfigurationError(
                "TextFields must be passed Tokens. "
                "Found: {} with types {}.".format(tokens, [type(x) for x in tokens])
            )

    # 这个@property装饰器用于将方法转换为属性，使得可以通过属性的方式访问方法返回的值，而不需要在方法名后加括号。
    @property
    # 这段代码用于获取 TokenIndexer 对象，它是一个将文本转换为整数序列的工具。
    def token_indexers(self) -> Dict[str, TokenIndexer]:
        if self._token_indexers is None:
            raise ValueError(
                "TextField's token_indexers have not been set.\n"
                "Did you forget to call DatasetReader.apply_token_indexers(instance) "
                "on your instance?\n"
                "If apply_token_indexers() is being called but "
                "you're still seeing this error, it may not be implemented correctly."
            )
        return self._token_indexers


    # 这个装饰器定义了一个名为 token_indexers 的属性的 setter 方法，允许设置 _token_indexers 字典。
    @token_indexers.setter
    def token_indexers(self, token_indexers: Dict[str, TokenIndexer]) -> None:
        self._token_indexers = token_indexers

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for indexer in self.token_indexers.values():
            for token in self.tokens:
                indexer.count_vocab_items(token, counter)

    # 这段代码用于将文本序列（tokens）转换为索引序列（indexed_tokens），以便在神经网络模型中使用。
    def index(self, vocab: Vocabulary):
        # 定义一个字典，用于存储不同索引器（indexer）生成的索引序列
        self._indexed_tokens = {}
        # 遍历所有索引器
        for indexer_name, indexer in self.token_indexers.items():
            # 使用当前索引器将文本序列转换为索引序列
            self._indexed_tokens[indexer_name] = indexer.tokens_to_indices(self.tokens, vocab)

    # 这段代码用于获取 `TextField` 的填充长度（padding lengths）。`TextField` 有一系列 `Tokens`，每个 `Token` 都会通过多个 `TokenIndexers` 被转换为数组。该方法获取每个数组的最大长度（over tokens）。
    def get_padding_lengths(self) -> Dict[str, int]:
           """
           The `TextField` has a list of `Tokens`, and each `Token` gets converted into arrays by
           (potentially) several `TokenIndexers`.  This method gets the max length (over tokens)
           associated with each of these arrays.
           """
           if self._indexed_tokens is None:
               raise ConfigurationError(
                   "You must call .index(vocabulary) on a field before determining padding lengths."
               )
           # 首先检查是否已经进行了索引（indexed）
           padding_lengths = {}
           for indexer_name, indexer in self.token_indexers.items():
               # 遍历每个索引器
               indexer_lengths = indexer.get_padding_lengths(self._indexed_tokens[indexer_name])
               # 遍历每个索引器的长度
               for key, length in indexer_lengths.items():
                   # 按照格式“索引器名称___键”将长度添加到 padding_lengths 中
                   padding_lengths[f"{indexer_name}___{key}"] = length
           return padding_lengths


    def sequence_length(self) -> int:
        return len(self.tokens)

    def as_tensor(self, padding_lengths: Dict[str, int]) -> TextFieldTensors:
        if self._indexed_tokens is None:
            raise ConfigurationError(
                "You must call .index(vocabulary) on a field before calling .as_tensor()"
            )

        tensors = {}

        indexer_lengths: Dict[str, Dict[str, int]] = defaultdict(dict)
        for key, value in padding_lengths.items():
            # We want this to crash if the split fails. Should never happen, so I'm not
            # putting in a check, but if you fail on this line, open a github issue.
            indexer_name, padding_key = key.split("___")
            indexer_lengths[indexer_name][padding_key] = value

        for indexer_name, indexer in self.token_indexers.items():
            tensors[indexer_name] = indexer.as_padded_tensor_dict(
                self._indexed_tokens[indexer_name], indexer_lengths[indexer_name]
            )
        return tensors

    def empty_field(self):
        text_field = TextField([], self._token_indexers)
        text_field._indexed_tokens = {}
        if self._token_indexers is not None:
            for indexer_name, indexer in self.token_indexers.items():
                text_field._indexed_tokens[indexer_name] = indexer.get_empty_token_list()
        return text_field

    def batch_tensors(self, tensor_list: List[TextFieldTensors]) -> TextFieldTensors:
        # This is creating a dict of {token_indexer_name: {token_indexer_outputs: batched_tensor}}
        # for each token indexer used to index this field.
        indexer_lists: Dict[str, List[Dict[str, torch.Tensor]]] = defaultdict(list)
        for tensor_dict in tensor_list:
            for indexer_name, indexer_output in tensor_dict.items():
                indexer_lists[indexer_name].append(indexer_output)
        batched_tensors = {
            # NOTE(mattg): if an indexer has its own nested structure, rather than one tensor per
            # argument, then this will break.  If that ever happens, we should move this to an
            # `indexer.batch_tensors` method, with this logic as the default implementation in the
            # base class.
            indexer_name: util.batch_tensor_dicts(indexer_outputs)
            for indexer_name, indexer_outputs in indexer_lists.items()
        }
        return batched_tensors

    def __str__(self) -> str:
        # Double tab to indent under the header.
        formatted_text = "".join(
            "\t\t" + text + "\n" for text in textwrap.wrap(repr(self.tokens), 100)
        )
        if self._token_indexers is not None:
            indexers = {
                name: indexer.__class__.__name__ for name, indexer in self._token_indexers.items()
            }
            return (
                f"TextField of length {self.sequence_length()} with "
                f"text: \n {formatted_text} \t\tand TokenIndexers : {indexers}"
            )
        else:
            return f"TextField of length {self.sequence_length()} with text: \n {formatted_text}"

    # Sequence[Token] methods
    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __getitem__(self, idx: int) -> Token:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)

    def duplicate(self):
        """
        Overrides the behavior of `duplicate` so that `self._token_indexers` won't
        actually be deep-copied.

        Not only would it be extremely inefficient to deep-copy the token indexers,
        but it also fails in many cases since some tokenizers (like those used in
        the 'transformers' lib) cannot actually be deep-copied.
        """
        if self._token_indexers is not None:
            new = TextField(deepcopy(self.tokens), {k: v for k, v in self._token_indexers.items()})
        else:
            new = TextField(deepcopy(self.tokens))
        new._indexed_tokens = deepcopy(self._indexed_tokens)
        return new

    def human_readable_repr(self) -> List[str]:
        return [str(t) for t in self.tokens]
