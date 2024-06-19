"""
这个文件 allennlp/data/__init__.py 是 AllenNLP 框架中的一个初始化文件，用于导入和组织数据处理相关的各种类和功能。
AllenNLP 是一个基于 PyTorch 的自然语言处理库。这个文件主要作用是从不同的模块中导入数据处理所需的类和函数，
以便于在框架的其他部分中使用这些类和函数。例如,它导入了数据加载器(DataLoader)、
数据集读取器vDatasetReader)、字段(Field)、词汇表(Vocabulary)等组件,这些都是处理、准备和加载NLP数据的基础设施。
"""

from allennlp.data.data_loaders import (
    DataLoader,
    TensorDict,
    allennlp_collate,
)
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, DatasetReaderInput
from allennlp.data.fields.field import DataArray, Field
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data.instance import Instance
from allennlp.data.samplers import BatchSampler
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.image_loader import ImageLoader, TorchImageLoader
