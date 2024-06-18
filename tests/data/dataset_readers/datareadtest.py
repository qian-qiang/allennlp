from collections import Counter, defaultdict

from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.instance import Instance

# 创建 Fields
tokens = [Token('The'), Token('best'), Token('movie'), Token('ever'), Token('!')]
token_indexers = {'tokens': SingleIdTokenIndexer()}
text_field = TextField(tokens, token_indexers=token_indexers)

label_field = LabelField('pos')

sequence_label_field = SequenceLabelField(
    ['DET', 'ADJ', 'NOUN', 'ADV', 'PUNKT'],
    text_field
)

# 创建 Instance
fields = {
    'tokens': text_field,
    'label': label_field,
}
instance = Instance(fields)

# 为instance增加字段
instance.add_field('label_seq', sequence_label_field)

# 重写类的__str__方法，所以可以通过print获得详细的信息
print(instance)

# 创建vocab
counter = defaultdict(Counter)
instance.count_vocab_items(counter)
vocab = Vocabulary(counter)

# 将所有文本映射成为id
instance.index_fields(vocab)

# 将instance转换为tensor字典，这个方法可以指定padding_lengths，padding_lengths样例如下
# {'tokens': {'tokens___tokens': 5}, 'label': {}, 'label_seq': {'num_tokens': 5}}
tensors = instance.as_tensor_dict()
print(tensors)

# 将token长度改为4
tensors_test = instance.as_tensor_dict(
    padding_lengths={'tokens': {'tokens___tokens': 4}, 'label': {}, 'label_seq': {'num_tokens': 5}})
print(tensors_test)
