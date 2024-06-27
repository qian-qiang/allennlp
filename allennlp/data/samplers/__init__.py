from allennlp.data.samplers.samplers import (
    Sampler,               # 所有采样器的基类。
    BatchSampler,          # 包装另一个采样器以生成小批量的索引。
    SequentialSampler,     # 顺序采样元素，总是以相同的顺序。
    SubsetRandomSampler,   # 从给定的索引列表中随机采样元素。
    WeightedRandomSampler, # 根据给定的概率（权重）从 [0,..,len(weights)-1] 中采样元素。
    RandomSampler,         # 随机采样元素。
    BasicBatchSampler,     # 一个简单的批采样器，将索引分组成批次。
)
from allennlp.data.samplers.bucket_batch_sampler import BucketBatchSampler
# 将实例分组成长度相似的批次，以最小化填充。
from allennlp.data.samplers.max_tokens_batch_sampler import MaxTokensBatchSampler
# 将实例分组成批次，使每个批次中的总令牌数不超过指定的限制。
