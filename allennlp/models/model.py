"""
`Model` is an abstract class representing
an AllenNLP model.
"""

import logging
import os
from os import PathLike
import re
from typing import Dict, List, Set, Type, Optional, Union

import numpy
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params, remove_keys_from_params
from allennlp.common.registrable import Registrable
from allennlp.data import Instance, Vocabulary
from allennlp.data.batch import Batch
from allennlp.nn import util
from allennlp.nn.regularizers import RegularizerApplicator

logger = logging.getLogger(__name__)

# When training a model, many sets of weights are saved. By default we want to
# save/load this set of weights.
_DEFAULT_WEIGHTS = "best.th"


class Model(torch.nn.Module, Registrable):
    """
    一个抽象类（abstract class），代表一个待训练的模型。这个模型不仅仅依赖于 PyTorch 的 `Module`，而是通过修改 `forward` 方法的输出规范为一个字典来扩展其功能。

    让我们逐段解释这段文档字符串的内容：

    ### 模型特性和功能

    1. **输出为字典**：
       - 该模型的 `forward` 方法的输出被修改为一个字典。这种设计允许模型的输出作为字典传递，可以轻松地解包并传递给其他层次。

    2. **与其他 PyTorch 模型兼容**：
       - 尽管输出是字典，使用这个 API 构建的模型仍然与其他 PyTorch 模型兼容，并且可以作为其他模型内的模块自然地使用。

    3. **使用 AllenNLP 模型时的注意事项**：
       - 如果希望在容器内（如 `nn.Sequential`）使用 AllenNLP 模型，必须在模型之间插入一个包装模块，将字典解包成张量列表。

    4. **训练要求**：
       - 若要使用 [`Trainer`](../training/trainer.md) API 训练模型，模型的输出字典必须包含一个 `"loss"` 键，该键将在训练过程中进行优化。

    5. **指标获取**：
       - 可以选择实现 `get_metrics` 方法，以便使用验证指标进行早停和最佳模型序列化。以 `"_"` 开头的指标不会被 `Trainer` 记录到进度条中。

    6. **从归档中加载模型**： - 该类的 `from_archive` 方法已注册为名称为 `"from_archive"` 的 `Model`。如果使用配置文件，可以指定模型为 `{"type":
    "from_archive", "archive_file": "/path/to/archive.tar.gz"}`，从给定位置提取模型并返回。

    ### 参数

    - **vocab**：`Vocabulary`
      - 在模型中使用词汇表的两个典型用例：在构建嵌入矩阵或输出分类器时获取词汇表大小，以及将模型输出翻译成人类可读的形式。

    - **regularize**：`RegularizeApplicator`, optional
      - 如果提供，`Trainer` 将使用此对象对模型参数进行正则化。

    - **serialization_dir**：`str`, optional
      - 训练输出保存的目录，或者模型加载的目录。

    这段文档字符串提供了关于如何使用和扩展这个抽象模型类的详细说明，以及在整合到 AllenNLP 框架中时需要注意的事项和最佳实践。
    """

    # 作用：用于存储那些在模型的输出字典中无法根据批次大小拆分的键。这个集合用于记录已经发出警告的键，以避免在每次遇到这些键时重复发出警告。
    _warn_for_unseparable_batches: Set[str] = set()

    # 作用：用于指定模型的默认预测器名称。预测器是 AllenNLP 中用于处理模型推理的组件。这个属性可以在需要时被设置为特定的预测器名称。
    default_predictor: Optional[str] = None

    def __init__(
        self,
        vocab: Vocabulary,
        regularizer: RegularizerApplicator = None,
        serialization_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab      # 词表（vocabulary）
        self._regularizer = regularizer     # 正则化器（_regularizer）
        self.serialization_dir = serialization_dir  # 序列化目录（serialization_dir）

    # 这段代码用于计算模型的正则化惩罚。该函数首先检查模型是否配置了正则化器，如果没有配置，则返回None。
    # 否则，它将调用正则化器并检查其返回值。如果正则化器返回的是一个浮点数，并且该浮点数为0，则将其转换为一个张量。
    # 最后，该函数返回正则化惩罚。
    def get_regularization_penalty(self) -> Optional[torch.Tensor]:
        """
        Computes the regularization penalty for the model.
        Returns None if the model was not configured to use regularization.
        """
        if self._regularizer is None:
            regularization_penalty = None
        else:
            try:
                regularization_penalty = self._regularizer(self)
                if isinstance(regularization_penalty, float):
                    assert regularization_penalty == 0.0
                    regularization_penalty = torch.tensor(regularization_penalty)
            except AssertionError:
                raise RuntimeError("The regularizer cannot be a non-zero float.")
        return regularization_penalty

    # 这段代码用于获取用于将模型参数记录为直方图并将其记录到 TensorBoard 的参数名称的列表。该方法遍历模型的命名参数并将其名称添加到列表中。
    def get_parameters_for_histogram_tensorboard_logging(self) -> List[str]:
        """
        Returns the name of model parameters used for logging histograms to tensorboard.
        """
        return [name for name, _ in self.named_parameters()]

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        """
        定义模型的前向传播方法。此外，为了便于训练，该方法设计为计算用户定义的损失函数。

        输入包括执行训练更新所需的一切内容，包括标签 - 你可以在这里定义签名！
        用户需要确保在推理时可以在没有这些标签的情况下进行。因此，任何在推理时不可用的输入
        仅应在条件块内使用。

        该方法的预期草图如下::

            def forward(self, input1, input2, targets=None):
                ....
                ....
                output1 = self.layer1(input1)
                output2 = self.layer2(input2)
                output_dict = {"output1": output1, "output2": output2}
                if targets is not None:
                    # 返回标量 torch.Tensor 的函数，由用户定义。
                    loss = self._compute_loss(output1, output2, targets)
                    output_dict["loss"] = loss
                return output_dict

        # 参数

        *inputs : `Any`
            包含执行训练更新所需的一切内容的张量，包括标签，这些标签应该是可选的（即默认值为 `None`）。
            在推理时，只需传递相关输入，不包括标签。

        # 返回

        output_dict : `Dict[str, torch.Tensor]`
            模型的输出。为了使用 `Trainer` API 训练模型，必须提供一个指向标量 `torch.Tensor` 的 "loss" 键，
            该标量表示要优化的损失。
        """
        raise NotImplementedError

    def forward_on_instance(self, instance: Instance) -> Dict[str, numpy.ndarray]:
        """
        接收一个 [`Instance`](../data/instance.md)，该实例通常包含原始文本，将这些文本
        使用模型的 [`Vocabulary`](../data/vocabulary.md) 转换为数组，通过 `self.forward()`
        和 `self.make_output_human_readable()`（默认情况下不做任何处理）传递这些数组，并返回结果。
        在返回结果之前，我们将任何 `torch.Tensors` 转换为 numpy 数组并移除批次维度。
        """
        return self.forward_on_instances([instance])[0]

    def forward_on_instances(self, instances: List[Instance]) -> List[Dict[str, numpy.ndarray]]:
        """
        接收一个 `Instances` 列表，将这些文本使用模型的 `Vocabulary` 转换为数组，
        通过 `self.forward()` 和 `self.make_output_human_readable()`（默认情况下不做任何处理）
        传递这些数组，并返回结果。在返回结果之前，我们将任何 `torch.Tensors` 转换为 numpy 数组，
        并将批量输出分离为每个实例的单独字典列表。注意，通常这在 GPU 上（有时在 CPU 上）会比重复调用
        `forward_on_instance` 更快。

        # 参数

        instances : `List[Instance]`, required
            要在模型上运行的实例。

        # 返回

        每个实例的模型输出列表。
        """
        batch_size = len(instances)
        with torch.no_grad():
            # 1.获取预测设备，即GPU或CPU。
            cuda_device = self._get_prediction_device()
            # 2.将Instance实例转换为Batch对象，并将其索引化为Vocabulary。
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            # 3.将Batch对象转换为tensor类型，并将其移动到预测设备。
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            # 4.使用模型进行预测，并将输出转换为可读的形式。
            outputs = self.make_output_human_readable(self(**model_input))
            # 5.将输出分离为每个实例的独立输出。
            instance_separated_output: List[Dict[str, numpy.ndarray]] = [
                {} for _ in dataset.instances
            ]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        将 `forward` 的结果转换为人类可读的形式。大多数情况下，这个方法只是将张量中的
        令牌/预测标签转换为人类可以理解的字符串。有时你也会在这里做一个 argmax 操作，
        但这通常发生在 `Model.forward` 中，在你计算指标之前。

        这个方法会 `修改` 输入字典，并且也会 `返回` 相同的字典。

        在基类中，默认情况下我们不做任何处理。
        """

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        返回一个包含指标的字典。这个方法会被 `allennlp.training.Trainer` 调用，
        以计算和使用模型指标进行早停和模型序列化。我们在这里返回一个空字典，而不是抛出异常，
        因为实现新模型的指标并不是必须的。一个布尔类型的 `reset` 参数会被传递，
        因为指标累加器通常会有一些状态需要在各个 epoch 之间重置。
        这也与 [`Metric`](../training/metrics/metric.md) 兼容。
        指标应该在调用 `forward` 时填充，由 `Metric` 处理指标的累加，直到调用此方法。
        """

        return {}

    def _get_prediction_device(self) -> int:
        """
         该方法检查模型参数的设备，以确定用于预测的 cuda_device。
         如果没有参数，则返回 -1。

         # 返回值

         用于预测的 cuda 设备。
             """
        devices = {util.get_device_of(param) for param in self.parameters()}

        if len(devices) > 1:
            devices_string = ", ".join(str(x) for x in devices)
            raise ConfigurationError(f"Parameters have mismatching cuda_devices: {devices_string}")
        elif len(devices) == 1:
            return devices.pop()
        else:
            return -1

    def _maybe_warn_for_unseparable_batches(self, output_key: str):
        """
        如果用户实现的模型返回的字典中包含无法根据批次大小拆分的值，
        该方法会发出一次警告。这个警告由类属性 `_warn_for_unseperable_batches` 控制，
        否则会非常冗长。
        """
        if output_key not in self._warn_for_unseparable_batches:
            logger.warning(
                f"Encountered the {output_key} key in the model's return dictionary which "
                "couldn't be split by the batch size. Key will be ignored."
            )
            # We only want to warn once for this key,
            # so we set this to false so we don't warn again.
            self._warn_for_unseparable_batches.add(output_key)

    # 该方法用于加载已经训练的模型，根据实验配置和一些可选的重写。 具体步骤功能如下
    @classmethod
    def _load(
        cls,
        config: Params,
        serialization_dir: Union[str, PathLike],
        weights_file: Optional[Union[str, PathLike]] = None,
        cuda_device: int = -1,
    ) -> "Model":
        """
        实例化一个已经训练好的模型，基于实验配置和一些可选的重写。
        """
        weights_file = weights_file or os.path.join(serialization_dir, _DEFAULT_WEIGHTS)

        # 1.从文件加载词汇表：
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        # If the config specifies a vocabulary subclass, we need to use it.
        vocab_params = config.get("vocabulary", Params({}))
        vocab_choice = vocab_params.pop_choice("type", Vocabulary.list_available(), True)
        vocab_class, _ = Vocabulary.resolve_class_name(vocab_choice)
        vocab = vocab_class.from_files(
            vocab_dir, vocab_params.get("padding_token"), vocab_params.get("oov_token")
        )

        # 2.从配置中加载模型参数：
        model_params = config.get("model")

        # 3.从文件加载模型权重：
        # 实验配置告诉我们如何训练模型，包括从哪里获取预训练的嵌入/权重。
        # 现在我们正在加载模型，所以这些权重已经存储在我们的模型中。
        # 我们不再需要任何预训练的权重文件或初始化器，并且我们不希望代码去查找它们，
        # 因此我们在这里将它们从参数中移除。
        remove_keys_from_params(model_params)
        model = Model.from_params(
            vocab=vocab, params=model_params, serialization_dir=serialization_dir
        )

        # 4.将模型移动到CPU或GPU，以确保嵌入dings与权重保持一致：
        if cuda_device >= 0:
            model.cuda(cuda_device)
        else:
            model.cpu()

        # 5.如果进行了词汇表和嵌入扩展，则从 from_params 初始化的模型和在 weights_file 中定义的状态字典可能具有不同的嵌入形状。
        # 例如，当模型嵌入模块与词汇表扩展一起传输时，初始化的嵌入权重形状将小于状态字典中的形状。
        # 因此，在调用 load_state_dict 之前需要调用模型嵌入扩展。
        # 如果词汇表和模型嵌入是同步的，以下操作将只是一个空操作。
        model.extend_embedder_vocab()

        # 6.加载状态字典。我们传递 `strict=False` 以避免 PyTorch 在状态字典缺少键时抛出 RuntimeError，
        # 因为我们会在下面处理这种情况。
        model_state = torch.load(weights_file, map_location=util.device_mapping(cuda_device))
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)

        # 7.模块可能会定义一个名为 `authorized_missing_keys` 的类变量，
        # 这是一个正则表达式模式的列表，用于告诉我们忽略与任何模式匹配的缺失键。
        # 有时我们需要这样做，以便使用较新版本的 AllenNLP 加载较旧的模型。
        def filter_out_authorized_missing_keys(module, prefix=""):
            nonlocal missing_keys
            for pat in getattr(module.__class__, "authorized_missing_keys", None) or []:
                missing_keys = [
                    k
                    for k in missing_keys
                    if k.startswith(prefix) and re.search(pat[len(prefix) :], k) is None
                ]
            for name, child in module._modules.items():
                if child is not None:
                    filter_out_authorized_missing_keys(child, prefix + name + ".")

        filter_out_authorized_missing_keys(model)

        # 8.如果出现意外的键或缺少键，则记录错误信息，并引发RuntimeError。
        if unexpected_keys or missing_keys:
            raise RuntimeError(
                f"Error loading state dict for {model.__class__.__name__}\n\t"
                f"Missing keys: {missing_keys}\n\t"
                f"Unexpected keys: {unexpected_keys}"
            )

        return model

    @classmethod
    def load(
        cls,
        config: Params,
        serialization_dir: Union[str, PathLike],
        weights_file: Optional[Union[str, PathLike]] = None,
        cuda_device: int = -1,
    ) -> "Model":
        """
        实例化一个已经训练好的模型，基于实验配置和一些可选的重写。

        # 参数

        config : `Params`
            用于训练模型的配置。它应该包含一个 `model` 部分，并且可能还包含一个 `trainer` 部分。
        serialization_dir: `str = None`
            包含模型的序列化权重、参数和词汇表的目录。
        weights_file: `str = None`
            默认情况下，我们从序列化目录中的 `best.th` 加载权重，但你可以在这里覆盖该值。
        cuda_device: `int = -1`
            默认情况下，我们在 CPU 上加载模型，但如果你想在 GPU 上使用模型，可以在这里指定 GPU 的 ID。

        # 返回

        model : `Model`
            配置中指定的模型，加载了序列化的词汇表和训练好的权重。
        """

        # Peak at the class of the model.
        model_type = (
            config["model"] if isinstance(config["model"], str) else config["model"]["type"]
        )

        # Load using an overridable _load method.
        # This allows subclasses of Model to override _load.

        model_class: Type[Model] = cls.by_name(model_type)  # type: ignore
        if not isinstance(model_class, type):
            # If you're using from_archive to specify your model (e.g., for fine tuning), then you
            # can't currently override the behavior of _load; we just use the default Model._load.
            # If we really need to change this, we would need to implement a recursive
            # get_model_class method, that recurses whenever it finds a from_archive model type.
            model_class = Model
        return model_class._load(config, serialization_dir, weights_file, cuda_device)

    def extend_embedder_vocab(self, embedding_sources_mapping: Dict[str, str] = None) -> None:
        """
        遍历模型中的所有嵌入模块，并确保它们可以使用扩展后的词汇表进行嵌入。
        这在微调或迁移学习场景中是必需的，其中模型使用原始词汇表进行训练，
        但在微调/迁移学习期间，它需要与扩展词汇表（原始词汇表 + 新数据词汇表）一起工作。

        # 参数

        embedding_sources_mapping : `Dict[str, str]`, optional (default = `None`)
            从模型路径到嵌入模块的预训练文件路径的映射。
            如果嵌入初始化时使用的预训练文件现在不可用，用户应传递此映射。
            模型路径是遍历模型属性直到此嵌入模块的路径。
            例如："_text_field_embedder.token_embedder_tokens"。
        """
        # self.named_modules() 返回所有子模块（包括嵌套的子模块）
        # 路径嵌套已经用 "." 分隔：例如 parent_module_name.child_module_name
        embedding_sources_mapping = embedding_sources_mapping or {}
        for model_path, module in self.named_modules():
            if hasattr(module, "extend_vocab"):
                pretrained_file = embedding_sources_mapping.get(model_path)
                module.extend_vocab(
                    self.vocab,
                    extension_pretrained_file=pretrained_file,
                    model_path=model_path,
                )

    @classmethod
    def from_archive(cls, archive_file: str, vocab: Vocabulary = None) -> "Model":
        """
        从归档文件中加载模型。基本上只是调用
        `return archival.load_archive(archive_file).model`。它作为一个方法存在于此，
        方便使用，并且我们可以将其注册，以便从配置文件中轻松微调现有模型。

        如果提供了 `vocab`，我们将使用传递的词汇表对象扩展加载的模型的词汇表
        （包括调用 `extend_embedder_vocab`，它扩展嵌入层）。
        """
        from allennlp.models.archival import load_archive  # here to avoid circular imports

        model = load_archive(archive_file).model
        if vocab:
            model.vocab.extend_from_vocab(vocab)
            model.extend_embedder_vocab()
        return model


# 我们不能用 `Model.register()` 装饰 `Model`，因为 `Model` 还没有定义。所以我们把它放在这里。
Model.register("from_archive", constructor="from_archive")(Model)


def remove_weights_related_keys_from_params(
    params: Params, keys: List[str] = ["pretrained_file", "initializer"]
):
    remove_keys_from_params(params, keys)


def remove_pretrained_embedding_params(params: Params):
    """This function only exists for backwards compatibility.
    Please use `remove_weights_related_keys_from_params()` instead."""
    remove_keys_from_params(params, ["pretrained_file"])
