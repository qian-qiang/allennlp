# AllenNLP 风格指南

我们的代码风格的最高优先级是让新加入代码库的人能够轻松阅读代码。深度学习容易出错，我们希望代码易于阅读，
让查看代码的人能够关注我们的建模决策，而不是试图理解代码的运行方式。

为此，我们使用描述性命名、类型注释和连贯的文档字符串。在处理张量的代码中，大多数计算张量的行都有描述张
量形状的注释。当代码中有有趣或重要的建模决策时，我们会写注释（可以是行内注释或适当的文档字符串）。

## 文档字符串

所有合理复杂的公共方法都应有文档字符串，描述其基本功能、输入和输出。私有方法也通常应该有文档字符串，
以便阅读代码的人知道该方法的用途。我们使用的文档字符串基本结构是：
(1) 简要描述方法的作用，有时还包括方法的实现方式或原因，
(2) 方法的参数/参数，
(3) 方法的返回值（如果有）。
如果方法特别简单或参数显而易见，可以省略 (2) 和 (3)。我们的文档使用 Markdown 格式，因此函数参数和返回
值应格式化为 Markdown 标题（例如 `# Parameters`），
在代码库中的几乎所有模型或模块中都能看到这种格式。我们将类文档字符串视为 `__init__` 方法的文档，
在此处给出参数并省略构造函数本身的文档字符串。对于模型/模块构造函数和类似 `forward` 的方法，
_始终_ 在文档字符串中包括参数和返回值（如果有）。

## 代码格式

我们使用 `flake8`、`black` 和 `mypy` 来强制实现格式的一致性。这些格式指南大致遵循
[Google 的 Python 风格指南](https://google.github.io/styleguide/pyguide.html#Python_Style_Rules)，
但有一些显著的例外。特别是因为我们使用类型注释和描述性变量名，所以我们使用 100 字符的行而不是 80 字符的行，
有时代码行长度可以适当超出。另外，我们使用 `mkdocs` 来构建文档，因此不适用 Google 的文档字符串格式。

## 命名

我们遵循 Google 的[通用命名规则](https://google.github.io/styleguide/cppguide.html#General_Naming_Rules)
和他们的[驼峰式命名法定义](https://google.github.io/styleguide/javaguide.html#s5.3-camel-case)。

## 模块布局和导入

为了防止文件过大，我们通常每个文件一个类，但与伴随类不可分割的小类也可以放在同一文件中（通常这些类是私有类）。

为了避免导入类时的冗长，当类以这种方式组织时，应该从其模块的 `__init__.py` 中导入。
例如，`Batch` 类在 `allennlp/data/batch.py` 中，但 `allennlp/data/__init__.py` 导入了该类，
因此你可以直接 `from allennlp.data import Batch`。

抽象类通常放在包含抽象类和所有内置实现的模块中。这包括 `Field`（在 `allennlp.data.fields` 中）、
`Seq2SeqEncoder`（在 `allennlp.modules.seq2seq_encoders` 中）以及许多其他类。在这些情况下，
抽象类应导入到上一级模块，以便可以 `from allennlp.data import Field`。
具体实现遵循上述相同的布局：`from allennlp.data.fields import TextField`。

导入应格式化在文件顶部，遵循[PEP 8 的建议](https://www.python.org/dev/peps/pep-0008/#imports)：
三个部分（标准库、第三方库、内部导入），每个部分排序并用空行分隔。

## 结论

我们采用的一些约定是任意的（例如，其他驼峰式命名法定义也有效），但我们坚持这些约定以保持代码库的一致风格，从而使代码更易于阅读和维护。