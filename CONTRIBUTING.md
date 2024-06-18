# 贡献

感谢您考虑贡献！我们希望AllenNLP成为进行前沿自然语言处理研究的首选工具，但这离不开社区的支持。

## 我可以如何贡献？

### 修复Bug和添加新功能

**发现了Bug？**

首先，请在[这里快速搜索](https://github.com/allenai/allennlp/issues)，看看您的问题是否已经有人报告过。
如果已经有人报告了相同的问题，请在现有问题下评论。

否则，请[创建一个新的GitHub问题](https://github.com/allenai/allennlp/issues)。确保包含清晰的标题和描述。
描述中应包括尽可能多的相关信息。说明如何重现错误行为以及您期望看到的行为。最好包含一个代码示例或可执行的测试案例，演示期望的行为。

**有功能增强建议？**

我们使用GitHub问题跟踪增强请求。在创建增强请求之前：

- 确保您对所需的增强有清晰的想法。如果您的想法模糊，请先在GitHub问题上讨论。
- 检查文档，确保您的功能不存在。
- 快速搜索一下，看看您的增强请求是否已经被提出。

创建增强请求时，请：

- 提供清晰的标题和描述。
- 解释为什么这个增强很有用。可能有助于突出其他库中的类似功能。
- 包含代码示例，演示如何使用增强功能。

### 提交Pull Request

当您准备好贡献代码来解决一个开放的问题时，请遵循以下指南，帮助我们能够快速审查您的Pull Request（PR）。

1. **初始设置**（仅需一次）

    <details><summary>展开详情 👇</summary><br/>

    如果您尚未这样做，请在GitHub上[fork](https://help.github.com/en/enterprise/2.13/user/articles/fork-a-repo)此存储库。

    然后使用以下命令在本地克隆您的fork：

        git clone https://github.com/USERNAME/allennlp.git
    
    或者

        git clone git@github.com:USERNAME/allennlp.git
    
    此时，您的fork本地克隆只知道它来自于*您的*repo，github.com/USERNAME/allennlp.git，但对*主*repo，[https://github.com/allenai/allennlp.git](https://github.com/allenai/allennlp.git)一无所知。您可以通过运行以下命令查看：

        git remote -v
    
    输出结果应如下：

        origin https://github.com/USERNAME/allennlp.git (fetch)
        origin https://github.com/USERNAME/allennlp.git (push)
    
    这意味着您的本地克隆只能跟踪来自您的fork的更改，而不能与主repo保持同步。因此，您需要向克隆添加另一个指向[https://github.com/allenai/allennlp.git](https://github.com/allenai/allennlp.git)的"remote"。运行以下命令：

        git remote add upstream https://github.com/allenai/allennlp.git
    
    现在，如果您再次运行 `git remote -v`，您将看到：

        origin https://github.com/USERNAME/allennlp.git (fetch)
        origin https://github.com/USERNAME/allennlp.git (push)
        upstream https://github.com/allenai/allennlp.git (fetch)
        upstream https://github.com/allenai/allennlp.git (push)

    最后，您需要创建一个适合在AllenNLP上工作的Python 3虚拟环境。有许多工具可帮助管理虚拟环境，但最直接的方法是使用标准库中的[`venv`模块](https://docs.python.org/3.7/library/venv.html)。

    一旦激活您的虚拟环境，您可以通过以下命令以“可编辑模式”安装本地克隆：

        pip install -U pip setuptools wheel
        pip install -e .[dev,all] 

    “可编辑模式”来自于`pip`的 `-e` 参数，实质上只是在您的虚拟环境的site-packages目录与本地克隆中的源代码之间创建了一个符号链接。这样，您所做的任何更改将立即反映在您的虚拟环境中。

    </details>

2. **确保您的fork是最新的**

    <details><summary>展开详情 👇</summary><br/>

    一旦添加了指向[https://github.com/allenai/allennlp.git](https://github.com/allenai/allennlp.git)的"upstream"远程，保持您的fork最新非常简单：

        git checkout main  # 如果尚未在主分支上
        git pull --rebase upstream main
        git push

    </details>

3. **创建新分支以处理您的修复或增强**

    <details><summary>展开详情 👇</summary><br/>

    不建议直接提交到您的fork的主分支。如果您打算进行多次贡献，最好为每个贡献创建一个单独的分支。

    您可以使用以下命令创建新分支：

        # 将BRANCH替换为您想要的任何名称
        git checkout -b BRANCH
        git push -u origin BRANCH

### 测试您的更改

<details><summary>展开详情 👇</summary>&lt;br/&gt;

我们的持续集成（CI）测试在每个Pull Request（PR）上运行[多个检查](https://github.com/allenai/allennlp/actions?query=workflow%3APR)，使用[GitHub Actions](https://github.com/features/actions)。您可以在本地运行大多数这些测试，这对于打开PR之前来说是非常重要的，可以加快审查过程并为我们提供更多便利。

首先，您应该运行[`black`](https://github.com/psf/black)来确保您的代码格式一致。许多集成开发环境支持代码格式化器作为插件，因此您可以设置黑色在每次保存时自动运行。例如，[`black.vim`](https://github.com/psf/black/tree/master/plugin)为Vim提供此功能。但是，您也可以直接从命令行运行`black`。只需从克隆的根目录运行以下命令：

```
black .
```

我们的CI还使用[`flake8`](https://github.com/allenai/allennlp/tree/main/tests)来检查代码库，并使用[`mypy`](http://mypy-lang.org/)进行类型检查。接下来，您应该运行以下两个命令：

```
flake8 .
```

和

```
make typecheck
```

我们还努力保持高测试覆盖率，因此大多数贡献应包括对[unit tests](https://github.com/allenai/allennlp/tree/main/tests)的添加。这些测试使用[`pytest`](https://docs.pytest.org/en/latest/)运行，您可以使用它来本地运行您添加或更改的任何测试模块。

例如，如果您修复了`allennlp/nn/util.py`中的错误，则可以使用以下命令运行特定于该模块的测试：

```
pytest -v tests/nn/util_test.py
```

我们的CI将自动检查测试覆盖率保持在某个阈值以上（大约为90%）。在本例中，您可以运行以下命令检查本地覆盖率：

```
pytest -v --cov allennlp.nn.util tests/nn/util_test.py
```

如果您的贡献涉及对API的任何公共部分的添加，我们要求您为添加的每个函数、方法、类或模块编写文档字符串。

有关语法详细信息，请参阅下面的[编写文档字符串](#编写文档字符串)部分。

您应该测试以确保API文档可以在没有错误的情况下构建：

```
make build-docs
```

如果构建失败，这很可能是由于小的格式问题。如果错误消息不清楚，请随时在您的PR中评论。

您还可以通过以下命令本地提供和查看文档：

```
make serve-docs
```

最后，请确保在[CHANGELOG](https://github.com/allenai/allennlp/blob/main/CHANGELOG.md)中更新"Unreleased"部分，以便记录您的贡献。

当所有上述检查都通过后，您现在可以打开[一个新的GitHub pull request](https://github.com/allenai/allennlp/pulls)。

请确保清楚地描述问题和解决方案，并包含与相关问题的链接。

我们期待审查您的PR！

</details>


### 编写文档字符串

我们的文档字符串基本上是使用Markdown编写的，还包含用于编写参数描述的额外特殊语法。

类的文档字符串应该以类的描述开始，然后是`# Parameters`部分，列出类的`__init__()`方法的所有参数的名称、类型和用途。

参数描述应该如下所示：

```
name : `type`

    参数的描述，缩进四个空格。
```

可选参数也可以这样写：

```
name : `type`, optional (default = `default_value`)

    参数的描述，缩进四个空格。
```

如果参数是不言而喻的，有时可以省略描述。

方法和函数的文档字符串类似，但还应包括在返回值不明显时的`# Returns`部分。其他有效的部分包括：

- `# Attributes`，用于列出类属性。这些应该与参数相同的格式。

- `# Raises`，用于列出函数或方法可能有意引发的任何错误。

- `# Examples`，您可以包含代码片段的地方。

以下是类中文档字符串的示例：

```python
class SentenceClassifier(Model):

    """

    用于分类句子的模型。


    基于[这篇论文](link-to-paper)。输入是一个句子，输出是每个目标标签的得分。


    # Parameters


    vocab : `Vocabulary`


    text_field_embedder : `TextFieldEmbedder`

        将用于创建源标记表示的文本字段嵌入器。


    seq2vec_encoder : `Seq2VeqEncoder`

        此编码器将从`text_field_embedder`获取的嵌入，并将它们编码成表示目标标签未归一化分数的向量。


    dropout : `Optional[float]`, optional (default = `None`)

        可选的dropout，应用于通过`seq2vec_encoder`之前的文本字段嵌入。


    """


    def __init__(

        self,

        vocab: Vocabulary,

        text_field_embedder: TextFieldEmbedder,

        seq2vec_encoder: Seq2SeqEncoder,

        dropout: Optional[float] = None,

    ) -> None:

        pass


    def forward(

        self,

        tokens: TextFieldTensors,

        labels: Optional[Tensor] = None,

    ) -> Dict[str, Tensor]:

        """

        运行模型的前向传播，计算预测的logits，并在提供`labels`时也计算损失。


        # Parameters


        tokens : `TextFieldTensors`

            对应于源序列的标记。


        labels : `Optional[Tensor]`, optional (default = `None`)

            目标标签。


        # Returns


        `Dict[str, Tensor]`

            包含`loss`和`logits`的输出字典。

        """

        pass

```

### 新模型

**您有新的尖端模型吗？**

我们始终在寻找新模型以添加到我们的集合中。最受欢迎的模型通常会被添加到官方的[AllenNLP Models](https://github.com/allenai/allennlp-models)存储库，并在某些情况下添加到[AllenNLP Demo](https://demo.allennlp.org/)中。

如果您认为您的模型应该成为AllenNLP Models的一部分，请在模型存储库中[创建一个pull request](https://github.com/allenai/allennlp-models/pulls)，其中包括：

* 支持您的新模型所需的任何代码更改。
* 模型本身的链接。请不要将模型检入GitHub存储库，而是在PR对话中上传它或提供外部位置的链接。

在PR的描述中，请清楚地解释您的模型执行的任务以及在已建立的数据集上的相关度量。
