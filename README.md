一个基于 PyTorch 构建的 Apache 2.0 NLP 研究库，用于开发各种语言任务的最先进的深度学习模型。

⚠️ **注意：** AllenNLP 现在处于维护模式。这意味着我们不再添加新功能或升级依赖项。直到 2022 年 12 月 16 日，我们仍会响应问题并处理错误。如果您有任何疑问或有兴趣维护 AllenNLP，请在此存储库上提出问题。

AllenNLP 取得了巨大成功，但是由于领域进展迅速，现在是专注于新项目的时候了。我们正在努力使 [AI2 Tango](https://github.com/allenai/tango) 成为组织研究代码库的最佳方式。如果您是 AllenNLP 的积极用户，这里有一些建议的替代方案：
* 如果您喜欢训练器、配置语言或者仅仅是寻找更好的实验管理方式，请查看 [AI2 Tango](https://github.com/allenai/tango)。
* 如果您喜欢 AllenNLP 的 `modules` 和 `nn` 包，请查看 [delmaksym/allennlp-light](https://github.com/delmaksym/allennlp-light)。它甚至与 [AI2 Tango](https://github.com/allenai/tango) 兼容！
* 如果您喜欢 AllenNLP 的框架方面，请查看 [flair](https://github.com/flairNLP/flair)。它拥有多个最先进的 NLP 模型，并且允许您轻松使用如 Transformers 的预训练嵌入。
* 如果您喜欢 AllenNLP 的度量包，请查看 [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)。它具有与 AllenNLP 相同的 API，因此切换应该是一个快速的学习过程。
* 如果您想要对文本进行向量化，请尝试 [transformers library](https://github.com/huggingface/transformers)。
* 如果您想要维护 AllenNLP 的公平性或解释组件，请联系我们。目前还没有替代方案，因此我们正在寻找专门的维护者。
* 如果您关注其他 AllenNLP 功能，请创建一个问题。也许我们可以找到另一种方式来继续支持您的用例。

## 快速链接

- ↗️ [网站](https://allennlp.org/)
- 🔦 [指南](https://guide.allennlp.org/)
- 🖼 [画廊](https://gallery.allennlp.org)
- 💻 [演示](https://demo.allennlp.org)
- 📓 [文档](https://docs.allennlp.org/) ( [最新](https://docs.allennlp.org/latest/) | [稳定](https://docs.allennlp.org/stable/) | [提交](https://docs.allennlp.org/main/) )
- ⬆️ [从 1.x 升级到 2.0 的指南](https://github.com/allenai/allennlp/discussions/4933)
- ❓ [Stack Overflow](https://stackoverflow.com/questions/tagged/allennlp)
- ✋ [贡献指南](CONTRIBUTING.md)
- 🤖 [官方支持的模型](https://github.com/allenai/allennlp-models)
    - [预训练模型](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/pretrained.py)
    - [文档](https://docs.allennlp.org/models/) ( [最新](https://docs.allennlp.org/models/latest/) | [稳定](https://docs.allennlp.org/models/stable/) | [提交](https://docs.allennlp.org/models/main/) )
- ⚙️ [持续构建](https://github.com/allenai/allennlp/actions)
- 🌙 [每夜版发布](https://pypi.org/project/allennlp/#history)

## 在此 README 中

- [开始使用库](#getting-started-using-the-library)
- [插件](#plugins)
- [包概述](#package-overview)
- [安装](#installation)
    - [通过 pip 安装](#installing-via-pip)
    - [使用 Docker 安装](#installing-using-docker)
    - [从源代码安装](#installing-from-source)
- [运行 AllenNLP](#running-allennlp)
- [问题](#issues)
- [贡献](#contributions)
- [引用](#citing)
- [团队](#team)

## 开始使用库

如果您对使用 AllenNLP 进行模型开发感兴趣，我们建议您查看 [AllenNLP 指南](https://guide.allennlp.org) 以深入了解库，然后查看我们更高级的指南，例如在 [GitHub 讨论](https://github.com/allenai/allennlp/discussions/categories/guides) 上。

当您准备开始项目时，我们创建了几个模板存储库，供您作为起点使用：

* 如果您想使用 `allennlp train` 和配置文件来指定实验，请使用 [此模板](https://github.com/allenai/allennlp-template-config-files)。我们推荐使用这种方法。
* 如果您更喜欢使用 Python 代码配置实验并运行训练循环，请使用 [此模板](https://github.com/allenai/allennlp-template-python-script)。在这种设置中，有些事情目前稍微麻烦一些（如加载保存的模型和使用分布式训练），但除此之外，功能与配置文件设置等效。

此外，还有一些外部教程：

* [使用 Optuna 对 AllenNLP 进行超参数优化](https://medium.com/optuna/hyperparameter-optimization-for-allennlp-using-optuna-54b4bfecd78b)
* [在 AllenNLP 中使用多个 GPU 进行训练](https://medium.com/ai2-blog/tutorial-how-to-train-with-multiple-gpus-in-allennlp-c4d7c17eb6d6)
* [在 AllenNLP 中使用更大批次进行训练，内存占用更少](https://medium.com/ai2-blog/tutorial-training-on-larger-batches-with-less-memory-in-allennlp-1cd2047d92ad)
* [如何将 AllenNLP 中的转换器权重和分词器上传到 HuggingFace](https://medium.com/ai2-blog/tutorial-how-to-upload-transformer-weights-and-tokenizers-from-allennlp-to-huggingface-ecf6c0249bf)

以及其他在 [AI2 AllenNLP 博客](https://medium.com/ai2-blog/allennlp/home) 上的教程。
## 插件

AllenNLP 支持动态加载 "插件"。插件就是提供自定义注册类或额外 `allennlp` 子命令的 Python 包。

有一个开源插件生态系统，其中一些由 AI2 团队在此维护，另一些由更广泛的社区维护。

"https://github.com/allenai/allennlp-models"一系列最先进的模型
"https://github.com/allenai/allennlp-semparse"用于构建语义解析器的框架
"https://github.com/allenai/allennlp-server"用于提供模型服务的简单演示服务器
"https://github.com/himkt/allennlp-optuna"
"https://himkt.github.io/profile/" 集成了"https://optuna.org/"的超参数优化


AllenNLP 将自动发现您已安装的所有官方 AI2 维护的插件，但为了让 AllenNLP 发现您安装的个人或第三方插件，
您还需要在运行 `allennlp` 命令的目录中创建一个名为 `.allennlp_plugins` 的本地插件文件，
或者在 `~/.allennlp/plugins` 处创建一个全局插件文件。该文件应列出您希望加载的插件模块，每行一个。

要测试 AllenNLP 是否可以找到并导入您的插件，您可以运行 `allennlp test-install` 命令。
每个发现的插件都将记录到终端上。

有关插件的更多信息，请参阅 [插件 API 文档](https://docs.allennlp.org/main/api/common/plugins/)。关于如何创建一个自定义子命令以分发为插件的信息，请参阅 [子命令 API 文档](https://docs.allennlp.org/main/api/commands/subcommand/)。

## 包概述

allennlp 基于 PyTorch 的开源 NLP 研究库
allennlp.commands CLI 功能 
allennlp.common  在整个库中使用的实用模块
allennlp.data 用于加载数据集并将字符串编码为矩阵中的整数表示的数据处理模块
allennlp.fairness  用于偏差缓解和公平性算法及指标的模块
allennlp.modules 用于文本的一组 PyTorch 模块 
allennlp.nn 张量实用函数，如初始化器和激活函数 
allennlp.training  用于训练模型的功能 

## 安装

AllenNLP 需要 Python 3.6.1 或更高版本以及 [PyTorch](https://pytorch.org/)。

我们支持在 Mac 和 Linux 环境下使用 AllenNLP。目前我们不支持 Windows，但欢迎贡献。

### 通过 conda-forge 安装

安装 AllenNLP 的最简单方法是使用 conda（您可以选择不同的 Python 版本）：

```
conda install -c conda-forge python=3.8 allennlp
```

要安装可选包，如 `checklist`，请使用：

```
conda install -c conda-forge allennlp-checklist
```

或者直接安装 `allennlp-all`。上面提到的插件同样可以安装，例如：

```
conda install -c conda-forge allennlp-models allennlp-semparse allennlp-server allennlp-optuna
```
### 使用 pip 安装

建议在安装 AllenNLP 之前先安装 PyTorch 生态系统，具体步骤请参考 [pytorch.org](https://pytorch.org/)。

安装完成后，只需运行以下命令来安装 AllenNLP：

```bash
pip install allennlp
```

> ⚠️ 如果您使用的是 Python 3.7 或更高版本，请确保在运行上述命令后不要安装了 PyPI 版本的 `dataclasses`，因为这可能会在某些平台上造成问题。您可以通过运行 `pip freeze | grep dataclasses` 快速检查。如果输出中看到类似 `dataclasses=0.6` 的内容，请运行 `pip uninstall -y dataclasses` 卸载它。

如果您需要建立合适的 Python 环境或者希望使用不同的安装方法，请参考下面的说明。

#### 设置虚拟环境

您可以使用 Conda 来设置一个包含所需 Python 版本的虚拟环境，用于安装 AllenNLP。如果您已经有一个想要使用的 Python 3 环境，可以直接跳到“通过 pip 安装”部分。

1. [下载并安装 Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)。

2. 创建一个包含 Python 3.8（也可以是 3.7 或 3.9）的 Conda 环境：

    ```bash
    conda create -n allennlp_env python=3.8
    ```

3. 激活 Conda 环境。您需要在每个希望使用 AllenNLP 的终端中激活 Conda 环境：

    ```bash
    conda activate allennlp_env
    ```

#### 安装库和依赖项

使用 `pip` 简单地安装库和依赖项：

```bash
pip install allennlp
```

要安装可选依赖项，如 `checklist`，运行：

```bash
pip install allennlp[checklist]
```

或者您也可以直接安装所有可选依赖项：

```bash
pip install allennlp[all]
```

*想要尝试最新的功能？您可以直接从 [pypi](https://pypi.org/project/allennlp/#history) 安装夜间版本。*

安装 AllenNLP 后，会在安装 Python 包时安装一个脚本，这样您就可以在终端中直接输入 `allennlp` 运行 AllenNLP 命令。例如，您现在可以使用 `allennlp test-install` 测试您的安装。

您可能还想安装 `allennlp-models`，其中包含了用于训练和运行我们官方支持的模型的 NLP 构件，其中许多模型托管在 [https://demo.allennlp.org](https://demo.allennlp.org)。

```bash
pip install allennlp-models
```

### 使用 Docker 安装

Docker 提供了一个虚拟机，其中已经安装了所有 AllenNLP 的依赖项，无论您是使用 GPU 还是 CPU 运行。Docker 提供更好的隔离性和一致性，并且可以轻松地将您的环境分发到计算集群中。

AllenNLP 提供了[官方 Docker 镜像](https://hub.docker.com/r/allennlp/allennlp)，已安装了库和所有依赖项。

安装 Docker 后，如果您有 GPU 可用，还应该安装 [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)。

然后运行以下命令以获取一个能在 GPU 上运行的环境：

```bash
mkdir -p $HOME/.allennlp/
docker run --rm --gpus all -v $HOME/.allennlp:/root/.allennlp allennlp/allennlp:latest
```

您可以使用以下命令测试 Docker 环境：

```bash
docker run --rm --gpus all -v $HOME/.allennlp:/root/.allennlp allennlp/allennlp:latest test-install 
```

如果没有 GPU 可用，只需省略 `--gpus all` 标志即可。

#### 构建自己的 Docker 镜像

出于各种原因，您可能需要创建自己的 AllenNLP Docker 镜像，比如需要不同版本的 PyTorch。只需从 AllenNLP 的本地克隆的根目录运行 `make docker-image` 即可。

默认情况下，这将使用 `allennlp/allennlp` 标签构建一个镜像，但您可以通过设置 `DOCKER_IMAGE_NAME` 标志来更改名称。例如，`make docker-image DOCKER_IMAGE_NAME=my-allennlp`。

如果您想使用不同版本的 Python 或 PyTorch，请将 `DOCKER_PYTHON_VERSION` 和 `DOCKER_TORCH_VERSION` 标志设置为类似 `3.9` 和 `1.9.0-cuda10.2` 的值。这些标志共同决定所使用的基础镜像。您可以在 GitHub Container Registry 上查看有效组合的列表：[github.com/allenai/docker-images/pkgs/container/pytorch](https://github.com/allenai/docker-images/pkgs/container/pytorch)。

构建完成后，您可以运行 `docker images allennlp` 查看已构建的镜像列表。

```
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
allennlp/allennlp   latest              b66aee6cb593        5 minutes ago       2.38GB
```

### 从源代码安装

您还可以通过克隆我们的 git 仓库来安装 AllenNLP：

```bash
git clone https://github.com/allenai/allennlp.git
```

创建一个 Python 3.7 或 3.8 的虚拟环境，并通过以下命令以 `editable` 模式安装 AllenNLP：

```bash
pip install -U pip setuptools wheel
pip install --editable .[dev,all]
```

这样会在系统中使 `allennlp` 可用，但会使用您克隆的本地源代码。

您可以使用 `allennlp test-install` 测试您的安装。有关从源代码安装 `allennlp-models` 的说明，请参阅 [https://github.com/allenai/allennlp-models](https://github.com/allenai/allennlp-models)。

## 运行 AllenNLP

安装完 AllenNLP 后，您可以使用 `allennlp` 命令运行命令行界面（无论是从 `pip` 还是从源代码安装）。`allennlp` 有各种子命令，如 `train`、`evaluate` 和 `predict`。要查看完整的使用信息，请运行 `allennlp --help`。

您可以通过运行 `allennlp test-install` 测试您的安装。

## 问题

欢迎所有人提交问题，无论是功能请求、错误报告还是一般问题。作为一个拥有自己内部目标的小团队，如果即时修复不符合我们的路线图，我们可能会要求贡献。为了保持整洁，我们通常会关闭我们认为已回答的问题，但如果需要进一步讨论，请随时跟进。

## 贡献

AI2 团队（[@allenai](https://github.com/allenai)）欢迎社区的贡献。如果您是第一次贡献者，我们建议您首先阅读我们的 [CONTRIBUTING.md](https://github.com/allenai/allennlp/blob/main/CONTRIBUTING.md) 指南。然后查看带有标签 [**`Good First Issue`**](https://github.com/allenai/allennlp/issues?q=is%3Aissue+is%3Aopen+label%3A%22Good+First+Issue%22) 的问题。

如果您想贡献较大的功能，请先创建一个带有提议设计的问题，以便讨论。这样可以