# AllenNLP GitHub 和 PyPI 发布流程

本文档描述了发布核心库新版本的流程。

> ❗️ 假设您正在使用指向 `git@github.com:allenai/allennlp.git` （或等效的 `HTTPS`）的主仓库克隆。

## 步骤

1. 设置环境变量 `TAG`，其形式应为 `v{版本号}`。

    例如，如果发布的版本号为 `1.0.0`，则应将 `TAG` 设置为 `v1.0.0`：

    ```bash
    export TAG='v1.0.0'
    ```

    或者如果您使用 `fish` shell：

    ```fish
    set -x TAG 'v1.0.0'
    ```

2. 更新 `allennlp/version.py` 文件中的版本号。然后检查以下命令的输出：

    ```
    python scripts/get_version.py current
    ```

    确保其与 `TAG` 环境变量匹配。

3. 更新 `CHANGELOG.md` 文件，将所有在 "Unreleased" 部分下的内容移动到与此发布对应的新版块下。

4. 更新 `CITATION.cff` 文件以引用正确的版本。

5. 使用以下命令提交并推送这些更改：

    ```
    git commit -a -m "Prepare for release $TAG" && git push
    ```

6. 然后在 Git 中添加标签以标记发布：

    ```
    git tag $TAG -m "Release $TAG" && git push --tags
    ```

7. 在 [GitHub](https://github.com/allenai/allennlp/tags) 上找到刚刚推送的标签，点击编辑，然后复制以下命令的输出：

    ```
    python scripts/release_notes.py
    ```

    在 macOS 上，例如，可以直接将上述命令的输出复制到剪贴板。

8. 如果发布是一个候选版本（以 `rc*` 结尾），请勾选 "This is a pre-release" 复选框。否则，保持未勾选状态。

9. 点击 "Publish Release"。GitHub Actions 将会处理剩下的工作，包括将软件包发布到 PyPI 和 Docker 镜像发布到 Docker Hub。

10. 在 [GitHub Actions 工作流程](https://github.com/allenai/allennlp/actions?query=workflow%3AMaster+event%3Arelease) 完成后，按照同样的步骤发布 `allennlp-models` 仓库的发布版本。

## 修复发布失败

如果由于某种原因导致 GitHub Actions 发布工作流程失败，需要进行修复，您必须删除 GitHub 上的标签和相应的发布。在修复后推送之后，可以使用以下命令从本地克隆中删除标签：

```bash
git tag -l | xargs git tag -d && git fetch -t
```

然后重复以上步骤。