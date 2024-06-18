from collections import defaultdict
from setuptools import find_packages, setup

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release

#在软件版本控制中，特别是遵循 PEP 440 规范的语义版本（Semantic Versioning），版本号通常采用一定的命名约定来表示不同的发布状态和类型。以下是常见的版本标记解释：
# **Final release (`X.Y`)**: 表示稳定版本或正式发布版本。通常意味着该版本经过了测试和验证，可以用于生产环境。
# **Alpha release (`X.YaN`)**: 表示预览版本或开发中版本。这种版本可能会包含较多的 bug，并且功能可能不完整。通常这些版本供开发人员和测试人员使用，不建议在生产环境中使用。
# **Beta release (`X.YbN`)**: 表示测试版本或公开测试版本。这种版本相对于 Alpha 版本来说更加稳定，但仍可能存在一些问题。通常这些版本用于公开测试和用户反馈，不建议在生产环境中使用。
# **Release Candidate (`X.YrcN`)**: 表示候选版本。这种版本通常是开发完成后的最后一个测试版本，如果没有发现重大问题，可能会成为正式的 Final Release 版本。
#在 PEP 440 中，版本号的格式和命名规则有详细的定义，以便在软件开发和版本管理中能够清晰地区分不同状态和类型的发布版本。


#这段代码是用来解析一个给定路径下的 requirements 文件，并根据文件内容生成两个主要的数据结构：requirements 列表和 extras 字典。
def parse_requirements_file(path, allowed_extras: set = None, include_all_extra: bool = True):
    requirements = []                       #存储主要的依赖项的列表。
    extras = defaultdict(list)              #extras: 使用 defaultdict 创建的字典，用于存储不同额外依赖的依赖项列表。defaultdict(list) 确保如果某个额外依赖还未出现，也能以空列表的形式进行初始化。
    with open(path) as requirements_file:
        import re

        #fix_url_dependencies 函数用于修正 URL 形式的依赖项描述，确保其符合 Pip 和 setuptools 的处理方式。它通过正则表达式匹配 GitHub 仓库的 URL，并返回修正后的依赖描述。
        def fix_url_dependencies(req: str) -> str:
            """Pip and setuptools disagree about how URL dependencies should be handled."""
            m = re.match(
                r"^(git\+)?(https|ssh)://(git@)?github\.com/([\w-]+)/(?P<name>[\w-]+)\.git", req
            )
            if m is None:
                return req
            else:
                return f"{m.group('name')} @ {req}"

        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            req, *needed_by = line.split("# needed by:")
            req = fix_url_dependencies(req.strip())
            if needed_by:
                for extra in needed_by[0].strip().split(","):
                    extra = extra.strip()
                    if allowed_extras is not None and extra not in allowed_extras:
                        raise ValueError(f"invalid extra '{extra}' in {path}")
                    extras[extra].append(req)
                if include_all_extra and req not in extras["all"]:
                    extras["all"].append(req)
            else:
                requirements.append(req)
    return requirements, extras


integrations = {"checklist"}

# Load requirements.
install_requirements, extras = parse_requirements_file(
    "requirements.txt", allowed_extras=integrations
)
dev_requirements, dev_extras = parse_requirements_file(
    "dev-requirements.txt", allowed_extras={"examples"}, include_all_extra=False
)
extras["dev"] = dev_requirements
extras.update(dev_extras)

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import allennlp whilst setting up.
VERSION = {}  # type: ignore
with open("allennlp/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="allennlp",
    version=VERSION["VERSION"],
    description="An open-source NLP research library, built on PyTorch.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="allennlp NLP deep learning machine reading",
    url="https://github.com/allenai/allennlp",
    author="Allen Institute for Artificial Intelligence",
    author_email="allennlp@allenai.org",
    license="Apache",
    packages=find_packages(
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
            "test_fixtures",
            "test_fixtures.*",
            "benchmarks",
            "benchmarks.*",
        ]
    ),
    install_requires=install_requirements,
    extras_require=extras,
    entry_points={"console_scripts": ["allennlp=allennlp.__main__:run"]},
    include_package_data=True,
    python_requires=">=3.7.1",
    zip_safe=False,
)
