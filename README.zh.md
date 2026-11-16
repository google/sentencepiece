# SentencePiece

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

[![Build C++](https://github.com/google/sentencepiece/actions/workflows/cmake.yml/badge.svg)](https://github.com/google/sentencepiece/actions/workflows/cmake.yml)
[![Build Wheels](https://github.com/google/sentencepiece/actions/workflows/wheel.yml/badge.svg)](https://github.com/google/sentencepiece/actions/workflows/wheel.yml)
[![GitHub Issues](https://img.shields.io/github/issues/google/sentencepiece.svg)](https://github.com/google/sentencepiece/issues)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sentencepiece)
[![PyPI version](https://badge.fury.io/py/sentencepiece.svg)](https://badge.fury.io/py/sentencepiece)
[![PyPi downloads](https://img.shields.io/pypi/dm/sentencepiece?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/sentencepiece/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![SLSA 3](https://slsa.dev/images/gh-badge-level3.svg)](https://slsa.dev)
[![Documentation](https://img.shields.io/badge/docs-google.github.io%2Fsentencepiece-blue)](https://google.github.io/sentencepiece/)

SentencePiece 是一款快速、轻量且无监督的文本分词与反分词（Tokenizer & Detokenizer）工具，专为在训练前预先设定固定词表大小的神经网络文本生成系统（如大语言模型 LLM）设计。

它原生实现了多种**子词单元（Subword Units）**算法——包括**字节对编码（Byte-Pair-Encoding, BPE）** [[Sennrich et al.](https://www.aclweb.org/anthology/P16-1162)] 与 **Unigram 语言模型** [[Kudo.](https://arxiv.org/abs/1804.10959)]——并且能够直接基于未经预处理的原始句子进行模型训练。通过将输入文本统一视为纯 Unicode 字符序列，SentencePiece 构建了纯端到端且完全与具体语言无关的处理流水线，彻底摒弃了对特定语种预分词或后处理规则的依赖。

***声明：本项目并非 Google 官方正式产品。***

---

## 快速上手 (Python)

SentencePiece 提供了开箱即用的 Python 模块。可通过 `pip` 直接安装：

```bash
pip install sentencepiece
```

### 基础使用示例

以下示例展示了如何训练模型、将文本编码为子词切片/词表 ID，以及将它们无损解码还原为原始字符串：

```python
import sentencepiece as spm

# 1. 直接从原始文本文件中训练分词模型。
# （无需任何预分词或特定语种的前置预处理！）
spm.SentencePieceTrainer.train(
    input='data/botchan.txt',
    model_prefix='m',
    vocab_size=1000
)

# 2. 加载训练好的模型。
sp = spm.SentencePieceProcessor(model_file='m.model')

# 3. 将原始文本编码为子词切片（字符串列表）或词表 ID（整数列表）。
text = "I saw a girl with a telescope."
pieces = sp.encode(text, return_type=str)
ids = sp.encode(text, return_type=int)

print(f"Pieces: {pieces}")
# 输出：[' ▁I', ' ▁saw', ' ▁a', ' ▁girl', ' ▁with', ' ▁a', ' ▁', 'te', 'le', 's', 'c', 'o', 'pe', '.']

print(f"IDs:    {ids}")
# 输出：[9, 459, 11, 939, 44, 11, 4, 142, 82, 8, 28, 21, 132, 6]

# 4. 将 ID 或切片解码还原为原始文本。
# 文本重建过程完全无损且严格可逆！
print(sp.decode(ids))
# 输出："I saw a girl with a telescope."

print(sp.decode(pieces))
# 输出："I saw a girl with a telescope."
```

---

## 核心技术优势 (Why SentencePiece?)

### 1. 严格可逆且无损的分词设计（将空格视为普通符号）
传统分词器往往在分词阶段丢弃空格信息（例如将 `Tokenize("World.")` 与 `Tokenize("World .")` 等同处理），导致反分词还原具有歧义且严重依赖特定语种规则。

SentencePiece 将输入文本统一作为纯 Unicode 字符流处理，通过专用元符号 ` ▁ `（U+2581）对空格进行转义，并将其完整纳入分词切片中。这种设计确保了**反分词还原仅需一次简单且无损的字符串拼接操作**，完全独立于具体语言：
```python
# 无损反分词还原
original_text = "".join(pieces).replace(" ▁ ", " ")
```

### 2. 纯数据驱动且语言无关
SentencePiece 直接从原始句子语料中训练分词与反分词模型。它**不需要**任何特定语种的预分词器（如 Moses、MeCab 或 KyTea）。这一特性使其在处理中文、日文、泰文等没有显式词边界的语言时表现尤为出色。

### 3. 子词正则化 (Subword Regularization) 与 BPE-Dropout
为了显著提升机器翻译与大语言模型的鲁棒性和泛化能力，SentencePiece 支持在训练阶段进行实时的动态子词采样。通过对相同输入文本采样不同的切分组合（Unigram 模型的子词正则化，BPE 模型的 BPE-Dropout），可以在训练过程中实现数据增强，使下游模型对拼写变异和文本噪声具备更强的抵御能力。

```python
# 动态实时采样不同的分词切分组合
for _ in range(3):
    print(sp.encode('New York', return_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1))
# 可能的输出切分：
# [' ▁ ', 'N', 'e', 'w', ' ▁ York']
# [' ▁ New', ' ▁ York']
# [' ▁ New', ' ▁ Y', 'o', 'r', 'k']
```

### 4. 极致高效、轻量且高度自包含
- **卓越性能**：底层采用深度优化的 C++ 实现，分词速度可达每秒约 50,000 句，运行时内存占用仅约 6MB。
- **完全自包含**：导出的 `.model` 模型文件完整包含了所有的文本标准化规则、词表映射与切分模型。只要使用相同的模型文件，无论在何种编程语言环境（C++、Python、Go 等）中运行，都能保证获得 100% 严格一致的分词结果。

---

## 性能基准评测 (SentencePiece vs. Hugging Face Fast)

### 评测基准配置
*   **测试环境**：24 核 CPU，Python 3.13。
*   **测试数据集**：来自 FLORES-200 的均衡多语言原始语料（包含英语、中文、日文和泰文的平行语料；体积 11.29 MB，共 60,720 行）。其中中日韩及泰文文本均为原始语料，未添加任何人造空格分隔符。
*   **批处理请求规模**：将整个数据集（**60,720 句**）作为**单次批处理请求**（单一 Python `list[str]`）一次性传入。
*   **评估指标**：编码吞吐速率，单位为 **MB/s**（数值越高代表性能越优）。

### 1. Unigram 模型：T5-base（32k 词表大小）

| 分词器 | 1 线程 | 2 线程 | 4 线程 | 8 线程 | 16 线程 | 24 线程 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SentencePiece** | **27.41** | **43.83** | **71.62** | **102.08** | **123.33** | **127.60** |
| Hugging Face Fast | 3.78 | 7.15 | 12.45 | 20.33 | 27.00 | 31.49 |

### 2. BPE 模型：Gemma 3（256k 词表大小）

| 分词器 | 1 线程 | 2 线程 | 4 线程 | 8 线程 | 16 线程 | 24 线程 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SentencePiece** | **7.44** | **12.82** | **23.03** | **36.66** | **48.65** | **52.43** |
| Hugging Face Fast | 3.66 | 6.37 | 10.45 | 15.54 | 21.05 | 20.48 |

### 吞吐量非完全线性扩展的原因剖析：
尽管底层核心分词计算（C++ 或 Rust）以并行方式运行，但将底层原生数据结构（C++ 或 Rust 的嵌套 vector）转换为 Python 对象（`list[list[int]]` 或 `list[Encoding]`）的最终步骤是**串行执行的，且必须在 Python 主线程上持有 GIL 运行**。在高并发线程数下，这一单线程序列化转换步骤成为了主要的性能瓶颈，限制了整体的扩展吞吐量。

关于更深度的性能分析及单线程基准对比，请查阅[性能评测细节文档](doc/performance_benchmark.md)。

如需自行复现运行上述评测，请参考[基准复现指引与脚本](benchmark/README.md)。

---

## SentencePiece Lite（零依赖 C++20 运行时）

针对移动终端应用、边缘计算设备、端侧大模型推理以及高吞吐量 C++ 服务端环境，代码库内置了 **SentencePiece Lite** 极简运行时（位于 [`lite/`](lite/README.md)）：

*   **零外部运行时依赖**：采用现代 C++20 编写，完全不依赖任何第三方运行时库（无需在运行时引入 Protobuf 或 Abseil）。
*   **零拷贝内存映射极速冷启动**：通过 `mmap` 直接读取 FlatBuffers 二进制模型（`.spm.fb`），实现毫秒级瞬间冷启动，且无论词表大小如何，堆内存开销仅有 **1.2 KB**。
*   **极度紧凑的二进制体积**：核心引擎源码不足 2,000 行，编译后的精简静态库体积仅约 **45 KB**（`libsentencepiece_lite.a`）。
*   **完全规范等价性**：在文本标准化与 Token ID 映射上，与标准版 SentencePiece 模型（Unigram 与 BPE）保持 100% 逐字节完全等价。

有关详细信息、基准测试数据以及 C++ 快速集成范式，请参阅 [SentencePiece Lite 文档](lite/README.md)。

---

## 文档与参考资源索引

如需查阅深入技术指南、API 文档以及进阶用法，请访问以下资源：

*   [SentencePiece Lite 运行时指南](lite/README.md)
*   [命令行工具 (CLI) 使用指南](doc/cli.md)
*   [CMake 编译构建指引](doc/cmake.md)
*   [Bazel 编译构建指引](doc/bazel.md)
*   [C++ API 参考文档](doc/cpp.md)
*   [Python API 参考文档](python/README.md) 与 [Python 模块源码目录](python/)
*   [Python 分词器选型对比速查表](python/tokenizer_comparison_cheat_sheet.md)
*   [性能评测详细分析报告](doc/performance_benchmark.md)
*   [性能评测代码与复现指引](benchmark/README.md)
*   [模型训练参数全集参考](doc/options.md)
*   [自动字符覆盖率与字节回退机制指南](doc/auto_character_coverage.md)
*   [稀疏 L1 Unigram 剪枝指南](doc/sparse_l1_unigram.md)
*   [文本标准化与自定义规则配置](doc/normalization.md)
*   [特殊符号与控制 Token 配置](doc/special_symbols.md)
*   [词表词元约束配置](doc/piece_constraints.md)
*   [模型 Protobuf 模式定义](doc/model_proto.md)
*   [Docker 容器化部署指南](contrib/docker/README.md)

---

## 开源许可证 (License)
SentencePiece 遵循 [Apache 2.0 许可证](LICENSE) 开源发布。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年11月16日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
