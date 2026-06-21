# SentencePiece

[![Build C++](https://github.com/google/sentencepiece/actions/workflows/cmake.yml/badge.svg)](https://github.com/google/sentencepiece/actions/workflows/cmake.yml)
[![Build Wheels](https://github.com/google/sentencepiece/actions/workflows/wheel.yml/badge.svg)](https://github.com/google/sentencepiece/actions/workflows/wheel.yml)
[![GitHub Issues](https://img.shields.io/github/issues/google/sentencepiece.svg)](https://github.com/google/sentencepiece/issues)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sentencepiece)
[![PyPI version](https://badge.fury.io/py/sentencepiece.svg)](https://badge.fury.io/py/sentencepiece)
[![PyPi downloads](https://img.shields.io/pypi/dm/sentencepiece?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/sentencepiece/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![SLSA 3](https://slsa.dev/images/gh-badge-level3.svg)](https://slsa.dev)

SentencePiece is a fast, lightweight, and unsupervised text tokenizer and detokenizer designed for neural network-based text generation systems (such as Large Language Models) where the vocabulary size is fixed prior to training.

It implements **subword units**—including **Byte-Pair-Encoding (BPE)** [[Sennrich et al.](https://www.aclweb.org/anthology/P16-1162)] and the **unigram language model** [[Kudo.](https://arxiv.org/abs/1804.10959)]—with the ability to train directly from raw sentences. By treating input text as a raw sequence of Unicode characters, SentencePiece enables a purely end-to-end, language-independent pipeline that completely eliminates the need for language-specific pre- or post-processing.

***This is not an official Google product.***

---

## Quick Start (Python)

SentencePiece provides an easy-to-use Python module. Install it via `pip`:

```bash
pip install sentencepiece
```

### Basic Example

Here is how to train a model, encode text into tokens/IDs, and decode them back to the original string:

```python
import sentencepiece as spm

# 1. Train a model directly from a raw text file.
# (No pre-tokenization or language-specific preprocessing required!)
spm.SentencePieceTrainer.train(
    input='data/botchan.txt', 
    model_prefix='m', 
    vocab_size=1000
)

# 2. Load the trained model.
sp = spm.SentencePieceProcessor(model_file='m.model')

# 3. Encode raw text into subword pieces (strings) or vocabulary IDs (integers).
text = "I saw a girl with a telescope."
pieces = sp.encode(text, out_type=str)
ids = sp.encode(text, out_type=int)

print(f"Pieces: {pieces}")
# Output: ['▁I', '▁saw', '▁a', '▁girl', '▁with', '▁a', '▁', 'te', 'le', 's', 'c', 'o', 'pe', '.']

print(f"IDs:    {ids}")
# Output: [9, 459, 11, 939, 44, 11, 4, 142, 82, 8, 28, 21, 132, 6]

# 4. Decode IDs or pieces back into the original text.
# The reconstruction is completely lossless and reversible!
print(sp.decode(ids))
# Output: "I saw a girl with a telescope."

print(sp.decode(pieces))
# Output: "I saw a girl with a telescope."
```

---

## Why SentencePiece?

### 1. Reversible & Lossless Tokenization (Whitespace as a Basic Symbol)
Traditional tokenizers drop whitespace information (e.g., treating `Tokenize("World.")` identically to `Tokenize("World .")`), making detokenization ambiguous and language-dependent. 

SentencePiece treats the input text as a raw sequence of Unicode characters. It escapes whitespaces with a meta-symbol `▁` (U+2581) and includes it in the tokenization. This design ensures that **detokenization is a simple, lossless string join operation**, entirely independent of the language:
```python
# Lossless detokenization
original_text = "".join(pieces).replace("▁", " ")
```

### 2. Purely Data-Driven & Language-Independent
SentencePiece trains tokenization and detokenization models directly from raw sentences. It does **not** require language-specific pre-tokenizers (such as Moses, MeCab, or KyTea). This makes it highly effective for languages without explicit word boundaries, such as Chinese, Japanese, and Korean.

### 3. Subword Regularization & BPE-Dropout
To improve the robustness and accuracy of translation and language models, SentencePiece supports on-the-fly subword sampling during training. By sampling different segmentations for the same input text (Subword Regularization for Unigram, BPE-Dropout for BPE), it virtually augments your training data and makes the model more resilient to spelling variations and noise.

```python
# Sample different segmentations on-the-fly
for _ in range(3):
    print(sp.encode('New York', out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1))
# May output:
# ['▁', 'N', 'e', 'w', '▁York']
# ['▁New', '▁York']
# ['▁New', '▁Y', 'o', 'r', 'k']
```

### 4. Fast, Lightweight, and Self-Contained
- **Performance**: Written in highly optimized C++. Segmentation speed is around 50,000 sentences per second, with a memory footprint of only ~6MB.
- **Self-Contained**: The generated `.model` file contains the entire normalization rules, vocabulary mapping, and segmentation model. You are guaranteed to get the exact same tokenization results in any environment (C++, Python, Go, etc.) as long as you use the same model file.

---

## Documentation & Resources

For detailed guides, API references, and advanced usage, please refer to the following resources:

*   [Command Line Interface (CLI) & Build Guide](doc/cli.md)
*   [C++ API Reference](doc/cpp.md)
*   [Python API Reference](python/README.md) & [Python Module Directory](python/)
*   [Python Tokenizer Comparison Cheat Sheet](python/tokenizer_comparison_cheat_sheet.md)
*   [Training Options Reference](doc/options.md)
*   [Text Normalization & Custom Rules](doc/normalization.md)
*   [Special Symbols & Control Tokens](doc/special_symbols.md)
*   [Vocabulary Piece Constraints](doc/piece_constraints.md)
*   [Model Protobuf Schema](doc/model_proto.md)
*   [Docker Deployment Guide](contrib/docker/README.md)
*   [NLCodec BPE Trainer (Contrib)](contrib/nlcodec/README.md)

---

## License
SentencePiece is licensed under the [Apache 2.0 License](LICENSE).
