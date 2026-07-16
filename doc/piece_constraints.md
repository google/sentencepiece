# Vocabulary Piece Constraints

SentencePiece is **deliberately designed to operate directly on raw, un-pretokenized text** (without requiring steps like space-splitting or regex-based rules). Instead of segmenting the input before tokenization, SentencePiece applies a set of constraints during training to determine which subwords (pieces) are valid candidates for the vocabulary.

## Design Philosophy: Why Avoid Pre-tokenization?

Many subword tokenizers require a pre-tokenization step (e.g., space-splitting or regex rules) that has several issues:

*   **Language Dependency**: Managing external morphological analyzers (e.g., MeCab, Jieba) for non-space-segmented languages adds system complexity.
*   **Security Risks (ReDoS)**: Regex-based splitters are vulnerable to Regular Expression Denial of Service (ReDoS) attacks (such as [CVE-2025-1194](https://nvd.nist.gov/vuln/detail/CVE-2025-1194) in Hugging Face's GPT-NeoX-Japanese tokenizer).
*   **Inconsistency & Portability**: Pre-tokenizers often rely on regex engines (such as Python or Rust) or runtime Unicode versions, which can behave differently across environments, making consistent tokenization and cross-language deployment extremely difficult.

To avoid these issues, SentencePiece operates directly on raw text and applies **piece constraints during training** to construct a static vocabulary. Because these constraints do not run during inference, the resulting model is completely self-contained, guaranteeing consistent, safe, and fast tokenization across all platforms. For cases where space-separated tokenization is desired, simple constraints like `split_by_whitespace=true` can safely approximate this behavior.

---

## Constraint Flags Reference

The following table describes the flags that control whether a candidate subword is considered a valid sentencepiece.

| Flag Name | Default Value | Description | Examples |
| :--- | :--- | :--- | :--- |
| **`max_sentencepiece_length`** | `16` | Maximum length (in Unicode characters) of a vocabulary piece. | If `16`, `understanding` (13 chars) is valid, `counterunderstanding` (20 chars) is invalid. |
| **`split_by_unicode_script`** | `true` | Prevents a single piece from crossing Unicode script boundaries (e.g., mixing Latin and Han scripts). | If `true`, `hello世界` is invalid (must be split into `hello` and `世界`). If `false`, `hello世界` can be a single piece. <br><br> *Note: Hiragana and Katakana are internally merged with Han (Kanji) script, allowing Japanese mixed-script words (e.g., `おいしい屋`) to remain in a single piece.* |
| **`split_by_number`** | `true` | Treats numbers as a separate script. When `split_by_unicode_script` is true, this prevents numbers from mixing with alphabetical letters in a single piece. | If `true`, `temp20a` is invalid. If `false`, `temp20a` can be a single piece. |
| **`split_digits`** | `false` | Forces all digits (0-9) to be split into individual pieces of length 1. | If `true`, `1999` must be split into `1`, `9`, `9`, `9`. `19` is an invalid piece. |
| **`split_by_whitespace`** | `true` | Prevents pieces from crossing whitespace boundaries. Whitespace (represented by the meta-symbol `▁`) can only appear at the boundary (prefix or suffix). | If `true`, `foo▁bar` is invalid. If `false`, `foo▁bar` (representing "foo bar") can be a single piece. |
| **`treat_whitespace_as_suffix`** | `false` | Controls the position of the whitespace meta-symbol. If `false`, whitespace must appear as a prefix. If `true`, whitespace must appear as a suffix. | If `false` (prefix): `▁hello` is valid, `hello▁` is invalid. <br>If `true` (suffix): `hello▁` is valid, `▁hello` is invalid. |
| **`allow_whitespace_only_pieces`** | `false` | Allows pieces that consist entirely of whitespace characters. | If `false`, `▁▁` is invalid (though a single `▁` is allowed). If `true`, `▁▁` is a valid piece. |
| **`pretokenization_delimiter`** | (empty string) | Defines a pre-tokenization delimiter. When specified, pieces crossing this delimiter cannot be included in the vocabulary. The delimiter itself is removed from the text during training, but it acts as a hard boundary. Supported in both Unigram and BPE models. | (See detailed section below) |
| **`pretokenizer`** | `nullptr` / `None` | Runtime callback function (`Callable[[str], List[str]]` in Python or `std::function<vector<string>(string_view)>` in C++) that receives normalized text and returns a list of pretokenized token chunks. *(Python support available in v0.2.3+)* | (See detailed section below) |
| **`allow_inconsistent_pretokenization`** | `false` | When `false` (default), SentencePiece verifies that reassembling the pretokenized chunks (`"".join(chunks)`) matches the input normalized sentence. If characters are dropped or modified, training fails with an error. When `true`, this check is bypassed (at user's own risk). *(Python support available in v0.2.3+)* | `pretokenizer=..., allow_inconsistent_pretokenization=True` |

---

## Pre-tokenization Delimiter and Callback Functions

> [!WARNING]
> Pre-tokenization constraints apply **ONLY during model training** to prevent vocabulary pieces from crossing split boundaries. They are **NOT executed during inference (`encode`)**, nor are external splitters required at runtime.
>
> **Consistency Requirement**: Boundaries must be applied 100% consistently across the entire corpus. If two tokens remain concatenated even once, that cross-boundary subword can still be extracted into the vocabulary.

SentencePiece supports two mechanisms to enforce segmentation boundaries during subword extraction (supported in both Unigram and BPE):
1. **`pretokenization_delimiter`**: A static delimiter string (e.g., `||||`) in pre-segmented text files.
2. **`pretokenizer`**: A dynamic Python/C++ callback (`Callable[[str], List[str]]`) called on **normalized text**. *(Python support available in v0.2.3+)*

Both mechanisms convert split points into internal markers (`0x001F`) during training to forbid cross-boundary subword candidates. `pretokenization_delimiter` and `pretokenizer` are mutually exclusive.

### Custom Pretokenizer Callbacks (Python)

> [!NOTE]
> Custom Python `pretokenizer` callbacks are available in SentencePiece **v0.2.3 and later**.

Passed via `pretokenizer` in `SentencePieceTrainer.train()`:

- **Input**: Normalized string (where spaces are converted to `▁` U+2581).
- **Output**: List of string chunks (`List[str]`).

```python
import re
import sentencepiece as spm

# Note: Callback receives normalized text (spaces are '▁' U+2581)
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='m_pretok',
    vocab_size=8000,
    pretokenizer=lambda text: text.split('▁')
)
```

#### Reassembly Check and `allow_inconsistent_pretokenization`

By default, SentencePiece verifies `"".join(chunks) == normalized_input`. If characters are dropped or modified, training fails:

```
RuntimeError: Pretokenized output mismatch: joined='hello', original='▁hello▁world'.
Set allow_inconsistent_pretokenization=true in TrainerComponents to bypass.
```

To intentionally filter/modify text during pre-tokenization, set `allow_inconsistent_pretokenization=True` (at user's own risk).

#### Common Python Pretokenizer Regex Patterns

You can pass standard `def` functions, `lambda` expressions, or a compiled regex's `.findall` method directly as `pretokenizer`. All patterns must preserve 100% character coverage (`"".join(pretokenizer(text)) == text`):

```python
import re

# 1. Whitespace Splitter
def whitespace_pretokenizer(text: str) -> list[str]:
    return [c for c in re.split(r'(▁+|\s+)', text) if c]

# 2. Word & Punctuation Splitter
PAT_WORD_PUNCT = re.compile(r'\w+|[^\w\s]|▁+|\s+')
# Usage: pretokenizer=PAT_WORD_PUNCT.findall

# 3. LLaMA / Qwen-style Contraction & Digit Splitter (Max 3 digits)
PAT_LLAMA = re.compile(r"'[a-zA-Z]+|\w+|\d{1,3}|[^\w\s]|▁+|\s+")
# Usage: pretokenizer=PAT_LLAMA.findall

# 4. CamelCase Splitter (e.g. "CamelCase" -> ["Camel", "Case"])
PAT_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+|[^\w\s]|▁+|\s+")
# Usage: pretokenizer=PAT_CAMEL.findall

# 5. CJK & Multilingual Script Splitter
PAT_CJK = re.compile(r'[\u3040-\u30ff\u4e00-\u9faf]+|[a-zA-Z]+|\d+|[^\w\s]|▁+|\s+')
# Usage: pretokenizer=PAT_CJK.findall

# 6. Combined All-in-One Splitter (Contractions + CamelCase + Digits <=3 + CJK + Punctuation)
PAT_COMBINED = re.compile(
    r"'[a-zA-Z]+|[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[\u3040-\u30ff\u4e00-\u9faf]+|\d{1,3}|[^\w\s]|▁+|\s+"
)
# Usage: pretokenizer=PAT_COMBINED.findall
```

---

## Offline Delimiter Example

Pre-tokenize training corpus offline with an external tool (e.g., MeCab) using a delimiter such as `||||`:

```
形態素||||の||||一般||||的||||な||||性質
```

> **Note**: Choose a delimiter string (such as `||||`) that is sufficiently unique and invariant under text normalization so that it will not be altered, converted, or stripped by the normalizer (e.g. NFKC normalization).

Train with `--pretokenization_delimiter="||||"`. The model learns subwords that never cross morphological boundaries without needing MeCab at inference time.

---

## Examples

Here is how to specify these piece constraint flags using the C++ CLI or Python API.

### Command Line Interface (CLI)

You can pass the flags directly to `spm_train`:

```bash
spm_train \
  --input=corpus.txt \
  --model_prefix=my_model \
  --vocab_size=8000 \
  --split_by_unicode_script=true \
  --split_digits=true \
  --pretokenization_delimiter="||||"
```

### Python API

Specify the flags or components as keyword arguments in `SentencePieceTrainer.train()`:

```python
import re
import sentencepiece as spm

# LLaMA / Qwen-style pre-tokenization regex pattern
PAT_LLAMA = re.compile(r"'[a-zA-Z]+|\w+|\d{1,3}|[^\w\s]|▁+|\s+")

spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_llama_model',
    vocab_size=32000,
    model_type='bpe',
    split_digits=True,
    pretokenizer=PAT_LLAMA.findall  # Pass pattern.findall directly as pretokenizer
)
```
