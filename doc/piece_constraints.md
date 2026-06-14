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
| **`pretokenization_delimiter`** | (empty string) | Defines a pre-tokenization delimiter. When specified, pieces crossing this delimiter cannot be included in the vocabulary. The delimiter itself is removed from the text during training, but it acts as a hard boundary. *(Unigram model only)* | (See detailed section below) |

---

## Pre-tokenization Delimiter
The `pretokenization_delimiter` flag (available only in `unigram` mode) is the most general way to introduce arbitrary segmentation boundaries or control constraints into a SentencePiece model. It allows you to enforce hard boundaries based on external pre-tokenization (e.g., word segmenters like MeCab, syntax parsers, or custom rules) without requiring those tools at inference time.

#### How it Works

When you specify a delimiter (e.g., `pretokenization_delimiter="||||"`):
1.  **Preparation**: You pre-tokenize your training corpus using an external tool and insert the delimiter between tokens (e.g., `Sentence||||Piece||||is||||cool`).
2.  **Splitting**: During training, SentencePiece splits the input at each occurrence of the delimiter.
3.  **Boundary Insertion**: It inserts an internal boundary marker at the split points, preventing any vocabulary candidate from crossing the delimiter location.
4.  **Removal**: The delimiter character itself is removed from the text before subwords are learned.

#### Example Use Case: Mimicking Word Boundaries

A common use case is training a model that respects morphological (word) boundaries for languages like Japanese or Chinese, without needing a morphological analyzer during inference.

1.  Tokenize your training data with a morphological analyzer (e.g., MeCab) and join with `||||`:
    `形態素||||の||||一般||||的||||な||||性質`
2.  Train with `--pretokenization_delimiter="||||"`.
3.  The model learns subwords that never cross these morphological boundaries.
4.  During inference, you can feed raw text (e.g., `形態素の一般的な性質`) to the model. The tokenizer will naturally split at the learned boundaries, mimicking the morphological analyzer's behavior.

For a detailed walkthrough of this approach, see the article: [Making SentencePiece Segmentation MeCab-like (in Japanese)](https://qiita.com/taku910/items/fbaeab4684665952d5a9).

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
  --split_digits=true
```

### Python API

Specify the flags as keyword arguments in `SentencePieceTrainer.train()`:

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='my_model',
    vocab_size=8000,
    split_by_unicode_script=True,
    split_digits=True
)
```
