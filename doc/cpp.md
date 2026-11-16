# SentencePiece C++ API Reference

This document describes the C++ API for loading models, tokenizing and detokenizing text, normalizing text, inspecting vocabularies, and training new models.

For instructions on building SentencePiece and linking it in your project, see:
- [Building with CMake](cmake.md)
- [Building with Bazel](bazel.md)
- [SentencePiece Lite Runtime Guide](../lite/README.md) (standalone zero-dependency FlatBuffer runtime)

---

## 1. Loading a SentencePiece Model

Include `<sentencepiece_processor.h>`, instantiate `sentencepiece::SentencePieceProcessor`, and load a model from a `.model` file, an in-memory serialized `ModelProto`, or a FlatBuffer `.spm.fb` binary:

```cpp
#include <sentencepiece_processor.h>

sentencepiece::SentencePieceProcessor processor;
const auto status = processor.Load("/path/to/model.model");
if (!status.ok()) {
  std::cerr << status.ToString() << std::endl;
  // Handle error
}

// Or load directly from an in-memory serialized ModelProto:
// absl::string_view serialized_proto = ...;
// const auto status = processor.LoadFromSerializedProto(serialized_proto);
```

### Zero-Copy FlatBuffer Loading (`.spm.fb`)

In SentencePiece 0.3.0+, `SentencePieceProcessor` executes natively on the FlatBuffer runtime and can load `.spm.fb` models directly:

```cpp
// Copy a .spm.fb FlatBuffer into an internal 8-byte aligned buffer:
const auto status = processor.LoadFromSerializedFlatbuffer(flatbuffer_bytes);

// Zero-copy initialization over caller-owned 8-byte aligned memory (e.g., mmap):
const auto status = processor.LoadFromFlatbufferView(mmap_view);
```

---

## 2. Tokenizing Text (Encoding)

Call `SentencePieceProcessor::Encode` to segment raw UTF-8 text into subword pieces (`std::vector<std::string>`) or vocabulary IDs (`std::vector<int>`).

```cpp
// Encode into subword pieces
std::vector<std::string> pieces;
processor.Encode("This is a test.", &pieces);
for (const std::string &token : pieces) {
  std::cout << token << std::endl;
}

// Encode into vocabulary IDs
std::vector<int> ids;
processor.Encode("This is a test.", &ids);
for (const int id : ids) {
  std::cout << id << std::endl;
}
```

### Encoding with Byte Offsets (`SentencePieceText`)

To obtain byte offsets (`begin`, `end`) and surface forms alongside piece strings and IDs, pass a `SentencePieceText` protobuf:

```cpp
#include "sentencepiece.pb.h"

sentencepiece::SentencePieceText spt;
processor.Encode("This is a test.", &spt);
for (const auto &sp : spt.pieces()) {
  std::cout << "id=" << sp.id()
            << " piece=" << sp.piece()
            << " surface=" << sp.surface()
            << " span=[" << sp.begin() << ", " << sp.end() << ")" << std::endl;
}
```

---

## 3. Detokenizing Text (Decoding)

Call `SentencePieceProcessor::Decode` to reconstruct raw text from a sequence of subword pieces (`absl::Span<const std::string>` or `absl::Span<const absl::string_view>`) or vocabulary IDs (`absl::Span<const int>`). In general, detokenization is the exact inverse of encoding on normalized text: `Decode(Encode(Normalize(input))) == Normalize(input)`.

```cpp
std::vector<std::string> pieces = {"▁This", "▁is", "▁a", "▁", "te", "st", "."};
std::string text;
processor.Decode(pieces, &text);
std::cout << text << std::endl;

std::vector<int> ids = {451, 26, 20, 3, 158, 128, 12};
processor.Decode(ids, &text);
std::cout << text << std::endl;
```

You can also pass a `SentencePieceText*` to `Decode` to inspect piece-to-decoded-text byte spans.

---

## 4. N-Best Segmentation & Sampling (Subword Regularization)

### N-Best Encoding

Use `NBestEncode` to obtain the top-`nbest_size` segmentations (supported for Unigram models):

```cpp
std::vector<std::vector<std::string>> nbest_pieces;
processor.NBestEncode("This is a test.", 5, &nbest_pieces);

std::vector<std::vector<int>> nbest_ids;
processor.NBestEncode("This is a test.", 5, &nbest_ids);
```

### Stochastic Sampling (`SampleEncode`)

Use `SampleEncode` for on-the-fly subword regularization (Unigram) or BPE-dropout (BPE):

```cpp
std::vector<std::string> pieces;
processor.SampleEncode("This is a test.", -1, 0.2, &pieces);

std::vector<int> ids;
processor.SampleEncode("This is a test.", -1, 0.2, &ids);
```

`SampleEncode` samples directly from the full lattice using inverse temperature `alpha` for Unigram (Gumbel-Max dynamic programming) or dropout probability `alpha` for BPE ([BPE-Dropout](https://arxiv.org/abs/1910.13267)).

---

## 5. Extra Options (BOS, EOS, Reverse)

Use `SetEncodeExtraOptions` and `SetDecodeExtraOptions` with colon-separated options (`"bos"`, `"eos"`, `"reverse"`):

```cpp
processor.SetEncodeExtraOptions("bos:eos");  // Prepend <s> and append </s>
```

---

## 6. Vocabulary Management

Use the following methods to query vocabulary metadata and convert between pieces and IDs:

```cpp
int vocab_size = processor.GetPieceSize();     // Total vocabulary size
int id = processor.PieceToId("▁foo");          // Vocabulary ID of "▁foo" (unk_id() if not found)
absl::string_view piece = processor.IdToPiece(10);  // Zero-copy string_view for ID 10
float score = processor.GetScore(10);          // Log probability / merge score

bool is_unk = processor.IsUnknown(id);         // True if <unk>
bool is_ctrl = processor.IsControl(id);        // True if control token (e.g., <s>, </s>)
bool is_unused = processor.IsUnused(id);       // True if unused token
bool is_byte = processor.IsByte(id);           // True if byte-fallback token (<0x00>..<0xFF>)

int unk_id = processor.unk_id();
int bos_id = processor.bos_id();
int eos_id = processor.eos_id();
int pad_id = processor.pad_id();
```

---

## 7. Text Normalization

You can normalize text directly via `SentencePieceProcessor::Normalize`, or use the standalone `SentencePieceNormalizer` / `normalizer::Normalizer` classes:

```cpp
std::string normalized;
processor.Normalize("ＡＢＣ　１２３", &normalized);

// With character alignment mapping (normalized byte offset -> original byte offset):
std::vector<size_t> norm_to_orig;
processor.Normalize("ＡＢＣ　１２３", &normalized, &norm_to_orig);
```

---

## 8. Training a Model (`SentencePieceTrainer`)

Include `<sentencepiece_trainer.h>` and call `sentencepiece::SentencePieceTrainer::Train` to train a new model.

### Using a Command-Line Flag String

```cpp
#include <sentencepiece_trainer.h>

sentencepiece::TrainerComponents components;
const auto status = sentencepiece::SentencePieceTrainer::Train(
    "--input=data/botchan.txt --model_prefix=m --vocab_size=1000",
    components);
```

### Using a Key-Value Map

```cpp
#include <sentencepiece_trainer.h>

sentencepiece::TrainerComponents components;
const auto status = sentencepiece::SentencePieceTrainer::Train(
    {
        {"input", "data/botchan.txt"},
        {"model_prefix", "m"},
        {"vocab_size", "1000"},
        {"model_type", "unigram"},
    },
    components);
```

### Using `TrainerComponents` (`TrainerSpec`, Pretokenizer, In-Memory Output)

For programmatic control—such as configuring `TrainerSpec` and `NormalizerSpec` directly, streaming sentences via `SentenceIterator`, attaching a custom `pretokenizer` callback (`std::function<std::vector<std::string>(absl::string_view)>`), or writing the trained serialized `ModelProto` directly to memory without creating files on disk—populate `TrainerComponents`:

```cpp
#include <sentencepiece_trainer.h>
#include "sentencepiece_model.pb.h"

sentencepiece::TrainerComponents components;

auto* trainer_spec = components.mutable_trainer_spec();
trainer_spec->add_input("data/botchan.txt");
trainer_spec->set_vocab_size(1000);
trainer_spec->set_model_type(sentencepiece::TrainerSpec::UNIGRAM);

auto* normalizer_spec = components.mutable_normalizer_spec();
normalizer_spec->set_name("nmt_nfkc");

// Optional: attach a custom pretokenizer callback
// components.pretokenizer = [](absl::string_view normalized) { ... };

std::string serialized_model_proto;
const auto status = sentencepiece::SentencePieceTrainer::Train(
    components, &serialized_model_proto);
```
