# SentencePiece 0.3.0 Release Notes (Preview / Branch)

> **Important Status Note**: SentencePiece **0.3.0** has **not** been officially released yet and is currently maintained in a development branch (while `master` HEAD is at **0.2.3**). APIs and behaviors described below apply to **0.3.0** compared to **v0.2.3** and are subject to final review before merging into `master`.

SentencePiece **0.3.0** unifies the `SentencePieceProcessor` runtime with the FlatBuffer-backed `sentencepiece::lite` engine (`SentencePieceLiteProcessor`). Models loaded from either traditional Protobuf `.model` files or FlatBuffer `.spm.fb` files now execute on the same backend, enabling native zero-copy / `mmap` model loading and a significantly lower memory footprint.

---

## 1. Unified FlatBuffer Runtime & Reduced Memory Footprint

- **Unified `SentencePieceLiteProcessor` Backend**:
  - `SentencePieceProcessor` now executes all tokenization, normalization, and detokenization operations on the `SentencePieceLiteProcessor` engine.
  - Traditional Protobuf `.model` files continue to work transparently and are converted to the FlatBuffer representation in memory upon load.
- **Lower Memory Footprint**:
  - During normal inference, `SentencePieceProcessor` no longer keeps a duplicate copy of all vocabulary pieces in a heap `ModelProto`.
  - Calling `model_proto()` or `serialized_model_proto()` reconstructs and caches the full `ModelProto` lazily on first access.

---

## 2. New FlatBuffer & Zero-Copy Loading APIs (C++ & Python)

### C++ API (`SentencePieceProcessor`)

| New Method | Description |
| :--- | :--- |
| `LoadFromSerializedFlatbuffer(absl::string_view serialized)` | Copies a `.spm.fb` FlatBuffer binary into an internal 8-byte aligned buffer and initializes the processor. |
| `LoadFromFlatbufferView(absl::string_view flatbuffer_view)` | **Zero-copy** initialization directly over caller-owned memory (e.g., `mmap` or static memory) when `flatbuffer_view.data()` is 8-byte aligned. Automatically falls back to copying into an internal aligned buffer if unaligned. |
| `serialized_flatbuffer() const -> absl::string_view` | Returns a `string_view` of the underlying FlatBuffer model binary, or `""` if no model is loaded. |

> **Note**: `Load(filename)` and `LoadFromSerializedProto(serialized)` strictly expect `.model` Protobuf data. Use `LoadFromSerializedFlatbuffer` or `LoadFromFlatbufferView` for `.spm.fb` FlatBuffer binaries.

### Python API (`sentencepiece.SentencePieceProcessor`)

- **New Class Constructors**:
  - `SentencePieceProcessor.from_flatbuffer(serialized: bytes) -> SentencePieceProcessor`: Loads a `.spm.fb` FlatBuffer by copying into an internal aligned buffer.
  - `SentencePieceProcessor.from_flatbuffer_view(buffer) -> SentencePieceProcessor`: Loads a `.spm.fb` FlatBuffer with **zero-copy** directly from any 1-D contiguous buffer-protocol object (`mmap.mmap`, `multiprocessing.shared_memory.SharedMemory.buf`, `memoryview`, `numpy.ndarray`, `bytes`, `bytearray`).
- **New `__init__` Keyword Arguments**:
  - `SentencePieceProcessor(model_flatbuffer=...)`
  - `SentencePieceProcessor(model_flatbuffer_view=...)`
- **New Instance Methods**:
  - `LoadFromSerializedFlatbuffer(serialized: bytes) -> bool`
  - `LoadFromFlatbufferView(buffer) -> bool`
  - `serialized_flatbuffer() -> bytes`
- **Safe Zero-Copy Lifetime Management**:
  - When a zero-copy view is loaded via `from_flatbuffer_view` / `LoadFromFlatbufferView`, `SentencePieceProcessor` holds an exported `memoryview` lock on the buffer. Attempting to close an underlying `mmap.mmap` or `SharedMemory` while the processor is still using it raises `BufferError` instead of causing a use-after-free crash.

**Python Zero-Copy `mmap` Example**:
```python
import mmap
import sentencepiece as spm

with open("model.spm.fb", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

sp = spm.SentencePieceProcessor.from_flatbuffer_view(mm)
ids = sp.encode("Hello world!")
```

---

## 3. Public API & Behavioral Changes in 0.3.0 (vs v0.2.3)

### 3.1 C++ `IdToPiece()` Return Type Changed to `absl::string_view`
- **v0.2.3**: `const std::string& IdToPiece(int id) const`
- **0.3.0**: `absl::string_view IdToPiece(int id) const`
  - Returns a zero-copy `absl::string_view`.
  - Returns `"<unk>"` instead of crashing when `id` is out of range or no model is loaded (use `SafeIdToPiece(int id, std::string* piece)` to explicitly check bounds).
  - Note: `PieceToId(absl::string_view piece)` is unchanged from **v0.2.3** — it returns `unk_id()` (typically `0`) when `piece` is not in the vocabulary, and `-1` only when no model is loaded or when `unk_id() == -1` (`--unk_id=-1 --byte_fallback=true`).

### 3.2 Deprecation of `NBestEncode()` & Removal of `SetNBestTimeout()`, `SetModel()`, and `SetNormalizer()`
- **`NBestEncode()` / `nbest_encode()` (and `spm_encode --output_format=nbest_piece|nbest_id`) are now deprecated.** Use `SampleEncode()` (`encode(..., enable_sampling=True)` in Python) for subword regularization instead.
- **v0.2.3**: Constructed a Unicode-character `Lattice` and ran backward A* search with a `FreeList` allocator and wall-clock timeout configured via `SetNBestTimeout(int)` (falling back to returning only the 1-best path when the hypothesis allocator or timeout limit was exceeded).
- **0.3.0**:
  - Retained for backward compatibility on **Unigram** models (`--model_type=unigram`, `1 <= nbest_size <= 512`) as a lightweight sampling-based approximation over `SentencePieceLiteProcessor`:
    1. **1st Candidate (`nbests[0]`)**: Exact deterministic 1-best Viterbi segmentation (with its actual Viterbi model score populated even when `nbest_size == 1`; `0.0f` is used only when the normalized input is empty).
    2. **Candidates `2..k`**: Deduplicated approximate segmentations sampled deterministically via `SampleNormalized` (using a fixed-seed portable 64-bit LCG across a Gumbel-Max temperature schedule) and sorted in non-increasing Unigram model score order. Fewer than `nbest_size` candidates are returned when sampling finds fewer distinct segmentations.
  - `nbest_size <= 0` returns an error instead of clamping to `1`.
  - `SetNBestTimeout(int)`, `SetModel(std::unique_ptr<ModelInterface>&&)`, and `SetNormalizer(std::unique_ptr<normalizer::Normalizer>&&)` have been removed from `sentencepiece_processor.h`.

### 3.3 `SampleEncode()` Ignores `nbest_size`
- **v0.2.3**: For Unigram models, `nbest_size > 0` sampled from the top-$l$ A* candidates, while `nbest_size < 0` sampled from the full lattice.
- **0.3.0**: `nbest_size` is ignored for both Unigram and BPE models; `SampleEncode` always samples directly from the full lattice (Gumbel-Max dynamic programming with inverse temperature `alpha` for Unigram; BPE-dropout with probability `alpha` for BPE, equivalent to `nbest_size = -1` in v0.2.3).

### 3.4 Deprecated `ModelProto.denormalizer_spec`
- `ModelProto.denormalizer_spec` (field 5 in `sentencepiece_model.proto`) is now marked `[deprecated = true]` and is no longer applied at runtime.
- **v0.2.3**: `Load()` failed when `denormalizer_spec.precompiled_charsmap` was malformed.
- **0.3.0**: The field is not parsed, so such a model loads successfully. A warning is logged when `denormalizer_spec.precompiled_charsmap` is non-empty.

### 3.5 Unigram L1 Sparse Pruning Now Default (Legacy Discrete Pruning Removed)
- Continuous L1 proximal sparse pruning (`TrainSparsePruning`) is now the default and sole Unigram training algorithm in 0.3.0, and the legacy discrete pruning EM loop (`TrainDiscretePruning`) and `--use_sparse_pruning` flag have been removed.
- `--auto_character_coverage=true` now works out of the box with `--model_type=unigram --byte_fallback=true` across both CLI and Python/C++ APIs.
- `TrainerSpec.shrinking_factor` and `TrainerSpec.num_sub_iterations` are marked `[deprecated = true]` in `sentencepiece_model.proto` (and retired in `spm_train`).

### 3.6 `status()` Returns `FAILED_PRECONDITION` When No Model Is Loaded
- **v0.2.3**: `status()` (and every method that checks it) returned `absl::StatusCode::kInternal` with `"Model is not initialized."` before a model was loaded.
- **0.3.0**: The same condition returns `absl::StatusCode::kFailedPrecondition` with the same message. Code that only checks `status().ok()` is unaffected.

### 3.7 Stricter Model Validation and Integer BPE Merge Ranks
- **BPE scores are used as `int32` merge ranks**: BPE piece scores are truncated to integers when the model is converted to the FlatBuffer representation. Models trained by `spm_train` store integer scores and tokenize identically to v0.2.3; a third-party BPE model with fractional scores may tokenize differently where two candidate merges truncate to the same rank.
- **Models rejected by 0.3.0 that v0.2.3 loaded**: a piece longer than 4096 bytes (v0.2.3 allowed up to 7999 bytes), a BPE piece with a NaN or out-of-`int32`-range score, a piece string that appears twice with different types (for example `<s>` as both `CONTROL` and `NORMAL`), and a CHAR model that has a `NORMAL` piece that is not exactly one Unicode character.
- **BPE: deep chains of `UNUSED` merges exceeding depth 100 map to `<unk>`**: Both v0.2.3 and 0.3.0 resegment merged `UNUSED` BPE pieces recursively up to depth 100 (`kMaxBpeResegmentDepth`). Beyond depth 100, v0.2.3 emitted the remaining `UNUSED` token ID as is, whereas 0.3.0 maps the un-resegmented `UNUSED` piece to `<unk>` (`unk_id`).
- **Model accepted by 0.3.0 that v0.2.3 rejected**: a Unigram model that contains only reserved pieces (no `NORMAL`, `USER_DEFINED`, or `UNUSED` piece) now loads and encodes every input as `<unk>`.

---

## 4. Runtime & Trainer Improvements

- **`Decode(ids)` Runs Directly on the Lite Engine**:
  - `Decode(absl::Span<const int> ids, SentencePieceText* spt)` now calls `SentencePieceLiteProcessor::Decode()` directly and builds pieces, surfaces, and offsets from its output, instead of converting ids to pieces and decoding them again.
- **Lite `Decode()` with an Empty `unk_surface`**:
  - An `<unk>` piece that emits nothing no longer cancels the stripping of the dummy-prefix whitespace of the following piece.
- **Trainer**:
  - `TrainerInterface::LoadSentences` builds the meta-piece normalizer only when meta pieces exist, and skips the extra normalization pass otherwise.
  - `MultiFileSentenceIterator` skips empty input files, and TSV input skips empty lines (#1373).
  - `--seed_sentencepiece_size=0` dynamically scales to `max(100000, 8 * vocab_size)` consistently across both corpus extraction and `--seed_sentencepieces_file` loading.
- **Python**:
  - Pickling or deep-copying a `SentencePieceProcessor` loaded via `LoadFromFlatbufferView` / `from_flatbuffer_view` cleanly drops the internal `memoryview` lock and restores from `serialized_model_proto()`.
- **Internal Cleanup (no public API change)**:
  - Removed unused internal helpers (`log_domain::LogSum`, `log_domain::SanitizeDiscreteProbs`, unused trainer `Lattice` accessors) and unused build dependencies.

---

## 5. Known Issues (Pre-release)

- **Empty `unk_surface` is not preserved by the Protobuf-to-FlatBuffer conversion**: A `.model` whose `TrainerSpec.unk_surface` is explicitly set to `""` is converted without the field, so the lite engine falls back to the default `" ⁇ "`. As a result `Decode(ids)` emits `" ⁇ "` for `<unk>` while `Decode(pieces)` emits nothing. Models with the default or a non-empty `unk_surface` are unaffected.
