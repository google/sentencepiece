# SentencePiece Lite Runtime

## 1. Overview

SentencePiece Lite Runtime is a compact (~50 KB binary size, < 2k LOC), zero-dependency C++ library designed to execute tokenization directly on FlatBuffers model binaries (`.spm.fb`). It provides the essential operations for SentencePiece tokenization—including Unigram and BPE algorithms, text normalization, byte fallback, raw input offset tracking, stochastic sampling, and safe boundary pre-tokenization.

By accessing memory-mapped FlatBuffers structures directly, the runtime achieves zero-copy startup, faster throughput, and reduced heap memory allocations compared to the original SentencePiece library.

While currently implemented as a standalone C++ library, we plan to adopt it as the core runtime engine for SentencePiece.

## 2. Core Design Principles & Constraints

*   **Fundamental Operations for Core Functionality**: Provides essential primitives—including Unicode text normalization, raw input byte offset tracking, stochastic subword sampling (Gumbel Unigram & BPE Dropout), byte fallback, and safe boundary detection—enabling callers to construct complete tokenization workflows without heavy library dependencies.
*   **Compact Footprint (~50 KB Binary Size, < 2k LOC)**: Implemented in under 2,000 lines of C++ code, compiling into a 45 KB static binary (`libsentencepiece_lite.a` stripped). It can be linked into mobile apps, iOS/Android SDKs, edge devices, and memory-constrained embedded systems.
*   **Minimal Dependencies**: Depends only on the FlatBuffers runtime library and standard C++20 (no runtime dependency on Protobuf or Abseil).
*   **Zero-Copy Startup**: Maps the FlatBuffers model binary directly into virtual memory (`mmap`). Vocabulary strings, feature scores, and offline Double-Array Trie (`Darts::DoubleArray`) structures are read in-place without heap allocations.
*   **Throughput**: Double-Array Trie lookups, bigram pre-tokenization, and zero-copy string views deliver up to 29.4x faster encoding (37.3x with token caching) and 9.4x faster decoding compared to the original Protobuf-based runtime.
*   **Amalgamation Support (2-File Standalone Distribution)**: Provides a consolidated 2-file distribution (`dist/sentencepiece_lite.h` and `dist/sentencepiece_lite.cc`) with all dependencies (FlatBuffers runtime and rapidhash) inlined. Drop these 2 files into any C++ project and compile directly with `-std=c++20` without external library linking.

## 3. Performance Evaluation & Resource Footprint

### 3.1. Single-Core Encoding Throughput

We evaluated single-thread encoding speed across **FineWeb2** datasets (5.0 MB English, 5.0 MB Japanese, and 8.9 MB Multilingual Mixed). We compared five configurations: [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite), **SentencePiece Lite**, **SentencePiece**, **Tiktoken (v0.14.0)**, and **Hugging Face Tokenizers (v0.23.1 Fast Mode)**.

#### (1) FineWeb2 English Corpus (5.0 MB)

| Engine | Model | Algorithm | Throughput | Tokens |
| :--- | :--- | :---: | :---: | :---: |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **LLM-jp-4 (196K)** | **Unigram** | **53.01 MB/s** | 1,189,510 |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **Gemma 3 (256K)** | **BPE** | **52.99 MB/s** | 1,165,287 |
| **SentencePiece Lite** | **Gemma 3 (256K)** | **BPE** | **41.70 MB/s** | 1,165,287 |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **PLaMo-13B (64K)** | **Unigram** | **30.37 MB/s** | 1,410,864 |
| **SentencePiece Lite** | **LLM-jp-4 (196K)** | **Unigram** | **23.57 MB/s** | 1,189,510 |
| **SentencePiece Lite** | **PLaMo-13B (64K)** | **Unigram** | **18.81 MB/s** | 1,410,785 |
| **Tiktoken (v0.14.0)** | **o200k_base (200K)** | **BPE** | **18.54 MB/s** | 1,119,612 |
| **Tiktoken (v0.14.0)** | **Llama 3 (128K)** | **BPE** | **17.96 MB/s** | 1,131,445 |
| **SentencePiece** | **PLaMo-13B (64K)** | **Unigram** | **17.47 MB/s** | 1,410,782 |
| **SentencePiece** | **LLM-jp-4 (196K)** | **Unigram** | **15.98 MB/s** | 1,189,431 |
| **Tiktoken (v0.14.0)** | **cl100k_base (100K)** | **BPE** | **12.97 MB/s** | 1,131,809 |
| **Hugging Face Tokenizers (v0.23.1)** | **Llama 3 (128K)** | **BPE** | **2.16 MB/s** | 1,131,446 |
| **Hugging Face Tokenizers (v0.23.1)** | **cl100k_base (100K)** | **BPE** | **2.10 MB/s** | 1,131,809 |
| **Hugging Face Tokenizers (v0.23.1)** | **Gemma 3 (256K)** | **BPE** | **2.04 MB/s** | 1,165,288* |
| **Hugging Face Tokenizers (v0.23.1)** | **PLaMo-13B (64K)** | **Unigram** | **1.71 MB/s** | 2,426,835 |
| **Hugging Face Tokenizers (v0.23.1)** | **LLM-jp-4 (196K)** | **Unigram** | **1.49 MB/s** | 1,189,542 |
| **SentencePiece** | **Gemma 3 (256K)** | **BPE** | **1.42 MB/s** | 1,165,287 |

#### (2) FineWeb2 Japanese Corpus (5.0 MB)

| Engine | Model | Algorithm | Throughput | Tokens |
| :--- | :--- | :---: | :---: | :---: |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **Gemma 3 (256K)** | **BPE** | **35.55 MB/s** | 1,040,163 |
| **SentencePiece Lite** | **PLaMo-13B (64K)** | **Unigram** | **33.53 MB/s** | 1,005,446 |
| **SentencePiece Lite** | **Gemma 3 (256K)** | **BPE** | **31.84 MB/s** | 1,040,163 |
| **SentencePiece Lite** | **LLM-jp-4 (196K)** | **Unigram** | **31.80 MB/s** | 816,214 |
| **SentencePiece** | **PLaMo-13B (64K)** | **Unigram** | **29.87 MB/s** | 1,005,432 |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **PLaMo-13B (64K)** | **Unigram** | **29.18 MB/s** | 1,005,449 |
| **SentencePiece** | **LLM-jp-4 (196K)** | **Unigram** | **26.06 MB/s** | 816,125 |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **LLM-jp-4 (196K)** | **Unigram** | **24.32 MB/s** | 816,214 |
| **Tiktoken (v0.14.0)** | **Llama 3 (128K)** | **BPE** | **10.97 MB/s** | 1,283,796 |
| **Tiktoken (v0.14.0)** | **cl100k_base (100K)** | **BPE** | **10.41 MB/s** | 1,916,377 |
| **Tiktoken (v0.14.0)** | **o200k_base (200K)** | **BPE** | **10.35 MB/s** | 1,381,640 |
| **SentencePiece** | **Gemma 3 (256K)** | **BPE** | **9.15 MB/s** | 1,040,163 |
| **Hugging Face Tokenizers (v0.23.1)** | **Gemma 3 (256K)** | **BPE** | **3.11 MB/s** | 1,040,163* |
| **Hugging Face Tokenizers (v0.23.1)** | **PLaMo-13B (64K)** | **Unigram** | **2.57 MB/s** | 1,003,605 |
| **Hugging Face Tokenizers (v0.23.1)** | **Llama 3 (128K)** | **BPE** | **2.13 MB/s** | 1,283,797 |
| **Hugging Face Tokenizers (v0.23.1)** | **LLM-jp-4 (196K)** | **Unigram** | **2.03 MB/s** | 816,213 |
| **Hugging Face Tokenizers (v0.23.1)** | **cl100k_base (100K)** | **BPE** | **1.88 MB/s** | 1,916,377 |

#### (3) FineWeb2 Multilingual Mixed Corpus (8.9 MB)

| Engine | Model | Algorithm | Throughput | Tokens |
| :--- | :--- | :---: | :---: | :---: |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **LLM-jp-4 (196K)** | **Unigram** | **47.78 MB/s** | 3,056,263 |
| **SentencePiece Lite** | **LLM-jp-4 (196K)** | **Unigram** | **40.95 MB/s** | 3,056,184 |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **PLaMo-13B (64K)** | **Unigram** | **31.97 MB/s** | 4,617,972 |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **Gemma 3 (256K)** | **BPE** | **31.88 MB/s** | 1,749,723 |
| **SentencePiece Lite** | **PLaMo-13B (64K)** | **Unigram** | **28.37 MB/s** | 4,617,992 |
| **SentencePiece Lite** | **Gemma 3 (256K)** | **BPE** | **26.50 MB/s** | 1,749,723 |
| **SentencePiece** | **LLM-jp-4 (196K)** | **Unigram** | **25.83 MB/s** | 3,055,825 |
| **SentencePiece** | **PLaMo-13B (64K)** | **Unigram** | **24.92 MB/s** | 4,617,377 |
| **Tiktoken (v0.14.0)** | **Llama 3 (128K)** | **BPE** | **13.45 MB/s** | 2,018,277 |
| **Tiktoken (v0.14.0)** | **o200k_base (200K)** | **BPE** | **12.99 MB/s** | 1,818,536 |
| **Tiktoken (v0.14.0)** | **cl100k_base (100K)** | **BPE** | **10.52 MB/s** | 2,858,926 |
| **SentencePiece** | **Gemma 3 (256K)** | **BPE** | **2.64 MB/s** | 1,749,723 |
| **Hugging Face Tokenizers (v0.23.1)** | **Gemma 3 (256K)** | **BPE** | **2.29 MB/s** | 1,749,724* |
| **Hugging Face Tokenizers (v0.23.1)** | **Llama 3 (128K)** | **BPE** | **1.84 MB/s** | 2,018,278 |
| **Hugging Face Tokenizers (v0.23.1)** | **PLaMo-13B (64K)** | **Unigram** | **1.81 MB/s** | 3,611,833 |
| **Hugging Face Tokenizers (v0.23.1)** | **LLM-jp-4 (196K)** | **Unigram** | **1.69 MB/s** | 3,056,266 |
| **Hugging Face Tokenizers (v0.23.1)** | **cl100k_base (100K)** | **BPE** | **1.56 MB/s** | 2,858,926 |

*\* Note: Hugging Face Tokenizers (v0.23.1) token counts for Gemma 3 are measured with `add_special_tokens=False`.*

### 3.2. Model File Size & Memory Metrics

#### Model File Sizes

| Model | Vocabulary | Algorithm | Raw ModelProto (`.model`) | FlatBuffers Blob (`.spm.fb`) |
| :--- | :---: | :---: | :---: | :---: |
| **Gemma 3** | 256K | BPE | 4.6 MB | **10.3 MB** |
| **LLM-jp-4** | 196K | Unigram | 3.2 MB | **7.5 MB** |
| **PLaMo-13B** | 64K | Unigram | 1.1 MB | **2.8 MB** |

#### Runtime Heap Memory Usage

* **`mmap` Mode (Pre-converted `.spm.fb`)**: Zero-copy loading requires **1.2 KB heap memory** regardless of model size.
* **On-the-Fly Mode (In-memory Conversion)**: The converted FlatBuffers blob size becomes the exact runtime heap usage.

| Model | Original SentencePiece Heap | Lite (`mmap` Mode) | Lite (On-the-Fly Heap) | Memory Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Gemma 3 (256K)** | 43.4 MB | **1.2 KB** | **10.3 MB** | ~4.2x smaller |
| **LLM-jp-4 (196K)** | 27.7 MB | **1.2 KB** | **7.5 MB** | ~3.7x smaller |
| **PLaMo-13B (64K)** | 9.5 MB | **1.2 KB** | **2.8 MB** | ~3.4x smaller |

---

## 4. Basic Usage & Quickstart

### C++ Quick Start

The following snippet demonstrates model loading via memory mapping, followed by encoding and decoding:

```cpp
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "sentencepiece_lite.h"

// 1. Load the model file via memory mapping (mmap)
// (MmapModelFile is a placeholder for your platform-specific file mapping utility)
std::string_view model_view = MmapModelFile("model.spm.fb");

// 2. Initialize the processor referencing the memory-mapped buffer.
// The mapped buffer must outlive the processor instance.
sentencepiece::lite::SentencePieceLiteProcessor processor(model_view);

if (processor.status() != sentencepiece::lite::StatusCode::kOk) {
  std::cerr << "Failed to initialize model." << std::endl;
  return;
}

// Alternatively, transfer ownership using std::shared_ptr<std::string>:
// auto buffer = std::make_shared<std::string>(LoadFileToString("model.spm.fb"));
// sentencepiece::lite::SentencePieceLiteProcessor processor(std::move(buffer));

// 3. Encode raw input text directly to token IDs (text is normalized internally)
std::vector<int> ids;
if (processor.Encode("hello world", &ids) == sentencepiece::lite::StatusCode::kOk) {
  // ids contains the tokenized sequence
}

// 4. Decode token IDs back to text
std::string text;
if (processor.Decode(ids, &text) == sentencepiece::lite::StatusCode::kOk) {
  // text contains the reconstructed string
}
```

---

## 5. Model Conversion & On-the-Fly Usage

> **Note**: While original `.model` files are endian-independent, FlatBuffers binary files (`.spm.fb`) are endian-dependent. They are fully interoperable across most common little-endian platforms (e.g., x86_64, ARM64, Apple Silicon). Please convert `.model` files on the target machine if using different endian architectures.

### 5.1. Offline Model Conversion (CLI)

Convert an original SentencePiece model (`.model`) to a FlatBuffers model (`.spm.fb`) using the converter tool:

```bash
# Convert .model to .spm.fb
./build/lite/spm_to_fb --model=model.model --output=model.spm.fb
# Or with Bazel:
bazel run //lite:spm_to_fb -- --model=model.model --output=model.spm.fb
```

Loading a pre-converted `.spm.fb` file using `mmap` provides zero-copy startup and zero heap allocations.

### 5.2. On-the-Fly Model Conversion (C++)

If you do not have a pre-converted `.spm.fb` file, you can convert a SentencePiece `ModelProto` in memory at runtime (**on-the-fly**):

```cpp
#include <memory>
#include <string>
#include <vector>

#include "sentencepiece_lite.h"
#include "sentencepiece_model_converters.h"

// 1. Load standard SentencePiece ModelProto
sentencepiece::ModelProto model_proto = LoadMyModelProto("model.model");

// 2. Convert ModelProto to FlatBuffers on-the-fly (returns absl::StatusOr<std::string>)
absl::StatusOr<std::string> fb_blob_or =
    sentencepiece::lite::ToFlatbuffer(model_proto);

if (fb_blob_or.ok()) {
  // 3. Create processor referencing the converted FlatBuffers string
  auto shared_fb_model = std::make_shared<std::string>(std::move(*fb_blob_or));
  sentencepiece::lite::SentencePieceLiteProcessor processor(shared_fb_model);

  std::vector<int> ids;
  processor.Encode("hello world", &ids);
}
```

* **Conversion Speed**: In-memory conversion takes less than 1 second for 256K vocabulary models, making it suitable for application startup when pre-converted files are not available.

---

## 6. Usage Recipes & Architecture

### 6.1. Architectural Philosophy

Instead of built-in high-level features, SentencePiece Lite provides fine-grained primitives so callers can build custom advanced features in their own applications.

For example, in addition to the standard `Encode` method, it exposes components like `Normalize`, `EncodeNormalized`, and `PretokenizeAtSafeBoundaries`. Exposing these primitives allows users to implement optimizations—such as parallel document splitting, thread-safe batching, or custom caching—with precise control, while keeping the core engine minimal and self-contained.

### 6.2. Concurrent Sentence Batching

Because `SentencePieceLiteProcessor` is immutable and stateless during execution, a single processor instance can be shared concurrently across multiple worker threads without locking:

```cpp
sentencepiece::lite::SentencePieceLiteProcessor processor(model_data);
std::vector<std::string> batch = {"sentence one", "sentence two"};
std::vector<std::future<std::vector<int>>> futures;

for (const auto& text : batch) {
  futures.push_back(std::async(std::launch::async, [&processor, &text]() {
    std::vector<int> ids;
    processor.Encode(text, &ids);
    return ids;
  }));
}

for (auto& f : futures) {
  std::vector<int> ids = f.get();
}
```

### 6.3. Safe Boundary Pre-tokenization

The public API provides boundary-detection capabilities to safely slice long text into independent segments without breaking vocabulary subword boundaries:

```cpp
// Zero-allocation callback overload (invokes `receiver` for each safe chunk)
StatusCode PretokenizeAtSafeBoundaries(
    std::string_view normalized,
    FunctionRef<void(std::string_view)> receiver) const;

// Convenience overload that collects chunks into a std::vector
StatusCode PretokenizeAtSafeBoundaries(
    std::string_view normalized, std::vector<std::string_view>* out) const;
```

#### Algorithm & Implementation Details

*   **Character Bigram Co-occurrence Principle**: If a character bigram $(C_i, C_{i+1})$ never appears inside any subword token in the vocabulary, no subword piece can span across the boundary between $C_i$ and $C_{i+1}$. The input text can therefore be split safely at this boundary before encoding.
*   **AC-style State Reuse Traversal**: All valid character bigrams present in the vocabulary are compiled offline into a Double-Array Trie (`char_bigram_trie_blob`). During input scanning, the runtime reuses single-character trie node state transitions across adjacent pairs (similar to Aho-Corasick failure links). This scans the text in linear $O(N)$ time with zero heap allocations.
*   **BPE Queue Complexity Reduction**: For BPE models, splitting a long sentence into $k$ independent chunks reduces priority-queue merge complexity from $O(N \log N)$ to $O(N \log(N/k))$, while enabling trie-based vocabulary shortcut lookups that bypass the priority queue entirely for frequent subwords.

#### Application Example: Parallel Document Tokenization

Individual safe chunks are typically short (one word or punctuation span). Dispatching threads on every fine-grained chunk incurs scheduling and allocation overhead. Instead, you can use `CanSkipNormalization` and the callback overload of `PretokenizeAtSafeBoundaries` to coalesce adjacent safe chunks into coarse-grained **256 KB – 1 MB blocks** (sized to fit per-core L2/L3 cache and amortize thread dispatch overhead) with zero string copies:

```cpp
// 1. Check if normalization can be bypassed (zero-copy fast path)
std::string normalized_buf;
std::string_view target = document;
if (!processor.CanSkipNormalization(document)) {
  processor.Normalize(document, &normalized_buf);
  target = normalized_buf;
}

// 2. Coalesce fine-grained safe chunks into ~1 MB contiguous blocks (zero-copy)
constexpr size_t kTargetBlockSize = 1024 * 1024;  // 1 MB (256 KB - 1 MB recommended)
std::vector<std::string_view> blocks;
std::string_view current_block;

processor.PretokenizeAtSafeBoundaries(target, [&](std::string_view chunk) {
  if (current_block.empty()) {
    current_block = chunk;
  } else {
    current_block = std::string_view(
        current_block.data(),
        (chunk.data() + chunk.size()) - current_block.data());
  }
  if (current_block.size() >= kTargetBlockSize) {
    blocks.push_back(current_block);
    current_block = std::string_view();
  }
});
if (!current_block.empty()) {
  blocks.push_back(current_block);
}

// 3. Encode independent ~1 MB blocks concurrently across worker threads
std::vector<std::vector<int>> block_results(blocks.size());
#pragma omp parallel for schedule(dynamic)
for (size_t i = 0; i < blocks.size(); ++i) {
  processor.EncodeNormalized(blocks[i], &block_results[i]);
}
// Concatenate block_results into final token ID sequence...
```

### 6.4. Token Caching (`CachedSentencePieceLite`)

For workloads involving repeated phrases, common words, or batch inference pipelines, SentencePiece Lite provides `CachedSentencePieceLite` and `TokenCache`.

```cpp
#include "sentencepiece_lite.h"
#include "cached_sentencepiece_lite.h"

// 1. Initialize processor once (immutable, thread-safe across threads)
sentencepiece::lite::SentencePieceLiteProcessor processor(model_view);

// 2. Tokenize text using a thread-local token cache.
// Thread-Local Multi-Model Partitioning: TokenCache mixes the processor's memory
// address into its hash seed. A single thread_local TokenCache instance can therefore
// be safely shared across different SentencePieceLiteProcessor models within the same
// thread without key collisions or cache contamination.
void TokenizeWorker(const sentencepiece::lite::SentencePieceLiteProcessor& processor,
                    std::string_view text) {
  // 64-byte L1 cacheline-aligned 2-way set-associative cache (default: 1 MB, 32,768 slots)
  static thread_local sentencepiece::lite::TokenCache cache;

  std::vector<int> ids;
  sentencepiece::lite::StatusCode status =
      sentencepiece::lite::CachedSentencePieceLite::Encode(processor, cache, text, &ids);
  if (status == sentencepiece::lite::StatusCode::kOk) {
    // Process token sequence...
  }
}
```

---

## 7. Standalone Amalgamation Distribution (`dist/`)

SentencePiece Lite can be bundled into a **2-file standalone distribution** (inspired by SQLite), requiring no external libraries (FlatBuffers, Abseil, or Protobuf) to compile:

```
dist/
├── sentencepiece_lite.h   # Consolidated C++20 header (Public & Caching API)
└── sentencepiece_lite.cc  # Self-contained implementation (FlatBuffers & rapidhash inlined)
```

```bash
# 1. Generate the standalone amalgamation files (dist/sentencepiece_lite.{h,cc})
python3 lite/amalgamate.py
# Or via CMake: cmake --build build --target amalgamate

# 2. Compile directly with any modern C++20 compiler (zero external dependencies)
g++ -std=c++20 -O3 -Idist your_program.cc dist/sentencepiece_lite.cc -o your_program
```

---

## 8. Build and Testing

### 8.1. Building with CMake

```bash
# Configure and build
mkdir -p build && cd build
cmake .. -DSPM_BUILD_TEST=ON
cmake --build . -j$(nproc)

# Run tests (including core, cached, and standalone amalgamation tests)
ctest --output-on-failure
```

### 8.2. Building with Bazel

```bash
# Build core runtime libraries
bazel build //lite:sentencepiece_lite
bazel build //lite:cached_sentencepiece_lite

# Build model converter tool
bazel build //lite:spm_to_fb

# Run all unit tests
bazel test //lite:...
```
