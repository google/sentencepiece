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
*   **Throughput**: Double-Array Trie lookups, bigram pre-tokenization, and zero-copy string views deliver up to 10.8x faster encoding and 9.3x faster decoding compared to the original Protobuf-based runtime.
*   **Amalgamation Support (2-File Standalone Distribution)**: Provides a consolidated 2-file distribution (`dist/sentencepiece_lite.h` and `dist/sentencepiece_lite.cc`) with all dependencies (FlatBuffers runtime and rapidhash) inlined. Drop these 2 files into any C++ project and compile directly with `-std=c++20` without external library linking.

## 3. Performance Evaluation & Resource Footprint

### 3.1. Single-Core Encoding Throughput

We evaluated single-thread encoding speed across **FineWeb2** datasets (5.0 MB English, 5.0 MB Japanese, and 8.9 MB Multilingual Mixed). We compared five configurations: [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite), **SentencePiece Lite**, **SentencePiece**, **Tiktoken (v0.14.0)**, and **Hugging Face Tokenizers (v0.23.1 Fast Mode)**.

#### (1) FineWeb2 English Corpus (5.0 MB)

| Engine | Model | Algorithm | Throughput | Tokens |
| :--- | :--- | :---: | :---: | :---: |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **Gemma 3 (256K)** | **BPE** | **30.64 MB/s** | 1,165,287 |
| **SentencePiece Lite** | **Gemma 3 (256K)** | **BPE** | **25.16 MB/s** | 1,165,287 |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **PLaMo-13B (64K)** | **Unigram** | **19.22 MB/s** | 1,388,013 |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **LLM-jp-4 (196K)** | **Unigram** | **17.93 MB/s** | 1,189,623 |
| **Tiktoken (v0.14.0)** | **o200k_base (200K)** | **BPE** | **12.33 MB/s** | 1,119,612 |
| **SentencePiece Lite** | **PLaMo-13B (64K)** | **Unigram** | **11.28 MB/s** | 1,387,931 |
| **SentencePiece Lite** | **LLM-jp-4 (196K)** | **Unigram** | **10.19 MB/s** | 1,189,623 |
| **SentencePiece** | **PLaMo-13B (64K)** | **Unigram** | **9.63 MB/s** | 1,410,863 |
| **Tiktoken (v0.14.0)** | **cl100k_base (100K)** | **BPE** | **8.42 MB/s** | 1,131,809 |
| **Tiktoken (v0.14.0)** | **Llama 3 (128K)** | **BPE** | **5.56 MB/s** | 1,131,445 |
| **SentencePiece** | **LLM-jp-4 (196K)** | **Unigram** | **8.22 MB/s** | 1,189,605 |
| **Hugging Face Tokenizers (v0.23.1)** | **Gemma 3 (256K)** | **BPE** | **1.37 MB/s** | 1,165,288* |
| **Hugging Face Tokenizers (v0.23.1)** | **Llama 3 (128K)** | **BPE** | **1.21 MB/s** | 1,131,446 |
| **Hugging Face Tokenizers (v0.23.1)** | **cl100k_base (100K)** | **BPE** | **1.10 MB/s** | 1,131,809 |
| **SentencePiece** | **Gemma 3 (256K)** | **BPE** | **1.07 MB/s** | 1,165,287 |
| **Hugging Face Tokenizers (v0.23.1)** | **PLaMo-13B (64K)** | **Unigram** | **0.97 MB/s** | 2,426,835 |
| **Hugging Face Tokenizers (v0.23.1)** | **LLM-jp-4 (196K)** | **Unigram** | **0.95 MB/s** | 1,189,542 |

#### (2) FineWeb2 Japanese Corpus (5.0 MB)

| Engine | Model | Algorithm | Throughput | Tokens |
| :--- | :--- | :---: | :---: | :---: |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **Gemma 3 (256K)** | **BPE** | **23.36 MB/s** | 1,040,163 |
| **SentencePiece Lite** | **Gemma 3 (256K)** | **BPE** | **21.98 MB/s** | 1,040,163 |
| **SentencePiece Lite** | **PLaMo-13B (64K)** | **Unigram** | **18.16 MB/s** | 1,001,575 |
| **SentencePiece Lite** | **LLM-jp-4 (196K)** | **Unigram** | **17.95 MB/s** | 816,103 |
| **SentencePiece** | **PLaMo-13B (64K)** | **Unigram** | **17.56 MB/s** | 1,005,451 |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **PLaMo-13B (64K)** | **Unigram** | **15.45 MB/s** | 1,001,577 |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **LLM-jp-4 (196K)** | **Unigram** | **14.28 MB/s** | 816,103 |
| **SentencePiece** | **LLM-jp-4 (196K)** | **Unigram** | **13.42 MB/s** | 816,119 |
| **Tiktoken (v0.14.0)** | **cl100k_base (100K)** | **BPE** | **6.95 MB/s** | 1,916,377 |
| **Tiktoken (v0.14.0)** | **o200k_base (200K)** | **BPE** | **6.72 MB/s** | 1,381,640 |
| **Tiktoken (v0.14.0)** | **Llama 3 (128K)** | **BPE** | **5.95 MB/s** | 1,283,796 |
| **SentencePiece** | **Gemma 3 (256K)** | **BPE** | **6.07 MB/s** | 1,040,163 |
| **Hugging Face Tokenizers (v0.23.1)** | **Gemma 3 (256K)** | **BPE** | **2.05 MB/s** | 1,040,163* |
| **Hugging Face Tokenizers (v0.23.1)** | **PLaMo-13B (64K)** | **Unigram** | **1.79 MB/s** | 1,003,605 |
| **Hugging Face Tokenizers (v0.23.1)** | **Llama 3 (128K)** | **BPE** | **1.45 MB/s** | 1,283,797 |
| **Hugging Face Tokenizers (v0.23.1)** | **LLM-jp-4 (196K)** | **Unigram** | **1.42 MB/s** | 816,213 |
| **Hugging Face Tokenizers (v0.23.1)** | **cl100k_base (100K)** | **BPE** | **1.17 MB/s** | 1,916,377 |

#### (3) FineWeb2 Multilingual Mixed Corpus (8.9 MB)

| Engine | Model | Algorithm | Throughput | Tokens |
| :--- | :--- | :---: | :---: | :---: |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **PLaMo-13B (64K)** | **Unigram** | **21.23 MB/s** | 4,599,607 |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **Gemma 3 (256K)** | **BPE** | **20.59 MB/s** | 1,749,723 |
| [**SentencePiece Lite (Cached)**](#64-token-caching-cachedsentencepiecelite) | **LLM-jp-4 (196K)** | **Unigram** | **19.25 MB/s** | 3,060,543 |
| **SentencePiece Lite** | **PLaMo-13B (64K)** | **Unigram** | **18.08 MB/s** | 4,599,631 |
| **SentencePiece Lite** | **Gemma 3 (256K)** | **BPE** | **17.05 MB/s** | 1,749,723 |
| **SentencePiece Lite** | **LLM-jp-4 (196K)** | **Unigram** | **16.68 MB/s** | 3,060,464 |
| **SentencePiece** | **PLaMo-13B (64K)** | **Unigram** | **13.28 MB/s** | 4,618,104 |
| **SentencePiece** | **LLM-jp-4 (196K)** | **Unigram** | **12.84 MB/s** | 3,061,458 |
| **Tiktoken (v0.14.0)** | **o200k_base (200K)** | **BPE** | **7.03 MB/s** | 1,818,536 |
| **Tiktoken (v0.14.0)** | **cl100k_base (100K)** | **BPE** | **6.57 MB/s** | 2,858,926 |
| **Tiktoken (v0.14.0)** | **Llama 3 (128K)** | **BPE** | **5.32 MB/s** | 2,018,277 |
| **SentencePiece** | **Gemma 3 (256K)** | **BPE** | **2.01 MB/s** | 1,749,723 |
| **Hugging Face Tokenizers (v0.23.1)** | **Gemma 3 (256K)** | **BPE** | **1.57 MB/s** | 1,749,724* |
| **Hugging Face Tokenizers (v0.23.1)** | **Llama 3 (128K)** | **BPE** | **1.05 MB/s** | 2,018,278 |
| **Hugging Face Tokenizers (v0.23.1)** | **PLaMo-13B (64K)** | **Unigram** | **0.87 MB/s** | 3,611,833 |
| **Hugging Face Tokenizers (v0.23.1)** | **LLM-jp-4 (196K)** | **Unigram** | **0.81 MB/s** | 3,056,266 |
| **Hugging Face Tokenizers (v0.23.1)** | **cl100k_base (100K)** | **BPE** | **0.64 MB/s** | 2,858,926 |

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
StatusCode PretokenizeAtSafeBoundaries(
    std::string_view normalized, std::vector<std::string_view>* out) const;
```

#### Algorithm & Implementation Details

*   **Character Bigram Co-occurrence Principle**: If a character bigram $(C_i, C_{i+1})$ never appears inside any subword token in the vocabulary, no subword piece can span across the boundary between $C_i$ and $C_{i+1}$. The input text can therefore be split safely at this boundary before encoding.
*   **AC-style State Reuse Traversal**: All valid character bigrams present in the vocabulary are compiled offline into a Double-Array Trie (`char_bigram_blob`). During input scanning, the runtime reuses single-character trie node state transitions across adjacent pairs (similar to Aho-Corasick failure links). This scans the text in linear $O(N)$ time with zero heap allocations.
*   **BPE Queue Complexity Reduction**: For BPE models, splitting a long sentence into $k$ independent chunks reduces priority-queue merge complexity from $O(N \log N)$ to $O(N \log(N/k))$, while enabling trie-based vocabulary shortcut lookups that bypass the priority queue entirely for frequent subwords.

#### Application Example: Parallel Document Tokenization

Because chunks split at safe boundaries are guaranteed to be independent, callers can tokenize document chunks concurrently across threads:

```cpp
// Pseudocode: Parallel document encoding via safe pre-tokenization
std::string normalized;
processor.Normalize(document, &normalized);

std::vector<std::string_view> chunks;
processor.PretokenizeAtSafeBoundaries(normalized, &chunks);

// Process independent chunks concurrently
std::vector<std::vector<int>> chunk_results(chunks.size());
#pragma omp parallel for
for (size_t i = 0; i < chunks.size(); ++i) {
  processor.EncodeNormalized(chunks[i], &chunk_results[i]);
}
// Concatenate chunk_results into final token ID sequence...
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
