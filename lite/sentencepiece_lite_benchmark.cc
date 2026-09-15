// Copyright 2026 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "cached_sentencepiece_lite.h"
#include "sentencepiece_lite.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_model_converters.h"

namespace sentencepiece {
namespace lite {
namespace {

std::string GetFilePath(std::string_view path) {
  static constexpr std::string_view kPrefixes[] = {
      "",
      "data/",
      "../data/",
      "../../data/",
  };
  for (const auto prefix : kPrefixes) {
    const std::string full_path = std::string(prefix) + std::string(path);
    if (std::ifstream(full_path, std::ios::binary).good()) {
      return full_path;
    }
  }
  return std::string(path);
}

::sentencepiece::ModelProto LoadModelProto(std::string_view filename) {
  std::string full_path = GetFilePath(filename);
  std::ifstream input(full_path, std::ios::binary);
  if (!input) {
    std::cerr << "Failed to open model file: " << full_path << "\n";
    std::exit(1);
  }
  std::string serialized((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  ::sentencepiece::ModelProto model_proto;
  if (!model_proto.ParseFromString(serialized)) {
    std::cerr << "Failed to parse ModelProto from: " << full_path << "\n";
    std::exit(1);
  }
  return model_proto;
}

std::string LoadInput(std::string_view filename) {
  std::string full_path = GetFilePath(filename);
  std::ifstream input(full_path, std::ios::binary);
  if (!input) {
    std::cerr << "Failed to open input file: " << full_path << "\n";
    std::exit(1);
  }
  return std::string((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
}

struct BenchmarkContext {
  std::string input;
  ::sentencepiece::ModelProto model_proto;
  std::string flatbuffer_model;
  std::vector<int> lite_ids;
};

BenchmarkContext* GetContext(std::string_view model_filename,
                             std::string_view input_filename) {
  static absl::NoDestructor<absl::flat_hash_map<
      std::pair<std::string, std::string>, BenchmarkContext>>
      cache;
  auto key =
      std::make_pair(std::string(model_filename), std::string(input_filename));
  auto [it, inserted] = cache->try_emplace(key);
  if (!inserted) {
    return &it->second;
  }

  BenchmarkContext& ctx = it->second;
  ctx.model_proto = LoadModelProto(model_filename);
  ctx.input = LoadInput(input_filename);

  // Convert to flatbuffer and serialize
  auto fb_model_or = ToFlatbuffer(ctx.model_proto);
  if (!fb_model_or.ok()) {
    std::cerr << "Failed to convert model to FlatBuffer: "
              << fb_model_or.status().message() << "\n";
    std::exit(1);
  }
  ctx.flatbuffer_model = std::move(fb_model_or.value());

  auto model_ptr = std::make_shared<std::string>(ctx.flatbuffer_model);
  SentencePieceLiteProcessor lite_processor(model_ptr);
  if (lite_processor.status() != StatusCode::kOk) {
    std::cerr << "Failed to initialize SentencePieceLiteProcessor\n";
    std::exit(1);
  }
  lite_processor.Encode(ctx.input, &ctx.lite_ids);

  return &ctx;
}

void BM_Lite_Encode(benchmark::State& state, std::string_view model_filename,
                    std::string_view input_filename) {
  BenchmarkContext* ctx = GetContext(model_filename, input_filename);
  auto model_ptr = std::make_shared<std::string>(ctx->flatbuffer_model);
  SentencePieceLiteProcessor processor(model_ptr);

  std::vector<int> ids;
  for (auto _ : state) {
    benchmark::DoNotOptimize(ctx->input);
    StatusCode result = processor.Encode(ctx->input, &ids);
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * ctx->input.size());
}

void BM_Lite_Cached_Encode(benchmark::State& state,
                           std::string_view model_filename,
                           std::string_view input_filename) {
  BenchmarkContext* ctx = GetContext(model_filename, input_filename);
  auto model_ptr = std::make_shared<std::string>(ctx->flatbuffer_model);
  SentencePieceLiteProcessor processor(model_ptr);
  TokenCache cache;

  std::vector<int> ids;
  for (auto _ : state) {
    benchmark::DoNotOptimize(ctx->input);
    StatusCode result =
        CachedSentencePieceLite::Encode(processor, cache, ctx->input, &ids);
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * ctx->input.size());
}

void BM_Lite_Decode(benchmark::State& state, std::string_view model_filename,
                    std::string_view input_filename) {
  BenchmarkContext* ctx = GetContext(model_filename, input_filename);
  auto model_ptr = std::make_shared<std::string>(ctx->flatbuffer_model);
  SentencePieceLiteProcessor processor(model_ptr);

  std::string output;
  for (auto _ : state) {
    benchmark::DoNotOptimize(ctx->lite_ids);
    StatusCode result = processor.Decode(ctx->lite_ids, &output);
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * ctx->input.size());
}

void BM_Lite_Normalize(benchmark::State& state, std::string_view model_filename,
                       std::string_view input_filename) {
  BenchmarkContext* ctx = GetContext(model_filename, input_filename);
  auto model_ptr = std::make_shared<std::string>(ctx->flatbuffer_model);
  SentencePieceLiteProcessor processor(model_ptr);

  std::string normalized;
  for (auto _ : state) {
    benchmark::DoNotOptimize(ctx->input);
    StatusCode result = processor.Normalize(ctx->input, &normalized);
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * ctx->input.size());
}

void BM_Lite_Pretokenize(benchmark::State& state,
                         std::string_view model_filename,
                         std::string_view input_filename) {
  BenchmarkContext* ctx = GetContext(model_filename, input_filename);
  auto model_ptr = std::make_shared<std::string>(ctx->flatbuffer_model);
  SentencePieceLiteProcessor processor(model_ptr);

  std::string normalized_str;
  processor.Normalize(ctx->input, &normalized_str);

  std::vector<std::string_view> chunks;
  for (auto _ : state) {
    benchmark::DoNotOptimize(normalized_str);
    chunks.clear();
    StatusCode result =
        processor.PretokenizeAtSafeBoundaries(normalized_str, &chunks);
    benchmark::DoNotOptimize(result);
  }
  state.SetBytesProcessed(state.iterations() * normalized_str.size());
}

// Benchmark Cases
void BM_Botchan_Lite_Encode(benchmark::State& state) {
  BM_Lite_Encode(state, "botchan_en_bpe_1000.model", "botchan.txt");
}
BENCHMARK(BM_Botchan_Lite_Encode);

void BM_Botchan_Lite_Cached_Encode(benchmark::State& state) {
  BM_Lite_Cached_Encode(state, "botchan_en_bpe_1000.model", "botchan.txt");
}
BENCHMARK(BM_Botchan_Lite_Cached_Encode);

void BM_Botchan_Lite_Decode(benchmark::State& state) {
  BM_Lite_Decode(state, "botchan_en_bpe_1000.model", "botchan.txt");
}
BENCHMARK(BM_Botchan_Lite_Decode);

void BM_Wagahai_Lite_Encode(benchmark::State& state) {
  BM_Lite_Encode(state, "wagahaiwa_nekodearu_ja_bpe_byte_2000.model",
                 "wagahaiwa_nekodearu.txt");
}
BENCHMARK(BM_Wagahai_Lite_Encode);

void BM_Wagahai_Lite_Cached_Encode(benchmark::State& state) {
  BM_Lite_Cached_Encode(state, "wagahaiwa_nekodearu_ja_bpe_byte_2000.model",
                        "wagahaiwa_nekodearu.txt");
}
BENCHMARK(BM_Wagahai_Lite_Cached_Encode);

void BM_Wagahai_Lite_Decode(benchmark::State& state) {
  BM_Lite_Decode(state, "wagahaiwa_nekodearu_ja_bpe_byte_2000.model",
                 "wagahaiwa_nekodearu.txt");
}
BENCHMARK(BM_Wagahai_Lite_Decode);

void BM_Unigram_Lite_Encode(benchmark::State& state) {
  BM_Lite_Encode(state, "botchan_en_unigram_1000.model", "botchan.txt");
}
BENCHMARK(BM_Unigram_Lite_Encode);

void BM_Unigram_Lite_Cached_Encode(benchmark::State& state) {
  BM_Lite_Cached_Encode(state, "botchan_en_unigram_1000.model", "botchan.txt");
}
BENCHMARK(BM_Unigram_Lite_Cached_Encode);

void BM_Unigram_Lite_Decode(benchmark::State& state) {
  BM_Lite_Decode(state, "botchan_en_unigram_1000.model", "botchan.txt");
}
BENCHMARK(BM_Unigram_Lite_Decode);

void BM_Wagahai_Lite_Normalize(benchmark::State& state) {
  BM_Lite_Normalize(state, "wagahaiwa_nekodearu_ja_bpe_byte_2000.model",
                    "wagahaiwa_nekodearu.txt");
}
BENCHMARK(BM_Wagahai_Lite_Normalize);

void BM_Wagahai_Lite_Pretokenize(benchmark::State& state) {
  BM_Lite_Pretokenize(state, "wagahaiwa_nekodearu_ja_bpe_byte_2000.model",
                      "wagahaiwa_nekodearu.txt");
}
BENCHMARK(BM_Wagahai_Lite_Pretokenize);

void BM_Botchan_Lite_Normalize(benchmark::State& state) {
  BM_Lite_Normalize(state, "botchan_en_bpe_1000.model", "botchan.txt");
}
BENCHMARK(BM_Botchan_Lite_Normalize);

void BM_Botchan_Lite_Pretokenize(benchmark::State& state) {
  BM_Lite_Pretokenize(state, "botchan_en_bpe_1000.model", "botchan.txt");
}
BENCHMARK(BM_Botchan_Lite_Pretokenize);

void BM_Gemma_Lite_Encode(benchmark::State& state) {
  BM_Lite_Encode(state, "gemma_tokenizer.model", "botchan.txt");
}
BENCHMARK(BM_Gemma_Lite_Encode);

void BM_Gemma_Lite_Cached_Encode(benchmark::State& state) {
  BM_Lite_Cached_Encode(state, "gemma_tokenizer.model", "botchan.txt");
}
BENCHMARK(BM_Gemma_Lite_Cached_Encode);

void BM_Gemma_Lite_Decode(benchmark::State& state) {
  BM_Lite_Decode(state, "gemma_tokenizer.model", "botchan.txt");
}
BENCHMARK(BM_Gemma_Lite_Decode);

void BM_Gemma_Wagahai_Lite_Encode(benchmark::State& state) {
  BM_Lite_Encode(state, "gemma_tokenizer.model", "wagahaiwa_nekodearu.txt");
}
BENCHMARK(BM_Gemma_Wagahai_Lite_Encode);

void BM_Gemma_Wagahai_Lite_Cached_Encode(benchmark::State& state) {
  BM_Lite_Cached_Encode(state, "gemma_tokenizer.model",
                        "wagahaiwa_nekodearu.txt");
}
BENCHMARK(BM_Gemma_Wagahai_Lite_Cached_Encode);

void BM_Gemma_Wagahai_Lite_Decode(benchmark::State& state) {
  BM_Lite_Decode(state, "gemma_tokenizer.model", "wagahaiwa_nekodearu.txt");
}
BENCHMARK(BM_Gemma_Wagahai_Lite_Decode);

void BM_Gemma_Lite_Normalize(benchmark::State& state) {
  BM_Lite_Normalize(state, "gemma_tokenizer.model", "botchan.txt");
}
BENCHMARK(BM_Gemma_Lite_Normalize);

void BM_Gemma_Lite_Pretokenize(benchmark::State& state) {
  BM_Lite_Pretokenize(state, "gemma_tokenizer.model", "botchan.txt");
}
BENCHMARK(BM_Gemma_Lite_Pretokenize);

}  // namespace
}  // namespace lite
}  // namespace sentencepiece

BENCHMARK_MAIN();
