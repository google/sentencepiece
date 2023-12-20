// Copyright 2016 Google Inc.
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
// limitations under the License.!

#include <cassert>
#include <functional>
#include <string>
#include <vector>

#include "common.h"
#include "filesystem.h"
#include "init.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_processor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "trainer_interface.h"
#include "util.h"
#include "mixed_text_code_handler.h"

ABSL_FLAG(std::string, model, "", "model file name");
ABSL_FLAG(
    std::string, output_format, "piece",
    "choose from piece, id, poolside, proto, nbest_piece, nbest_id, or nbest_proto");
ABSL_FLAG(std::string, input, "", "input filename");
ABSL_FLAG(std::string, output, "", "output filename");
ABSL_FLAG(std::string, extra_options, "",
          "':' separated encoder extra options, e.g., \"reverse:bos:eos\"");
ABSL_FLAG(int32, nbest_size, 10, "NBest size");
ABSL_FLAG(double, alpha, 0.5, "Smoothing parameter for sampling mode.");
ABSL_FLAG(uint32, random_seed, ~0u,
          "Seed value for random generator.");
ABSL_FLAG(int, num_threads, 4,
          "Number of CPU threads to use for the encoding procedure.");
ABSL_FLAG(int, new_line_delim, '\n', "Sentence delimiter char.");


// Piece restriction with vocabulary file.
// https://github.com/rsennrich/subword-nmt#best-practice-advice-for-byte-pair-encoding-in-nmt
ABSL_FLAG(std::string, vocabulary, "",
          "Restrict the vocabulary. The encoder only emits the "
          "tokens in \"vocabulary\" file");
ABSL_FLAG(int32, vocabulary_threshold, 0,
          "Words with frequency < threshold will be treated as OOV");
ABSL_FLAG(bool, generate_vocabulary, false,
          "Generates vocabulary file instead of segmentation");

// Blocks delimiters.
ABSL_FLAG(int, verbatim_control_char, -2, "Control character to process the "
          "following sentence without whitespace normalization."
          "-1 to disable, -2 to use the value from the model");
ABSL_FLAG(int, code_block_end, -2,
          "Control character at the end of each code block."
          "-1 to disable, -2 to use the value from the model");
ABSL_FLAG(int, code_meta_block_begin, -2,
          "Control character at the beginning of each code meta block."
          "-1 to disable, -2 to use the value from the model");
ABSL_FLAG(int, code_meta_block_end, -2,
          "Control character at the end of each code meta block."
          "-1 to disable, -2 to use the value from the model");

#define ReadBlockDelimiter(name)\
  int32 name = absl::GetFlag(FLAGS_##name);\
  if (name == -2) name = sp.model_proto()->trainer_spec().name();

int main(int argc, char *argv[]) {
  std::atomic<int64_t> pending_size {0};
  sentencepiece::ScopedResourceDestructor cleaner;
  sentencepiece::ParseCommandLineFlags(argv[0], &argc, &argv, true);
  std::vector<std::string> rest_args;

  if (absl::GetFlag(FLAGS_input).empty()) {
    for (int i = 1; i < argc; ++i) {
      rest_args.emplace_back(argv[i]);
    }
  } else {
    rest_args.push_back(absl::GetFlag(FLAGS_input));
  }

  if (absl::GetFlag(FLAGS_random_seed) != ~0u) {
    sentencepiece::SetRandomGeneratorSeed(absl::GetFlag(FLAGS_random_seed));
  }

  if (rest_args.empty())
    rest_args.push_back("");  // empty means that reading from stdin.

  CHECK(!absl::GetFlag(FLAGS_model).empty());

  sentencepiece::SentencePieceProcessor sp;
  CHECK_OK(sp.Load(absl::GetFlag(FLAGS_model)));
  CHECK_OK(sp.SetEncodeExtraOptions(absl::GetFlag(FLAGS_extra_options)));

  if (!absl::GetFlag(FLAGS_vocabulary).empty()) {
    CHECK_OK(sp.LoadVocabulary(absl::GetFlag(FLAGS_vocabulary),
                               absl::GetFlag(FLAGS_vocabulary_threshold)));
  }

  auto output =
      sentencepiece::filesystem::NewWritableFile(absl::GetFlag(FLAGS_output));
  CHECK_OK(output->status());
  absl::flat_hash_map<std::string, int> vocab;
  sentencepiece::SentencePieceText spt;
  sentencepiece::NBestSentencePieceText nbest_spt;
  std::function<void(absl::string_view line)> process = nullptr;
  std::vector<uint32_t> sentence_sizes;
  ReadBlockDelimiter(verbatim_control_char);
  ReadBlockDelimiter(code_block_end);
  ReadBlockDelimiter(code_meta_block_begin);
  ReadBlockDelimiter(code_meta_block_end);
  int num_threads = absl::GetFlag(FLAGS_num_threads);
  sentencepiece::ThreadPool pool(num_threads);
  std::mutex sync;
  constexpr int thread_chunk_size = 1000;
  constexpr int64_t pending_limit = 1ll << 31;

  const int nbest_size = absl::GetFlag(FLAGS_nbest_size);
  const float alpha = absl::GetFlag(FLAGS_alpha);

  char delim = absl::GetFlag(FLAGS_new_line_delim);

  if (absl::GetFlag(FLAGS_generate_vocabulary)) {
    process = [&](absl::string_view line) {
      sentencepiece::SentencePieceText spt;
      CHECK_OK(sp.Encode(line, &spt));
      for (const auto &piece : spt.pieces()) {
        if (!sp.IsUnknown(piece.id()) && !sp.IsControl(piece.id()))
          vocab[piece.piece()]++;
      }
    };
  } else if (absl::GetFlag(FLAGS_output_format) == "piece") {
    process = [&](absl::string_view line) {
      std::vector<std::string> sps;
      CHECK_OK(sp.Encode(line, &sps));
      std::lock_guard lock(sync);
      output->Write(absl::StrJoin(sps, " "));
      output->Write(std::string(1, delim));
    };
  } else if (absl::GetFlag(FLAGS_output_format) == "id") {
    process = [&](absl::string_view line) {
      std::vector<int> ids;
      CHECK_OK(sp.Encode(line, &ids));
      std::lock_guard lock(sync);
      output->WriteLine(absl::StrJoin(ids, " "));
    };
  } else if (absl::GetFlag(FLAGS_output_format) == "proto") {
    process = [&](absl::string_view line) { CHECK_OK(sp.Encode(line, &spt)); };
  } else if (absl::GetFlag(FLAGS_output_format) == "sample_piece") {
    process = [&](absl::string_view line) {
      std::vector<std::string> sps;
      CHECK_OK(sp.SampleEncode(line, nbest_size, alpha, &sps));
      std::lock_guard lock(sync);
      output->WriteLine(absl::StrJoin(sps, " "));
    };
  } else if (absl::GetFlag(FLAGS_output_format) == "sample_id") {
    process = [&](absl::string_view line) {
      std::vector<int> ids;
      CHECK_OK(sp.SampleEncode(line, nbest_size, alpha, &ids));
      std::lock_guard lock(sync);
      output->WriteLine(absl::StrJoin(ids, " "));
    };
  } else if (absl::GetFlag(FLAGS_output_format) == "sample_proto") {
    process = [&](absl::string_view line) {
      CHECK_OK(sp.SampleEncode(line, nbest_size, alpha, &spt));
    };
  } else if (absl::GetFlag(FLAGS_output_format) == "nbest_piece") {
    process = [&](absl::string_view line) {
      std::vector<std::vector<std::string>> nbest_sps;
      CHECK_OK(sp.NBestEncode(line, nbest_size, &nbest_sps));
      std::lock_guard lock(sync);
      for (const auto &result : nbest_sps) {
        output->WriteLine(absl::StrJoin(result, " "));
      }
    };
  } else if (absl::GetFlag(FLAGS_output_format) == "nbest_id") {
    process = [&](absl::string_view line) {
      std::vector<std::vector<int>> nbest_ids;
      CHECK_OK(sp.NBestEncode(line, nbest_size, &nbest_ids));
      std::lock_guard lock(sync);
      for (const auto &result : nbest_ids) {
        output->WriteLine(absl::StrJoin(result, " "));
      }
    };
  } else if (absl::GetFlag(FLAGS_output_format) == "nbest_proto") {
    process = [&](absl::string_view line) {
      CHECK_OK(sp.NBestEncode(line, nbest_size, &nbest_spt));
    };
  } else if (absl::GetFlag(FLAGS_output_format) == "poolside") {
      /**
       * poolside_process will split the document into text and code blocks.
       *
       * Documents with code blocks
       *
       * Preprocess: Separate blocks by 0x01 and 0x02.
       *   From:
       *   [bytes(<some text>), 0x01, bytes(<some code>), 0x02, bytes(<some text>), 0x00]
       *   To:
       *   [bytes(<some text>),
       *    0x01, bytes(<some code>),
       *    bytes(<some text>)]
       *
       *   Please note that a single document containing both NL and PL should be splitted
       *   into multiple documents containing a single type of text (NL or PL) only.
       *   This is due to the fact that 0x01 is only checked at the beginning of a document.
       *
       * Post-process: Join blocks with 0x02, append 0x00.
       *   From:
       *   [bytes(<some text tokens>),
       *    0x01, bytes(<some code tokens>),
       *    bytes(<some text tokens>)]
       *   To:
       *   [bytes(<some text tokens>),
       *    0x01, bytes(<some code tokens>), 0x02,
       *    bytes(<some text tokens>), 0x00]
       *
       *
       * Singular code file with metadata
       *
       * Preprocess: Separate blocks by 0x01 and 0x02. Remove 0x03, 0x04.
       *   From:
       *   [0x03, bytes(“File path: x.cpp\n”), 0x04, 0x01, bytes(<file content>), 0x02, 0x00]
       *   To:
       *   [bytes(“File path: x.cpp\n”),
       *    0x01, bytes(<file content>)]
       *
       * Post process: Add 0x03, 0x04, and append the 0x00.
       *   From:
       *   [bytes(“File path : x . cpp \n”),
       *    0x01, bytes(<file content tokens>)]
       *   To:
       *   [0x03, bytes(“File path : x . cpp \n”), 0x04,
       *    0x01, bytes(<file content tokens>), 0x02, 0x00]
      */

    auto ps_code_start = sp.bos_id();
    auto ps_code_end = sp.PieceToId("<0x00>");
    auto ps_code_meta_start = sp.PieceToId("<0x01>");
    auto ps_code_meta_end = sp.PieceToId("<0x02>");
    auto ps_doc_end = sp.eos_id();

    process = [&](absl::string_view line) {
      sentencepiece::MixedTextCodeIterator blocks_iterator(line,
        verbatim_control_char,
        code_block_end,
        code_meta_block_begin,
        code_meta_block_end);

      std::vector<int> ids;
      absl::string_view block;
      while (blocks_iterator.HasNext()) {
        auto r = blocks_iterator.Next(&block);
        if (!r.has_value()) {
          // Line ends with an emtpy document.
          break;
        }
        switch(*r) {
          case sentencepiece::MixedTextCodeIterator::BlockType::Text:
            CHECK_OK(sp.Encode(block, &ids, false));
            break;
          case sentencepiece::MixedTextCodeIterator::BlockType::Code:
            ids.push_back(ps_code_start);
            CHECK_OK(sp.Encode(block, &ids, false));
            ids.push_back(ps_code_end);
            break;
          case sentencepiece::MixedTextCodeIterator::BlockType::CodeHeader:
            ids.push_back(ps_code_meta_start);
            CHECK_OK(sp.Encode(block, &ids, false));
            ids.push_back(ps_code_meta_end);
            break;
          default:
            LOG(FATAL) << "Unrecognized BlockType met during encoding.";
        }
      }
      ids.push_back(ps_doc_end);
      {
        std::lock_guard lock(sync);
        output->Write(absl::string_view(
            reinterpret_cast<const char *>(ids.data()), sizeof(uint32_t) * ids.size()));
        sentence_sizes.push_back(ids.size() - 1); // do not count the trailing eos
      }
    };
  } else {
    LOG(FATAL) << "Unknown output format: "
               << absl::GetFlag(FLAGS_output_format);
  }

  std::atomic<int64_t> processed {0};
  auto process_chunk = [&pool, &process, &processed, &pending_size](
      std::vector<sentencepiece::filesystem::ps_string>& chunk) {
    pool.Schedule([&process, &processed, chunk, &pending_size](){
      size_t size = 0;
      for (auto &line : chunk) {
        if (auto sv = std::get_if<absl::string_view>(&line); sv != nullptr) {
          size += sv->length();
          process(*sv);
        } else {
          auto& data = std::get<std::shared_ptr<std::string>>(line);
          size += data->length();
          process(*data);
        }
      }
      int64_t prev = processed.fetch_add(chunk.size()) + chunk.size();
      if ((prev / thread_chunk_size) % 100 == 0) {
        LOG(INFO) << "Encoded " << prev << " sentences";
      }
      pending_size -= size;
    });
    chunk.clear();
  };

  for (const auto &filename : rest_args) {
    auto input = sentencepiece::filesystem::NewReadableFile(
        filename, delim != '\n', delim);
    CHECK_OK(input->status());
    std::vector<sentencepiece::filesystem::ps_string> chunk;
    chunk.reserve(thread_chunk_size);
    sentencepiece::filesystem::ps_string line;
    while (input->ReadLineStdin(&line)) {
      chunk.emplace_back(line);
      if (auto sv = std::get_if<absl::string_view>(&line); sv != nullptr) {
        pending_size += sv->length();
      } else {
        auto& data = std::get<std::shared_ptr<std::string>>(line);
        pending_size += data->length();
      }
      if (chunk.size() == thread_chunk_size) {
        process_chunk(chunk);
        if (pending_size.load() >= pending_limit) {
          LOG(INFO) << "Throttled input at " << pending_size.load() << " pending bytes";
          // busy loop to one half of the queue size
          while (pending_size.load() > pending_limit / 2) {
            usleep(0);  // actually works instead of pthread_yield()
          }
        }
      }
    }
    if (chunk.size() > 0) {
      process_chunk(chunk);
    }
    pool.Wait();
    LOG(INFO) << "Encoded " << processed.load() << " sentences";
  }

  if (absl::GetFlag(FLAGS_output_format) == "poolside") {
    size_t count = sentence_sizes.size();
    output->Write(absl::string_view(reinterpret_cast<char *>(sentence_sizes.data()),
                  sizeof(sentence_sizes[0]) * sentence_sizes.size()));
    output->Write(absl::string_view(reinterpret_cast<char *>(&count), sizeof(count)));
  }

  if (absl::GetFlag(FLAGS_generate_vocabulary)) {
    for (const auto &it : sentencepiece::Sorted(vocab, num_threads)) {
      output->WriteLine(it.first + "\t" +
                        sentencepiece::string_util::SimpleItoa(it.second));
    }
  }

  return 0;
}
