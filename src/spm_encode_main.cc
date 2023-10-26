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

ABSL_FLAG(std::string, model, "", "model file name");
ABSL_FLAG(
    std::string, output_format, "piece",
    "choose from piece, id, bid, proto, nbest_piece, nbest_id, or nbest_proto");
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

int main(int argc, char *argv[]) {
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
  std::function<void(absl::string_view line, std::vector<int>& ids)> process_ex = nullptr;
  std::vector<uint32_t> sentence_sizes;
  int num_threads = absl::GetFlag(FLAGS_num_threads);
  sentencepiece::ThreadPool pool(num_threads);
  std::recursive_mutex sync;
  constexpr int thread_chunk_size = 1000;

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
  } else if (absl::GetFlag(FLAGS_output_format) == "bid") {
    process_ex = [&](absl::string_view line, std::vector<int>& ids) {
      std::vector<int> _ids;
      CHECK_OK(sp.Encode(line, &_ids));
      ids.insert(ids.end(), _ids.begin(), _ids.end());
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
  } else {
    LOG(FATAL) << "Unknown output format: "
               << absl::GetFlag(FLAGS_output_format);
  }

  std::atomic<int64_t> processed = 0;

  const bool isBid = absl::GetFlag(FLAGS_output_format) == "bid";
  int ps_code_start, ps_code_end, ps_code_meta_start, ps_code_meta_end, ps_doc_end;
  if (isBid) {
    ps_code_start = sp.PieceToId("<0x01>");
    ps_code_end = sp.PieceToId("<0x02>");
    ps_code_meta_start = sp.PieceToId("<0x03>");
    ps_code_meta_end = sp.PieceToId("<0x04>");
    ps_doc_end = sp.PieceToId("<0x00>");
  }

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
  auto poolside_process_code_file_header = [&](absl::string_view line, std::vector<int>& ids) {
    auto head = line.data();
    auto tail = reinterpret_cast<const char *>(memchr(head, '\x04', line.length()));
    assert((void("Code meta block did not end with 0x04"), tail != nullptr));
    ids.push_back(ps_code_meta_start);
    process_ex(absl::string_view(head + 1, tail - head - 1), ids);
    ids.push_back(ps_code_meta_end);
    return tail + 1;
  };

  auto poolside_process = [&](absl::string_view line) {
    auto head = line.data();
    std::vector<int> ids;
    if (*head == '\x03') {
      head = poolside_process_code_file_header(line, ids);
    }
    auto end = head + line.length();
    auto tail = head;
    do {
      bool hasVerbatim = *head == '\x01';
      for (tail = hasVerbatim ? head + 1 : head; tail != end; tail++) {
        if (*tail == '\x01' || *tail == '\x02') {
          break;
        }
      }
      assert((void("Code block did not end with 0x02"), !hasVerbatim || (tail != end && *tail == '\x02')));
      assert((void("Text block did not end with 0x01 or 0x00"), hasVerbatim || tail == end || *tail == '\x01'));
      if (tail == end) {
        // Last regular text block
        process_ex(absl::string_view(head, line.length() - (head - line.data())), ids);
        break;
      } else if (*tail == '\x01') {
        // A regular text block
        process_ex(absl::string_view(head, tail - head), ids);
        // keep 0x01 to the next loop
      } else { // *tail == '\x02'
        // A code block
        ids.push_back(ps_code_start);
        process_ex(absl::string_view(head, tail - head), ids);
        // write 0x02
        ids.push_back(ps_code_end);
        // skip 0x02
        tail += 1;
      }
      head = tail;
    } while (tail != end);
    {
      ids.push_back(ps_doc_end);
      std::lock_guard lock(sync);
      output->Write(absl::string_view(
            reinterpret_cast<const char *>(ids.data()), sizeof(int) * ids.size()));
    }
    sentence_sizes.push_back(ids.size());
  };

  auto _process = isBid ? poolside_process : process;

  auto processChunk = [&pool, &_process, &processed](std::vector<absl::string_view>& chunk) {
    pool.Schedule([&_process, &processed, chunk](){
      for (auto &line : chunk) {
        _process(line);
      }
      int64_t prev = processed.fetch_add(chunk.size()) + chunk.size();
      if ((prev / thread_chunk_size) % 100 == 0) {
        LOG(INFO) << "Encoded " << prev << " sentences";
      }
    });
    chunk.clear();
  };

  for (const auto &filename : rest_args) {
    if (filename.empty()) {
      LOG(FATAL) << "Pipe input is not supported. Please use --input to specify the names of the input files";
      continue;
    }
    auto input = sentencepiece::filesystem::NewReadableFile(
        filename, delim != '\n', delim);
    CHECK_OK(input->status());
    std::vector<absl::string_view> chunk;
    chunk.reserve(thread_chunk_size);
    absl::string_view line;
    while (input->ReadLine(&line)) {
      chunk.emplace_back(line);
      if (chunk.size() == thread_chunk_size) {
        processChunk(chunk);
      }
    }
    if (chunk.size() > 0) {
      processChunk(chunk);
    }
    pool.Wait();
    LOG(INFO) << "Encoded " << processed.load() << " sentences";
  }

  if (absl::GetFlag(FLAGS_output_format) == "bid") {
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
