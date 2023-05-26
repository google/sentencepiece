#include <malloc.h>

#ifdef TCMALLOC
#include <gperftools/malloc_extension.h>
#endif

#include <forward_list>

#include <absl/container/flat_hash_map.h>
#include <absl/strings/string_view.h>
#include <boost/sort/block_indirect_sort/block_indirect_sort.hpp>

#include "common.h"
#include "filesystem.h"
#include "init.h"
#include "util.h"

ABSL_FLAG(std::string, output, "", "Output file path.");
ABSL_FLAG(int, save_interval, 4,
          "Write the intermediate result on each n-th file.");
ABSL_FLAG(bool, already_sorted, false, "Skip sorting the sentences.");

struct SingleLinkedStringsWithFreq {
  explicit SingleLinkedStringsWithFreq(size_t chunk_size = (1 << 26))
      : chunk_pos_(0),
        chunk_size_(0),
        default_chunk_size_(chunk_size),
        allocated_(0),
        count_(0),
        head_(nullptr) {}

  char *insert_after(absl::string_view str, int64 freq, char *node) {
    auto size = str.size() + 1 + sizeof(int64) + sizeof(char *);
    if (chunk_size_ - chunk_pos_ >= size) {
      auto ptr = storage_.front().get() + chunk_pos_;
      memcpy(ptr, str.data(), str.size());
      ptr[str.size()] = 0;
      *reinterpret_cast<int64 *>(reinterpret_cast<intptr_t>(ptr + str.size() + 1)) = freq;
      if (node != nullptr) {
        auto next_slot = reinterpret_cast<char **>(node + strlen(node) + 1 + sizeof(int64));
        *reinterpret_cast<char **>(ptr + str.size() + 1 + sizeof(int64)) = *next_slot;
        *next_slot = ptr;
      } else {
        *reinterpret_cast<char **>(ptr + str.size() + 1 + sizeof(int64)) = head_;
        head_ = ptr;
      }
      chunk_pos_ += size;
      count_++;
      return ptr;
    } else {
      if (size > default_chunk_size_) {
        AllocateChunk(size);
      } else {
        AllocateChunk(default_chunk_size_);
      }
      return insert_after(str, freq, node);
    }
  }

  void clear() {
    chunk_pos_ = 0;
    chunk_size_ = 0;
    allocated_ = 0;
    count_ = 0;
    head_ = nullptr;
    storage_.clear();
  }

  char *head() const noexcept { return head_; }
  size_t allocated() const noexcept { return allocated_; }
  size_t size() const noexcept { return count_; }

 private:
  void AllocateChunk(size_t size) {
    storage_.emplace_front(std::make_unique<char[]>(size));
    chunk_pos_ = 0;
    chunk_size_ = size;
    allocated_ += size;
  }

  std::forward_list<std::unique_ptr<char[]>> storage_;
  size_t chunk_pos_;
  size_t chunk_size_;
  size_t default_chunk_size_;
  size_t allocated_;
  size_t count_;
  char *head_;
};

struct CachedSentences {
  absl::flat_hash_map<char32, int64> required_chars;
  SingleLinkedStringsWithFreq sentences;
  size_t sentences_size;

  sentencepiece::util::Status MergeFromFile(
      const std::string &file, bool already_sorted) {
    auto reader = sentencepiece::filesystem::NewReadableFile(file, true, 0);
    if (reader->status() != sentencepiece::util::OkStatus()) {
      return reader->status();
    }
    std::string freq(sizeof(int64), 0),
                size(sizeof(size_t), 0),
                rcp(sizeof(char32) + sizeof(int64), 0);
    absl::string_view token;
    size_t required_chars_size;
    reader->ReadBuffer(&size);
    required_chars_size = *reinterpret_cast<size_t *>(size.data());
    for (size_t i = 0; i < required_chars_size; i++) {
      if (!reader->ReadBuffer(&rcp)) {
        return reader->status();
      }
      required_chars[*reinterpret_cast<char32 *>(rcp.data())] +=
          *reinterpret_cast<int64 *>(rcp.data() + 4);
    }
    LOG(INFO) << "Read " << required_chars_size << " required chars";

    size_t sentences_read = 0, sentences_existing = 0;
    sentences_size = 0;
    char *it = sentences.head(), *prev = nullptr;
    if (already_sorted) {
      LOG(INFO) << "Joining sorted lists...";
      while (reader->ReadLine(&token) && reader->ReadBuffer(&freq)) {
        if (++sentences_read % 10000000 == 0) {
          LOG(INFO) << "Read " << sentences_read << " sentences";
        }
        auto freq_val = *reinterpret_cast<const int64 *>(freq.data());
        while (true) {
          int cmp = 1;
          if (it != nullptr) {
            cmp = strncmp(it, token.data(), token.size());
            if (cmp == 0) {
              cmp = strlen(it) - token.size();
            }
          }
          if (cmp == 0) {
            auto &it_freq = *reinterpret_cast<int64 *>(it + strlen(it) + 1);
      	    it_freq += freq_val;
      	    sentences_existing++;
      	    break;
          } else if (cmp > 0) {
            prev = sentences.insert_after(token, freq_val, prev);
      	    if (++sentences_size % 10000000 == 0) {
      	      LOG(INFO) << "Descended " << sentences_size << " sentences, "
                        << sentences_existing << " existing";
      	    }
      	    break;
          }
          prev = it;
      	  if (++sentences_size % 10000000 == 0) {
      	    LOG(INFO) << "Descended " << sentences_size << " sentences, "
                      << sentences_existing << " existing";
      	  }
      	  it = *reinterpret_cast<char **>(it + strlen(it) + 1 + sizeof(int64));
        }
      }
    } else {
      std::vector<std::pair<absl::string_view, int64>> loaded;
      while (reader->ReadLine(&token) && reader->ReadBuffer(&freq)) {
        loaded.emplace_back(token, *reinterpret_cast<const int64 *>(freq.data()));
        if (++sentences_read % 10000000 == 0) {
      	  LOG(INFO) << "Read " << sentences_read << " sentences";
        }
      }
      LOG(INFO) << "Read " << sentences_read << " sentences, sorting...";
      boost::sort::block_indirect_sort(
          loaded.begin(), loaded.end(),
          std::thread::hardware_concurrency());
      LOG(INFO) << "Joining sorted lists...";
      for (auto &s : loaded) {
        while (true) {
          int cmp = 1;
          if (it != nullptr) {
            cmp = strncmp(it, s.first.data(), s.first.size());
            if (cmp == 0) {
              cmp = strlen(it) - s.first.size();
            }
          }
          if (cmp == 0) {
            auto &it_freq = *reinterpret_cast<int64 *>(it + strlen(it) + 1);
      	    it_freq += s.second;
      	    sentences_existing++;
      	    break;
          } else if (cmp > 0) {
            prev = sentences.insert_after(s.first, s.second, prev);
      	    if (++sentences_size % 10000000 == 0) {
      	      LOG(INFO) << "Descended " << sentences_size << " sentences, "
                        << sentences_existing << " existing";
      	    }
      	    break;
          }
          prev = it;
      	  if (++sentences_size % 10000000 == 0) {
      	    LOG(INFO) << "Descended " << sentences_size << " sentences, "
                      << sentences_existing << " existing";
      	  }
      	  it = *reinterpret_cast<char **>(it + strlen(it) + 1 + sizeof(int64));
        }
      }
    }
    LOG(INFO) << "Descended " << sentences.size() << " sentences, "
              << sentences_existing << " existing";
    return reader->status();
  }

  sentencepiece::util::Status WriteToFile(const std::string &file) {
    auto writer = sentencepiece::filesystem::NewWritableFile(file, true);
    if (writer->status() != sentencepiece::util::OkStatus()) {
      return writer->status();
    }
    auto required_chars_size = required_chars.size();
    writer->Write(absl::string_view(
        reinterpret_cast<char *>(&required_chars_size),
        sizeof(required_chars_size)));
    for (auto &c : required_chars) {
      char buf[4 + sizeof(int64)];
      *reinterpret_cast<char32 *>(buf) = c.first;
      *reinterpret_cast<int64 *>(buf + 4) = c.second;
      writer->Write(absl::string_view(buf, sizeof(buf)));
    }
    LOG(INFO) << "Wrote " << required_chars.size() << " required chars";
    size_t written_count = 0;
    auto it = sentences.head();
    size_t prefix_len = (it != nullptr)? (strlen(it) + 1 + sizeof(int64)) : 0;
    for (; it != nullptr;
         it = *reinterpret_cast<char **>(it + prefix_len),
         prefix_len = (it != nullptr)? (strlen(it) + 1 + sizeof(int64)) : 0) {
      if (!writer->Write(absl::string_view(it, prefix_len))) {
        return writer->status();
      }
      if (++written_count % 10000000 == 0) {
        LOG(INFO) << "Wrote " << written_count << " sentences";
      }
    }
    return writer->status();
  }

  void Sort() {
    std::vector<std::pair<absl::string_view, int64>> v;
    auto it = sentences.head();
    size_t prefix_len = (it != nullptr)? (strlen(it) + 1 + sizeof(int64)) : 0;
    for (; it != nullptr;
         it = *reinterpret_cast<char **>(it + prefix_len),
         prefix_len = (it != nullptr)? (strlen(it) + 1 + sizeof(int64)) : 0) {
      auto it_len = prefix_len - sizeof(int64);
      v.emplace_back(absl::string_view(it, it_len - 1),
                     *reinterpret_cast<int64 *>(it + it_len));
    }
    boost::sort::block_indirect_sort(
      v.begin(), v.end(),
      [](const auto &p1, const auto &p2) {
        return (p1.second > p2.second ||
               (p1.second == p2.second && p1.first < p2.first));
      },
      std::thread::hardware_concurrency());
    auto new_sentences = SingleLinkedStringsWithFreq();
    char *head = nullptr;
    for (auto &s : v) {
      head = new_sentences.insert_after(s.first, s.second, head);
    }
    sentences = std::move(new_sentences);
  }

  size_t allocated() const noexcept { return sentences.allocated(); }
};

int main(int argc, char *argv[]) {
  sentencepiece::ScopedResourceDestructor cleaner;
  sentencepiece::ParseCommandLineFlags(argv[0], &argc, &argv, true);
  auto output = absl::GetFlag(FLAGS_output);
  auto save_interval = absl::GetFlag(FLAGS_save_interval);
  auto already_sorted = absl::GetFlag(FLAGS_already_sorted);
  if (output.empty()) {
    LOG(FATAL) << "Must specify --output file";
    return 1;
  }
  std::vector<std::string> inputs;
  for (int i = 1; i < argc; ++i) {
    inputs.emplace_back(argv[i]);
  }
  if (inputs.empty()) {
    LOG(FATAL) << "Must specify at least one input file";
    return 2;
  }
  
  CachedSentences merged;
  int merged_count = 0;
  for (auto &file : inputs) {
    LOG(INFO) << "Merging with " << file;
    CHECK_OK(merged.MergeFromFile(file, already_sorted));
    LOG(INFO) << merged.sentences_size << " sentences, "
              << merged.required_chars.size() << " chars; allocated "
              << merged.allocated();
    #ifdef TCMALLOC
    MallocExtension::instance()->ReleaseFreeMemory();
    #endif
    malloc_stats();
    if (++merged_count % save_interval == 0 && inputs.size() > 1) {
      LOG(INFO) << "Writing to " << output;
      CHECK_OK(merged.WriteToFile(output));
    }
  }
  if (inputs.size() == 1) {
    LOG(INFO) << "Sorting in ascending frequency order...";
    merged.Sort();
  }
  if (inputs.size() == 1 || merged_count % save_interval != 0) {
    LOG(INFO) << "Writing to " << output;
    CHECK_OK(merged.WriteToFile(output));
  }
  return 0;
}