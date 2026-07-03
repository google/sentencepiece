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

#include "util.h"

#include <atomic>
#include <cstddef>
#include <memory>

#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"

namespace sentencepiece {

namespace {
static constexpr uint32_t kDefaultSeed = static_cast<uint32_t>(-1);
static std::atomic<uint32_t> g_seed = kDefaultSeed;
static std::atomic<int> g_nbest_timeout_ms = 30000;
}  // namespace

void SetRandomGeneratorSeed(uint32_t seed) { g_seed.store(seed); }

uint32_t GetRandomGeneratorSeed() { return g_seed.load(); }

namespace {
std::shared_ptr<const std::string>* GetSharedDataDir() {
  static auto g_data_dir = std::make_shared<const std::string>(INSTALL_DATADIR);
  return &g_data_dir;
}
}  // namespace

std::string GetDataDir() {
  auto shared_data_dir = std::atomic_load(GetSharedDataDir());
  return *shared_data_dir;
}

void SetDataDir(absl::string_view data_dir) {
  auto shared_data_dir =
      std::make_shared<const std::string>(std::string(data_dir));
  std::atomic_store(GetSharedDataDir(), std::move(shared_data_dir));
}

void SetMinLogLevel(int v) {
  absl::SetMinLogLevel(static_cast<absl::LogSeverityAtLeast>(v));
}

void SetNBestTimeout(int timeout_ms) { g_nbest_timeout_ms.store(timeout_ms); }

int GetNBestTimeout() { return g_nbest_timeout_ms.load(); }

namespace string_util {

// mblen sotres the number of bytes consumed after decoding.
char32_t DecodeUTF8(const char* begin, const char* end, size_t* mblen) {
  const size_t len = end - begin;

  if (static_cast<unsigned char>(begin[0]) < 0x80) {
    *mblen = 1;
    return static_cast<unsigned char>(begin[0]);
  } else if (len >= 2 && (begin[0] & 0xE0) == 0xC0) {
    const char32_t cp = (((begin[0] & 0x1F) << 6) | ((begin[1] & 0x3F)));
    if (IsTrailByte(begin[1]) && cp >= 0x0080 && IsValidCodepoint(cp)) {
      *mblen = 2;
      return cp;
    }
  } else if (len >= 3 && (begin[0] & 0xF0) == 0xE0) {
    const char32_t cp = (((begin[0] & 0x0F) << 12) | ((begin[1] & 0x3F) << 6) |
                       ((begin[2] & 0x3F)));
    if (IsTrailByte(begin[1]) && IsTrailByte(begin[2]) && cp >= 0x0800 &&
        IsValidCodepoint(cp)) {
      *mblen = 3;
      return cp;
    }
  } else if (len >= 4 && (begin[0] & 0xf8) == 0xF0) {
    const char32_t cp = (((begin[0] & 0x07) << 18) | ((begin[1] & 0x3F) << 12) |
                       ((begin[2] & 0x3F) << 6) | ((begin[3] & 0x3F)));
    if (IsTrailByte(begin[1]) && IsTrailByte(begin[2]) &&
        IsTrailByte(begin[3]) && cp >= 0x10000 && IsValidCodepoint(cp)) {
      *mblen = 4;
      return cp;
    }
  }

  // Invalid UTF-8.
  *mblen = 1;
  return kUnicodeError;
}

bool IsStructurallyValid(absl::string_view str) {
  const char* begin = str.data();
  const char* end = str.data() + str.size();
  size_t mblen = 0;
  while (begin < end) {
    const char32_t c = DecodeUTF8(begin, end, &mblen);
    if (c == kUnicodeError && mblen != 3) return false;
    if (!IsValidCodepoint(c)) return false;
    begin += mblen;
  }
  return true;
}

size_t EncodeUTF8(char32_t c, char* output) {
  if (c <= 0x7F) {
    *output = static_cast<char>(c);
    return 1;
  }

  if (c <= 0x7FF) {
    output[1] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[0] = 0xC0 | c;
    return 2;
  }

  // if `c` is out-of-range, convert it to REPLACEMENT CHARACTER (U+FFFD).
  // This treatment is the same as the original runetochar.
  if (c > 0x10FFFF) c = kUnicodeError;

  if (c <= 0xFFFF) {
    output[2] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[1] = 0x80 | (c & 0x3F);
    c >>= 6;
    output[0] = 0xE0 | c;
    return 3;
  }

  output[3] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[2] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[1] = 0x80 | (c & 0x3F);
  c >>= 6;
  output[0] = 0xF0 | c;

  return 4;
}

std::string UnicodeCharToUTF8(const char32_t c) {
  return UnicodeTextToUTF8({c});
}

UnicodeText UTF8ToUnicodeText(absl::string_view utf8) {
  UnicodeText uc;
  const char* begin = utf8.data();
  const char* end = utf8.data() + utf8.size();
  while (begin < end) {
    size_t mblen;
    const char32_t c = DecodeUTF8(begin, end, &mblen);
    uc.push_back(c);
    begin += mblen;
  }
  return uc;
}

std::string UnicodeTextToUTF8(const UnicodeText& utext) {
  char buf[8];
  std::string result;
  for (const char32_t c : utext) {
    const size_t mblen = EncodeUTF8(c, buf);
    result.append(buf, mblen);
  }
  return result;
}

UnicodeTextAndOffsets UTF8ToUnicodeTextAndOffsets(absl::string_view utf8) {
  UnicodeTextAndOffsets ret;
  size_t running_offset = 0;
  ret.unicode_text.reserve(utf8.size());
  ret.offsets.reserve(utf8.size() + 1);
  ret.offsets.push_back(0);
  const char* begin = utf8.data();
  const char* end = utf8.data() + utf8.size();
  while (begin < end) {
    size_t mblen;
    const char32_t c = DecodeUTF8(begin, end, &mblen);
    running_offset += mblen;
    ret.unicode_text.push_back(c);
    ret.offsets.push_back(running_offset);
    begin += mblen;
  }
  return ret;
}

}  // namespace string_util

namespace random {
absl::BitGen* GetRandomGenerator() {
  // Thread-locals occupy stack space in every thread ever created by the
  // program, even if that thread never uses the thread-local variable.
  thread_local static auto mt =
      GetRandomGeneratorSeed() == kDefaultSeed
          ? std::make_unique<absl::BitGen>()
          : std::make_unique<absl::BitGen>(
                std::seed_seq{GetRandomGeneratorSeed()});
  return mt.get();
}
}  // namespace random

namespace util {

std::string StrError(int errnum) {
  constexpr int kStrErrorSize = 1024;
  char buffer[kStrErrorSize];
  char* str = nullptr;
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
  str = strerror_r(errnum, buffer, kStrErrorSize - 1);
#elif defined(_WIN32)
  strerror_s(buffer, kStrErrorSize - 1, errnum);
  str = buffer;
#else
  strerror_r(errnum, buffer, kStrErrorSize - 1);
  str = buffer;
#endif
  return absl::StrCat(str, " Error #", errnum);
}

std::vector<std::string> StrSplitAsCSV(absl::string_view text) {
  std::string buf = std::string(text);
  char* str = const_cast<char*>(buf.data());
  char* eos = str + text.size();
  char* start = nullptr;
  char* end = nullptr;

  std::vector<std::string> result;
  for (; str < eos; ++str) {
    if (*str == '"') {
      start = ++str;
      end = start;
      for (; str < eos; ++str) {
        if (*str == '"') {
          str++;
          if (*str != '"') break;
        }
        *end++ = *str;
      }
      str = std::find(str, eos, ',');
    } else {
      start = str;
      str = std::find(str, eos, ',');
      end = str;
    }
    *end = '\0';
    result.push_back(start);
  }

  return result;
}

#ifdef OS_WIN
std::wstring Utf8ToWide(absl::string_view input) {
  const int output_length = ::MultiByteToWideChar(
      CP_UTF8, 0, input.data(), static_cast<int>(input.size()), nullptr, 0);
  if (output_length == 0) {
    return L"";
  }
  std::wstring output(output_length, 0);
  const int result = ::MultiByteToWideChar(CP_UTF8, 0, input.data(),
                                           static_cast<int>(input.size()),
                                           output.data(), output.size());
  return result == output_length ? output : L"";
}
#endif
}  // namespace util

class ThreadPool::Impl {
 public:
  explicit Impl(int num_threads) {
    num_threads = std::min<int>(std::max<int>(1, num_threads), 65536);
    threads_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      threads_.push_back(std::thread(&Impl::WorkLoop, this));
    }
  }

  ~Impl() {
    {
      absl::MutexLock l(mu_);
      for (size_t i = 0; i < threads_.size(); i++) {
        queue_.push(nullptr);  // Shutdown signal.
      }
    }
    for (auto& thread : threads_) thread.join();
  }

  void Schedule(absl::AnyInvocable<void()> func) {
    absl::MutexLock l(mu_);
    queue_.push(std::move(func));
  }

  size_t num_threads() const { return threads_.size(); }

 private:
  bool WorkAvailable() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return !queue_.empty();
  }

  void WorkLoop() {
    while (true) {
      absl::AnyInvocable<void()> func;
      {
        absl::MutexLock l(mu_);
        mu_.Await(absl::Condition(this, &Impl::WorkAvailable));
        func = std::move(queue_.front());
        queue_.pop();
      }
      if (func == nullptr) {  // Shutdown signal.
        break;
      }
      func();
    }
  }

  absl::Mutex mu_;
  std::queue<absl::AnyInvocable<void()>> queue_ ABSL_GUARDED_BY(mu_);
  std::vector<std::thread> threads_;
};

ThreadPool::ThreadPool(size_t num_threads)
    : impl_(std::make_unique<Impl>(num_threads)) {}

ThreadPool::~ThreadPool() {}

void ThreadPool::Schedule(std::function<void()> func) {
  impl_->Schedule(std::move(func));
}

size_t ThreadPool::num_threads() const { return impl_->num_threads(); }

absl::Status RunBatch(size_t total_tasks,
                      std::function<absl::Status(size_t)> task_func,
                      ThreadPool& pool) {
  if (total_tasks == 0) return absl::OkStatus();

  // Cap workers to thread pool capacity.
  const size_t num_workers = std::min<size_t>(pool.num_threads(), total_tasks);

  std::atomic<size_t> index{0};      // For dynamic load-balancing
  std::atomic<bool> aborted{false};  // For early-abort

  absl::BlockingCounter barrier(num_workers);
  absl::Mutex status_mutex;
  absl::Status batch_status = absl::OkStatus();

  for (size_t n = 0; n < num_workers; ++n) {
    pool.Schedule([&]() {
      size_t i = 0;

      // Fetch next task index dynamically. Relaxed ordering is sufficient.
      while (!aborted.load(std::memory_order_relaxed) &&
             (i = index.fetch_add(1, std::memory_order_relaxed)) <
                 total_tasks) {
        absl::Status status = task_func(i);

        if (!status.ok()) {
          // Signal other workers to stop.
          aborted.store(true, std::memory_order_relaxed);

          // Keep the first error encountered.
          absl::MutexLock lock(&status_mutex);
          batch_status = std::move(status);
        }
      }

      barrier.DecrementCount();
    });
  }

  // Wait for all workers to finish.
  barrier.Wait();

  return batch_status;
}

namespace log_domain {
double LogSum(const std::vector<double>& xs) {
  if (xs.empty()) {
    return -1.0 * std::numeric_limits<double>::max();
  }
  double sum = xs.front();

  auto log_add = [](double xa, double xb) {
    if (xa > xb) {
      std::swap(xa, xb);
    }
    return xb + std::log1p(std::exp(xa - xb));
  };
  for (int i = 1; i < xs.size(); ++i) {
    sum = log_add(sum, xs[i]);
  }
  return sum;
}
}  // namespace log_domain
}  // namespace sentencepiece
