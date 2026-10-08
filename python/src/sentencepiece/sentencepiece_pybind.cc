#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sentencepiece.pb.h>
#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "filesystem.h"
#include "sentencepiece_lite.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_model_converters.h"
#include "util.h"

namespace py = pybind11;

namespace {

// Helper to check if a python object is bytes or unicode, and get it as
// std::string_view without copying.
struct PyInputStringView {
  bool is_bytes = false;
  std::string_view value;

  explicit PyInputStringView(const py::object& obj) {
    if (py::isinstance<py::bytes>(obj)) {
      is_bytes = true;
      value = obj.cast<std::string_view>();
    } else if (py::isinstance<py::str>(obj)) {
      is_bytes = false;
      value = obj.cast<std::string_view>();
    } else {
      throw py::type_error("Input must be str or bytes");
    }
  }
};

// Helper to convert std::string to py::str or py::bytes based on flag.
py::object ToPyString(const std::string& str, bool is_bytes) {
  if (is_bytes) {
    return py::bytes(str);
  } else {
    return py::str(str);
  }
}

// Helper to convert std::vector<std::string> to py::list of py::str or
// py::bytes.
py::list ToPyStringList(const std::vector<std::string>& vec, bool is_bytes) {
  py::list l(vec.size());
  if (is_bytes) {
    for (size_t i = 0; i < vec.size(); ++i) {
      l[i] = py::bytes(vec[i]);
    }
  } else {
    for (size_t i = 0; i < vec.size(); ++i) {
      l[i] = py::str(vec[i]);
    }
  }
  return l;
}

// Exception translator
void RegisterExceptionTranslator() {
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const absl::Status& status) {
      std::string msg = status.ToString();
      switch (status.code()) {
        case absl::StatusCode::kNotFound:
          throw std::runtime_error(msg);  // Maps to RuntimeError
        case absl::StatusCode::kOutOfRange:
          throw std::out_of_range(msg);  // Maps to IndexError
        case absl::StatusCode::kInvalidArgument:
          throw std::invalid_argument(msg);  // Maps to ValueError
        default:
          throw std::runtime_error(msg);  // Maps to RuntimeError
      }
    }
  });
}

// PySentenceIterator wraps a Python iterator and implements
// sentencepiece::SentenceIterator.
class PySentenceIterator final : public sentencepiece::SentenceIterator {
 public:
  explicit PySentenceIterator(py::iterator it) : it_(std::move(it)) { Next(); }

  bool done() const override { return done_; }

  void Next() override {
    py::gil_scoped_acquire acquire;
    if (it_ == py::iterator::sentinel()) {
      done_ = true;
      return;
    }
    try {
      py::object item = py::reinterpret_borrow<py::object>(*it_);
      if (py::isinstance<py::str>(item) || py::isinstance<py::bytes>(item)) {
        value_ = item.cast<std::string>();
      } else {
        status_ = absl::Status(absl::StatusCode::kInvalidArgument,
                               "Iterator must return str or bytes.");
        done_ = true;
        return;
      }
      // Strip trailing \r or \n
      while (!value_.empty() &&
             (value_.back() == '\r' || value_.back() == '\n')) {
        value_.pop_back();
      }
      ++it_;
    } catch (const py::error_already_set& e) {
      status_ = absl::Status(absl::StatusCode::kInternal, e.what());
      done_ = true;
    }
  }

  const std::string& value() const override { return value_; }

  absl::Status status() const override { return status_; }

 private:
  py::iterator it_;
  std::string value_;
  absl::Status status_;
  bool done_ = false;
};

// Helper functions for RewriteIds (adapted from sentencepiece_swig.h)
absl::Status RewriteIds(const sentencepiece::SentencePieceProcessor& sp,
                        std::vector<int>* ids, bool add_bos, bool add_eos,
                        bool reverse) {
  if (add_bos && sp.bos_id() == -1) {
    return absl::InvalidArgumentError(
        "BOS token is not defined as a control symbol in this model.");
  }
  if (add_eos && sp.eos_id() == -1) {
    return absl::InvalidArgumentError(
        "EOS token is not defined as a control symbol in this model.");
  }
  if (!add_bos && !add_eos && !reverse) return absl::OkStatus();
  if (reverse) std::reverse(ids->begin(), ids->end());
  if (add_bos) {
    ids->insert(ids->begin(), sp.bos_id());
  }
  if (add_eos) {
    ids->push_back(sp.eos_id());
  }
  return absl::OkStatus();
}

absl::Status RewriteIds(const sentencepiece::SentencePieceProcessor& sp,
                        std::vector<std::string>* pieces, bool add_bos,
                        bool add_eos, bool reverse, bool emit_unk_piece) {
  if (add_bos && sp.bos_id() == -1) {
    return absl::InvalidArgumentError(
        "BOS token is not defined as a control symbol in this model.");
  }
  if (add_eos && sp.eos_id() == -1) {
    return absl::InvalidArgumentError(
        "EOS token is not defined as a control symbol in this model.");
  }
  if (!add_bos && !add_eos && !reverse && !emit_unk_piece)
    return absl::OkStatus();
  if (reverse) std::reverse(pieces->begin(), pieces->end());
  if (add_bos) {
    pieces->insert(pieces->begin(), sp.IdToPiece(sp.bos_id()));
  }
  if (add_eos) {
    pieces->push_back(sp.IdToPiece(sp.eos_id()));
  }
  if (emit_unk_piece) {
    if (sp.unk_id() == -1) {
      return absl::InvalidArgumentError(
          "UNK token is not defined in this model.");
    }
    const auto& unk = sp.IdToPiece(sp.unk_id());
    for (auto& piece : *pieces) {
      const int id = sp.PieceToId(piece);
      if (id == sp.unk_id()) {
        piece = unk;
      }
    }
  }
  return absl::OkStatus();
}

void RewriteIdsThrowException(const sentencepiece::SentencePieceProcessor& sp,
                              std::vector<int>* ids, bool add_bos, bool add_eos,
                              bool reverse) {
  auto status = RewriteIds(sp, ids, add_bos, add_eos, reverse);
  if (!status.ok()) throw status;
}

void RewriteIdsThrowException(const sentencepiece::SentencePieceProcessor& sp,
                              std::vector<std::string>* pieces, bool add_bos,
                              bool add_eos, bool reverse, bool emit_unk_piece) {
  auto status =
      RewriteIds(sp, pieces, add_bos, add_eos, reverse, emit_unk_piece);
  if (!status.ok()) throw status;
}

absl::Status CheckIds(absl::Span<const int> ids, int num_pieces) {
  for (int id : ids) {
    if (id < 0 || id >= num_pieces) {
      return absl::Status(absl::StatusCode::kOutOfRange,
                          "piece id is out of range.");
    }
  }
  return absl::OkStatus();
}

void CheckIdsThrowException(absl::Span<const int> ids, int num_pieces) {
  auto status = CheckIds(ids, num_pieces);
  if (!status.ok()) throw status;
}

int GetNumThreads(int num_threads) {
  if (num_threads < 0) {
    num_threads = std::thread::hardware_concurrency();
  }
  return std::max<int>(1, std::min<int>(num_threads, 1024));
}

class PyThreadPool {
 public:
  explicit PyThreadPool(int num_threads)
      : resolved_num_threads_(GetNumThreads(num_threads)),
        pool_(std::make_unique<sentencepiece::ThreadPool>(
            resolved_num_threads_)) {}
  sentencepiece::ThreadPool* get() { return pool_.get(); }
  int num_threads() const { return resolved_num_threads_; }

 private:
  int resolved_num_threads_;
  std::unique_ptr<sentencepiece::ThreadPool> pool_;
};

// Helper class to manage ThreadPool lifetime and acquisition in bindings
class WorkerPool {
 public:
  WorkerPool(int num_threads, py::object thread_pool, size_t max_tasks = 1024) {
    if (!thread_pool.is_none()) {
      pool_ = thread_pool.cast<PyThreadPool*>()->get();
    } else {
      const int max_workers =
          static_cast<int>(std::min<size_t>(max_tasks, 1024));
      const int n = std::max<int>(
          1, std::min<int>(GetNumThreads(num_threads), max_workers));
      pool_impl_ = std::make_unique<sentencepiece::ThreadPool>(n);
      pool_ = pool_impl_.get();
    }
  }

  sentencepiece::ThreadPool* get() { return pool_; }

 private:
  sentencepiece::ThreadPool* pool_ = nullptr;
  std::unique_ptr<sentencepiece::ThreadPool> pool_impl_;
};

// Wrapper to cast py::list or py::tuple to std::vector<std::string_view> and
// keep the underlying Python objects alive for the lifetime of this object.
class PyListStringViewVector {
 public:
  explicit PyListStringViewVector(const py::handle& ins_obj) {
    if (py::isinstance<py::list>(ins_obj)) {
      Init(py::reinterpret_borrow<py::list>(ins_obj));
    } else if (py::isinstance<py::tuple>(ins_obj)) {
      Init(py::reinterpret_borrow<py::tuple>(ins_obj));
    } else if (py::isinstance<py::sequence>(ins_obj) &&
               !py::isinstance<py::str>(ins_obj) &&
               !py::isinstance<py::bytes>(ins_obj)) {
      InitSequence(py::reinterpret_borrow<py::sequence>(ins_obj));
    } else {
      throw py::type_error("Input must be a list or tuple of str or bytes");
    }
  }

  absl::Span<const absl::string_view> views() const { return views_; }
  size_t size() const { return views_.size(); }
  bool empty() const { return views_.empty(); }
  bool is_bytes() const { return is_bytes_; }
  absl::string_view operator[](size_t i) const { return views_[i]; }

 private:
  template <typename FastSeq>
  void Init(const FastSeq& ins) {
    const size_t n = ins.size();
    keep_alive_.reserve(n);
    views_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      try {
        py::object obj = py::reinterpret_borrow<py::object>(ins[i]);
        if (i == 0) is_bytes_ = py::isinstance<py::bytes>(obj);
        views_.push_back(obj.cast<std::string_view>());
        keep_alive_.push_back(std::move(obj));
      } catch (const py::cast_error&) {
        throw py::type_error("List elements must be str or bytes");
      }
    }
  }

  void InitSequence(const py::sequence& ins) {
    const size_t n = ins.size();
    keep_alive_.reserve(n);
    views_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      try {
        py::object obj = ins[i];
        if (i == 0) is_bytes_ = py::isinstance<py::bytes>(obj);
        views_.push_back(obj.cast<std::string_view>());
        keep_alive_.push_back(std::move(obj));
      } catch (const py::cast_error&) {
        throw py::type_error("List elements must be str or bytes");
      }
    }
  }

  std::vector<py::object> keep_alive_;
  std::vector<absl::string_view> views_;
  bool is_bytes_ = false;
};

// Wrapper class to hold either a zero-copy Span or an owned vector of ints.
class IntSpanOrVector {
 public:
  IntSpanOrVector() : is_owned_(false) {}
  IntSpanOrVector(absl::Span<const int> span, py::buffer_info&& info)
      : info_(std::move(info)), span_(span), is_owned_(false) {}
  explicit IntSpanOrVector(std::vector<int>&& vec)
      : vec_(std::move(vec)), is_owned_(true) {}

  IntSpanOrVector(const IntSpanOrVector&) = delete;
  IntSpanOrVector& operator=(const IntSpanOrVector&) = delete;
  IntSpanOrVector(IntSpanOrVector&&) = default;
  IntSpanOrVector& operator=(IntSpanOrVector&&) = default;

  absl::Span<const int> span() const {
    return is_owned_ ? absl::Span<const int>(vec_) : span_;
  }

 private:
  py::buffer_info info_;
  std::vector<int> vec_;
  absl::Span<const int> span_;
  bool is_owned_;
};

// Wrapper for std::vector<int> to expose it as a Python buffer
class VectorBuffer {
 public:
  VectorBuffer() = default;
  explicit VectorBuffer(std::vector<int>&& vec) : vec_(std::move(vec)) {}

  const int* data() const { return vec_.data(); }
  size_t size() const { return vec_.size(); }

 private:
  std::vector<int> vec_;
};

bool IsSignedIntegerFormat(const std::string& format) {
  if (format.empty()) return false;
  char c = format.back();
  return (c == 'b' || c == 'h' || c == 'i' || c == 'l' || c == 'q' || c == 'n');
}

bool IsUnsignedIntegerFormat(const std::string& format) {
  if (format.empty()) return false;
  char c = format.back();
  return (c == 'B' || c == 'H' || c == 'I' || c == 'L' || c == 'Q' || c == 'N');
}

bool IsIntegerFormat(const std::string& format) {
  return IsSignedIntegerFormat(format) || IsUnsignedIntegerFormat(format);
}

int ClampSignedToInt(int64_t v) {
  if (v < std::numeric_limits<int>::min()) {
    return std::numeric_limits<int>::min();
  }
  if (v > std::numeric_limits<int>::max()) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(v);
}

int ClampUnsignedToInt(uint64_t v) {
  if (v > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(v);
}

// Helper to convert py::object (list, tuple, numpy array, etc) to
// std::vector<int>
IntSpanOrVector CastToIntSpanOrVector(const py::object& ids_obj) {
  if (py::isinstance<py::buffer>(ids_obj)) {
    try {
      py::buffer_info info = ids_obj.cast<py::buffer>().request();
      if (!IsIntegerFormat(info.format)) {
        throw py::type_error("Buffer must be of integer type");
      }
      if (info.ndim != 1) {
        throw py::type_error("Buffer must be 1-dimensional");
      }
      if (info.itemsize != 1 && info.itemsize != 2 && info.itemsize != 4 &&
          info.itemsize != 8) {
        throw py::type_error(
            "Unsupported buffer integer size (must be 8/16/32/64-bit)");
      }

      const py::ssize_t stride =
          info.strides.empty() ? info.itemsize : info.strides[0];
      const bool is_unsigned = IsUnsignedIntegerFormat(info.format);

      // Zero-copy path: read-only, contiguous, signed 32-bit int
      if (info.readonly && !is_unsigned && stride == info.itemsize &&
          info.itemsize == 4 &&
          reinterpret_cast<uintptr_t>(info.ptr) % alignof(int) == 0) {
        absl::Span<const int> span(static_cast<const int*>(info.ptr),
                                   info.shape[0]);
        return IntSpanOrVector(span, std::move(info));
      }

      std::vector<int> ids(info.shape[0]);
      const char* base = static_cast<const char*>(info.ptr);
      if (info.itemsize == 4 && !is_unsigned && stride == info.itemsize) {
        std::memcpy(ids.data(), info.ptr, info.shape[0] * 4);
      } else {
        for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
          const char* p = base + i * stride;
          if (is_unsigned) {
            uint64_t val = 0;
            if (info.itemsize == 1) {
              uint8_t v;
              std::memcpy(&v, p, sizeof(v));
              val = v;
            } else if (info.itemsize == 2) {
              uint16_t v;
              std::memcpy(&v, p, sizeof(v));
              val = v;
            } else if (info.itemsize == 4) {
              uint32_t v;
              std::memcpy(&v, p, sizeof(v));
              val = v;
            } else {
              std::memcpy(&val, p, sizeof(val));
            }
            ids[i] = ClampUnsignedToInt(val);
          } else {
            int64_t val = 0;
            if (info.itemsize == 1) {
              int8_t v;
              std::memcpy(&v, p, sizeof(v));
              val = v;
            } else if (info.itemsize == 2) {
              int16_t v;
              std::memcpy(&v, p, sizeof(v));
              val = v;
            } else if (info.itemsize == 4) {
              int32_t v;
              std::memcpy(&v, p, sizeof(v));
              val = v;
            } else {
              std::memcpy(&val, p, sizeof(val));
            }
            ids[i] = ClampSignedToInt(val);
          }
        }
      }
      return IntSpanOrVector(std::move(ids));
    } catch (const py::error_already_set&) {
      // Fallback if buffer request fails
    }
  }

  try {
    return IntSpanOrVector(ids_obj.cast<std::vector<int>>());
  } catch (const py::cast_error&) {
    throw py::type_error(
        "Input must be an integer buffer, list of integers, or a sequence of "
        "integers.");
  }
}

// Helper to convert py::object (sequence of sequences/buffers) to
// std::vector<std::vector<int>>
std::vector<IntSpanOrVector> CastToVectorIntSpanOrVector(
    const py::object& ins_obj) {
  py::sequence seq;
  try {
    seq = ins_obj.cast<py::sequence>();
  } catch (const py::cast_error&) {
    throw py::type_error(
        "Batch input must be a sequence of sequences or a 2D integer array.");
  }
  std::vector<IntSpanOrVector> outs;
  outs.reserve(seq.size());
  for (size_t i = 0; i < seq.size(); ++i) {
    outs.push_back(
        CastToIntSpanOrVector(py::reinterpret_borrow<py::object>(seq[i])));
  }
  return outs;
}

template <typename FuncType>
decltype(auto) SingleCall(const sentencepiece::SentencePieceProcessor& self,
                          int id, FuncType func) {
  CheckIdsThrowException(absl::Span<const int>(&id, 1), self.GetPieceSize());
  return (self.*func)(id);
}

template <typename FuncType>
py::list BatchCall(const sentencepiece::SentencePieceProcessor& self,
                   const py::object& ids_obj, FuncType func) {
  IntSpanOrVector ids = CastToIntSpanOrVector(ids_obj);
  CheckIdsThrowException(ids.span(), self.GetPieceSize());
  py::list outs(ids.span().size());
  for (size_t i = 0; i < ids.span().size(); ++i) {
    outs[i] = (self.*func)(ids.span()[i]);
  }
  return outs;
}

#define REGISTER_ID_METHOD(NAME)                                          \
  .def(#NAME,                                                             \
       [](const sentencepiece::SentencePieceProcessor& self, int id) {    \
         return SingleCall(self, id,                                      \
                           &sentencepiece::SentencePieceProcessor::NAME); \
       })                                                                 \
      .def(#NAME, [](const sentencepiece::SentencePieceProcessor& self,   \
                     const py::object& ids_obj) {                         \
        return BatchCall(self, ids_obj,                                   \
                         &sentencepiece::SentencePieceProcessor::NAME);   \
      })

void CheckProtoArgsThrowException(bool add_bos, bool add_eos, bool reverse,
                                  bool emit_unk_piece) {
  if (add_bos || add_eos || reverse || emit_unk_piece) {
    throw absl::Status(absl::StatusCode::kUnimplemented,
                       "add_bos, add_eos, reverse, and emit_unk_piece is not "
                       "supported in proto API");
  }
}

py::dict ExtractOffsetMapping(const sentencepiece::SentencePieceText& spt,
                              bool return_bytes, bool include_text = true) {
  std::vector<int> utf8_to_unicode;
  if (!return_bytes) {
    utf8_to_unicode =
        sentencepiece::string_util::UTF8ToUnicodeOffsets(spt.text());
  }

  const size_t num_pieces = spt.pieces_size();
  py::list ids_list(num_pieces);
  py::list pieces(num_pieces);
  py::list offsets(num_pieces);

  for (size_t i = 0; i < num_pieces; ++i) {
    const auto& piece = spt.pieces(i);
    ids_list[i] = piece.id();
    if (return_bytes) {
      pieces[i] = py::bytes(piece.piece());
      offsets[i] = py::make_tuple(piece.begin(), piece.end());
    } else {
      pieces[i] = py::str(piece.piece());
      if (piece.begin() >= utf8_to_unicode.size() ||
          piece.end() >= utf8_to_unicode.size()) {
        throw py::value_error("Invalid piece offsets in SentencePieceText");
      }
      offsets[i] = py::make_tuple(utf8_to_unicode[piece.begin()],
                                  utf8_to_unicode[piece.end()]);
    }
  }

  py::dict result;
  if (include_text) {
    result["text"] = ToPyString(spt.text(), return_bytes);
  }
  result["ids"] = ids_list;
  result["pieces"] = pieces;
  result["offsets"] = offsets;
  return result;
}

absl::Status ConvertModelProtoToFlatbuffer(
    const sentencepiece::ModelProto& model_proto,
    const std::string& output_file, bool skip_char_bigrams,
    bool treat_null_byte_as_unused, bool allow_unsupported_model_type,
    std::string* fb_out) {
  sentencepiece::lite::ConverterOptions options;
  options.skip_char_bigrams = skip_char_bigrams;
  options.treat_null_byte_as_unused = treat_null_byte_as_unused;
  options.allow_unsupported_model_type = allow_unsupported_model_type;
  auto fb_or = sentencepiece::lite::ToFlatbuffer(model_proto, options);
  if (!fb_or.ok()) return fb_or.status();
  sentencepiece::lite::SentencePieceLiteProcessor processor(*fb_or);
  if (processor.status() != sentencepiece::lite::StatusCode::kOk) {
    return absl::InvalidArgumentError(
        "Model verification failed: Converted FlatBuffers model is invalid.");
  }
  if (!output_file.empty()) {
    auto output = sentencepiece::filesystem::NewWritableFile(output_file, true);
    if (!output->status().ok()) return output->status();
    if (!output->Write(*fb_or)) {
      return absl::InternalError("Failed to write FlatBuffer model to " +
                                 output_file);
    }
  } else {
    *fb_out = std::move(*fb_or);
  }
  return absl::OkStatus();
}

}  // namespace

PYBIND11_MODULE(_sentencepiece, m, py::mod_gil_not_used()) {
  RegisterExceptionTranslator();

  py::class_<VectorBuffer>(m, "VectorBuffer", py::buffer_protocol())
      .def_buffer([](const VectorBuffer& m) -> py::buffer_info {
        return py::buffer_info(const_cast<int*>(m.data()), sizeof(int),
                               py::format_descriptor<int>::format(), 1,
                               {m.size()}, {sizeof(int)},
                               true  // readonly
        );
      })
      .def("data_address", [](const VectorBuffer& self) {
        return reinterpret_cast<uintptr_t>(self.data());
      });

  // Global functions
  m.def("SetRandomGeneratorSeed", &sentencepiece::SetRandomGeneratorSeed);
  m.def("SetMinLogLevel", &sentencepiece::SetMinLogLevel);
  m.def("SetNBestTimeout", &sentencepiece::SetNBestTimeout);
  m.def("SetDataDir", [](const std::string& data_dir) {
    sentencepiece::SetDataDir(data_dir);
  });
  m.def("_ConvertFileToFlatbuffer",
        [](const std::string& model_file, const std::string& output_file,
           bool skip_char_bigrams, bool treat_null_byte_as_unused,
           bool allow_unsupported_model_type) -> py::object {
          std::string fb_out;
          {
            py::gil_scoped_release release;
            sentencepiece::ModelProto model_proto;
            auto status =
                sentencepiece::io::LoadModelProto(model_file, &model_proto);
            if (!status.ok()) throw status;
            status = ConvertModelProtoToFlatbuffer(
                model_proto, output_file, skip_char_bigrams,
                treat_null_byte_as_unused, allow_unsupported_model_type,
                &fb_out);
            if (!status.ok()) throw status;
          }
          if (!output_file.empty()) return py::none();
          return py::bytes(fb_out);
        });
  m.def(
      "_ConvertSerializedProtoToFlatbuffer",
      [](const py::bytes& serialized, const std::string& output_file,
         bool skip_char_bigrams, bool treat_null_byte_as_unused,
         bool allow_unsupported_model_type) -> py::object {
        std::string_view serialized_view = serialized.cast<std::string_view>();
        std::string fb_out;
        {
          py::gil_scoped_release release;
          sentencepiece::ModelProto model_proto;
          if (!model_proto.ParseFromArray(serialized_view.data(),
                                          serialized_view.size())) {
            throw absl::InvalidArgumentError(
                "Failed to parse ModelProto from serialized bytes.");
          }
          auto status = ConvertModelProtoToFlatbuffer(
              model_proto, output_file, skip_char_bigrams,
              treat_null_byte_as_unused, allow_unsupported_model_type, &fb_out);
          if (!status.ok()) throw status;
        }
        if (!output_file.empty()) return py::none();
        return py::bytes(fb_out);
      });

  py::class_<PyThreadPool>(m, "ThreadPool")
      .def(py::init<int>())
      .def("num_threads", &PyThreadPool::num_threads);

  py::class_<sentencepiece::SentencePieceProcessor>(m, "SentencePieceProcessor")
      .def(py::init<>())
      .def("LoadFromFile",
           [](sentencepiece::SentencePieceProcessor& self,
              const std::string& filename) {
             py::gil_scoped_release release;
             auto status = self.Load(filename);
             if (!status.ok()) throw status;
             return true;
           })
      .def("LoadFromSerializedProto",
           [](sentencepiece::SentencePieceProcessor& self,
              const py::bytes& serialized) {
             std::string_view serialized_view =
                 serialized.cast<std::string_view>();
             py::gil_scoped_release release;
             auto status = self.LoadFromSerializedProto(serialized_view);
             if (!status.ok()) throw status;
             return true;
           })

      // Single Encode APIs
      .def("_EncodeAsIds",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input, bool enable_sampling, int nbest_size,
              float alpha, bool add_bos, bool add_eos, bool reverse) {
             PyInputStringView in(input);
             std::vector<int> ids;
             {
               py::gil_scoped_release release;
               absl::Status status;
               if (enable_sampling) {
                 status = self.SampleEncode(in.value, nbest_size, alpha, &ids);
               } else {
                 status = self.Encode(in.value, &ids);
               }
               if (!status.ok()) throw status;
               RewriteIdsThrowException(self, &ids, add_bos, add_eos, reverse);
             }
             return ids;
           })
      .def("_EncodeAsBuffer",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input, bool enable_sampling, int nbest_size,
              float alpha, bool add_bos, bool add_eos, bool reverse) {
             PyInputStringView in(input);
             std::vector<int> ids;
             {
               py::gil_scoped_release release;
               absl::Status status;
               if (enable_sampling) {
                 status = self.SampleEncode(in.value, nbest_size, alpha, &ids);
               } else {
                 status = self.Encode(in.value, &ids);
               }
               if (!status.ok()) throw status;
               RewriteIdsThrowException(self, &ids, add_bos, add_eos, reverse);
             }
             return VectorBuffer(std::move(ids));
           })
      .def("_EncodeAsPieces",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input, bool enable_sampling, int nbest_size,
              float alpha, bool add_bos, bool add_eos, bool reverse,
              bool emit_unk_piece, bool return_bytes) {
             PyInputStringView in(input);
             std::vector<std::string> pieces;
             {
               py::gil_scoped_release release;
               absl::Status status;
               if (enable_sampling) {
                 status =
                     self.SampleEncode(in.value, nbest_size, alpha, &pieces);
               } else {
                 status = self.Encode(in.value, &pieces);
               }
               if (!status.ok()) throw status;
               RewriteIdsThrowException(self, &pieces, add_bos, add_eos,
                                        reverse, emit_unk_piece);
             }
             return ToPyStringList(pieces, return_bytes);
           })
      .def("_EncodeAsSerializedProto",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input, bool enable_sampling, int nbest_size,
              float alpha, bool add_bos, bool add_eos, bool reverse,
              bool emit_unk_piece) {
             CheckProtoArgsThrowException(add_bos, add_eos, reverse,
                                          emit_unk_piece);
             PyInputStringView in(input);
             sentencepiece::SentencePieceText spt;
             {
               py::gil_scoped_release release;
               absl::Status status;
               if (enable_sampling) {
                 status = self.SampleEncode(in.value, nbest_size, alpha, &spt);
               } else {
                 status = self.Encode(in.value, &spt);
               }
               if (!status.ok()) throw status;
             }
             return py::bytes(spt.SerializeAsString());
           })
      .def("_EncodeAsOffsetMapping",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input, bool enable_sampling, int nbest_size,
              float alpha, bool add_bos, bool add_eos, bool reverse,
              bool emit_unk_piece, bool return_bytes) {
             CheckProtoArgsThrowException(add_bos, add_eos, reverse,
                                          emit_unk_piece);
             PyInputStringView in(input);
             sentencepiece::SentencePieceText spt;
             {
               py::gil_scoped_release release;
               absl::Status status;
               if (enable_sampling) {
                 status = self.SampleEncode(in.value, nbest_size, alpha, &spt);
               } else {
                 status = self.Encode(in.value, &spt);
               }
               if (!status.ok()) throw status;
             }
             return ExtractOffsetMapping(spt, return_bytes,
                                         /*include_text=*/false);
           })

      // Batch Encode APIs
      .def("_EncodeAsIdsBatch",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::list& ins, int num_threads, py::object thread_pool,
              bool enable_sampling, int nbest_size, float alpha, bool add_bos,
              bool add_eos, bool reverse) {
             PyListStringViewVector C_ins(ins);
             std::vector<std::vector<int>> outs(ins.size());
             WorkerPool pool(num_threads, thread_pool, ins.size());
             {
               py::gil_scoped_release release;
               auto status = sentencepiece::RunBatch(
                   ins.size(),
                   [&](size_t i) {
                     std::vector<int> out;
                     absl::Status s;
                     if (enable_sampling) {
                       s = self.SampleEncode(C_ins[i], nbest_size, alpha, &out);
                     } else {
                       s = self.Encode(C_ins[i], &out);
                     }
                     if (!s.ok()) return s;
                     s = RewriteIds(self, &out, add_bos, add_eos, reverse);
                     if (!s.ok()) return s;
                     outs[i] = std::move(out);
                     return absl::OkStatus();
                   },
                   *pool.get());
               if (!status.ok()) throw status;
             }
             return outs;
           })
      .def("_EncodeAsBufferBatch",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::list& ins, int num_threads, py::object thread_pool,
              bool enable_sampling, int nbest_size, float alpha, bool add_bos,
              bool add_eos, bool reverse) {
             PyListStringViewVector C_ins(ins);
             std::vector<std::vector<int>> temp_outs(ins.size());
             WorkerPool pool(num_threads, thread_pool, ins.size());
             {
               py::gil_scoped_release release;
               auto status = sentencepiece::RunBatch(
                   ins.size(),
                   [&](size_t i) {
                     std::vector<int> out;
                     absl::Status s;
                     if (enable_sampling) {
                       s = self.SampleEncode(C_ins[i], nbest_size, alpha, &out);
                     } else {
                       s = self.Encode(C_ins[i], &out);
                     }
                     if (!s.ok()) return s;
                     s = RewriteIds(self, &out, add_bos, add_eos, reverse);
                     if (!s.ok()) return s;
                     temp_outs[i] = std::move(out);
                     return absl::OkStatus();
                   },
                   *pool.get());
               if (!status.ok()) throw status;
             }
             std::vector<VectorBuffer> outs(ins.size());
             for (size_t i = 0; i < ins.size(); ++i) {
               outs[i] = VectorBuffer(std::move(temp_outs[i]));
             }
             return outs;
           })
      .def("_EncodeAsPiecesBatch",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::list& ins, int num_threads, py::object thread_pool,
              bool enable_sampling, int nbest_size, float alpha, bool add_bos,
              bool add_eos, bool reverse, bool emit_unk_piece,
              bool return_bytes) {
             if (ins.empty()) return py::list();
             PyListStringViewVector C_ins(ins);
             std::vector<std::vector<std::string>> outs(ins.size());
             WorkerPool pool(num_threads, thread_pool, ins.size());
             {
               py::gil_scoped_release release;
               auto status = sentencepiece::RunBatch(
                   ins.size(),
                   [&](size_t i) {
                     std::vector<std::string> out;
                     absl::Status s;
                     if (enable_sampling) {
                       s = self.SampleEncode(C_ins[i], nbest_size, alpha, &out);
                     } else {
                       s = self.Encode(C_ins[i], &out);
                     }
                     if (!s.ok()) return s;
                     s = RewriteIds(self, &out, add_bos, add_eos, reverse,
                                    emit_unk_piece);
                     if (!s.ok()) return s;
                     outs[i] = std::move(out);
                     return absl::OkStatus();
                   },
                   *pool.get());
               if (!status.ok()) throw status;
             }
             py::list py_outs(outs.size());
             for (size_t i = 0; i < outs.size(); ++i) {
               py_outs[i] = ToPyStringList(outs[i], return_bytes);
             }
             return py_outs;
           })
      .def("_EncodeAsSerializedProtoBatch",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::list& ins, int num_threads, py::object thread_pool,
              bool enable_sampling, int nbest_size, float alpha, bool add_bos,
              bool add_eos, bool reverse, bool emit_unk_piece) {
             CheckProtoArgsThrowException(add_bos, add_eos, reverse,
                                          emit_unk_piece);
             PyListStringViewVector C_ins(ins);
             std::vector<std::string> outs(ins.size());
             WorkerPool pool(num_threads, thread_pool, ins.size());
             {
               py::gil_scoped_release release;
               auto status = sentencepiece::RunBatch(
                   ins.size(),
                   [&](size_t i) {
                     sentencepiece::SentencePieceText spt;
                     absl::Status s;
                     if (enable_sampling) {
                       s = self.SampleEncode(C_ins[i], nbest_size, alpha, &spt);
                     } else {
                       s = self.Encode(C_ins[i], &spt);
                     }
                     if (!s.ok()) return s;
                     outs[i] = spt.SerializeAsString();
                     return absl::OkStatus();
                   },
                   *pool.get());
               if (!status.ok()) throw status;
             }
             py::list py_outs(outs.size());
             for (size_t i = 0; i < outs.size(); ++i) {
               py_outs[i] = py::bytes(outs[i]);
             }
             return py_outs;
           })
      .def("_EncodeAsOffsetMappingBatch",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::list& ins, int num_threads, py::object thread_pool,
              bool enable_sampling, int nbest_size, float alpha, bool add_bos,
              bool add_eos, bool reverse, bool emit_unk_piece,
              bool return_bytes) {
             CheckProtoArgsThrowException(add_bos, add_eos, reverse,
                                          emit_unk_piece);
             PyListStringViewVector C_ins(ins);
             std::vector<sentencepiece::SentencePieceText> spts(ins.size());
             WorkerPool pool(num_threads, thread_pool, ins.size());
             {
               py::gil_scoped_release release;
               auto status = sentencepiece::RunBatch(
                   ins.size(),
                   [&](size_t i) {
                     absl::Status s;
                     if (enable_sampling) {
                       s = self.SampleEncode(C_ins[i], nbest_size, alpha,
                                             &spts[i]);
                     } else {
                       s = self.Encode(C_ins[i], &spts[i]);
                     }
                     if (!s.ok()) return s;
                     return absl::OkStatus();
                   },
                   *pool.get());
               if (!status.ok()) throw status;
             }

             py::list py_results(ins.size());
             for (size_t batch_idx = 0; batch_idx < ins.size(); ++batch_idx) {
               py_results[batch_idx] = ExtractOffsetMapping(
                   spts[batch_idx], return_bytes, /*include_text=*/false);
             }
             return py_results;
           })

      // Single Decode APIs
      .def("_DecodeIds",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& ids_obj) {
             IntSpanOrVector ids = CastToIntSpanOrVector(ids_obj);
             std::string detok;
             {
               py::gil_scoped_release release;
               CheckIdsThrowException(ids.span(), self.GetPieceSize());
               auto status = self.Decode(ids.span(), &detok);
               if (!status.ok()) throw status;
             }
             return py::str(detok);
           })
      .def("_DecodeIdsAsBytes",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& ids_obj) {
             IntSpanOrVector ids = CastToIntSpanOrVector(ids_obj);
             std::string detok;
             {
               py::gil_scoped_release release;
               CheckIdsThrowException(ids.span(), self.GetPieceSize());
               auto status = self.Decode(ids.span(), &detok);
               if (!status.ok()) throw status;
             }
             return py::bytes(detok);
           })
      .def("_DecodePieces",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::sequence& pieces) {
             if (pieces.empty()) return py::object(py::str(""));
             PyListStringViewVector C_pieces(pieces);
             std::string detok;
             {
               py::gil_scoped_release release;
               auto status = self.Decode(C_pieces.views(), &detok);
               if (!status.ok()) throw status;
             }
             return ToPyString(detok, C_pieces.is_bytes());
           })
      .def("_DecodePiecesAsBytes",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::sequence& pieces) {
             if (pieces.empty()) return py::bytes("");
             PyListStringViewVector C_pieces(pieces);
             std::string detok;
             {
               py::gil_scoped_release release;
               auto status = self.Decode(C_pieces.views(), &detok);
               if (!status.ok()) throw status;
             }
             return py::bytes(detok);
           })
      .def("_DecodeIdsAsSerializedProto",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& ids_obj) {
             IntSpanOrVector ids = CastToIntSpanOrVector(ids_obj);
             sentencepiece::SentencePieceText spt;
             {
               py::gil_scoped_release release;
               CheckIdsThrowException(ids.span(), self.GetPieceSize());
               auto status = self.Decode(ids.span(), &spt);
               if (!status.ok()) throw status;
             }
             return py::bytes(spt.SerializeAsString());
           })
      .def("_DecodePiecesAsSerializedProto",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::sequence& pieces) {
             PyListStringViewVector C_pieces(pieces);
             sentencepiece::SentencePieceText spt;
             {
               py::gil_scoped_release release;
               auto status = self.Decode(C_pieces.views(), &spt);
               if (!status.ok()) throw status;
             }
             return py::bytes(spt.SerializeAsString());
           })

      .def("_DecodeAsOffsetMapping",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input, bool return_bytes) {
             bool input_is_pieces = false;
             bool detected_bytes = false;

             if (py::isinstance<py::list>(input) ||
                 py::isinstance<py::tuple>(input)) {
               py::sequence seq = py::reinterpret_borrow<py::sequence>(input);
               if (!seq.empty()) {
                 py::object first = seq[0];
                 if (py::isinstance<py::str>(first)) {
                   input_is_pieces = true;
                 } else if (py::isinstance<py::bytes>(first)) {
                   input_is_pieces = true;
                   detected_bytes = true;
                 }
               }
             }

             if (input_is_pieces) {
               return_bytes = detected_bytes;
             }

             sentencepiece::SentencePieceText spt;
             absl::Status status;
             if (input_is_pieces) {
               PyListStringViewVector pieces(input);
               {
                 py::gil_scoped_release release;
                 status = self.Decode(pieces.views(), &spt);
               }
             } else {
               IntSpanOrVector ids = CastToIntSpanOrVector(input);
               {
                 py::gil_scoped_release release;
                 CheckIdsThrowException(ids.span(), self.GetPieceSize());
                 status = self.Decode(ids.span(), &spt);
               }
             }
             if (!status.ok()) throw status;

             return ExtractOffsetMapping(spt, return_bytes);
           })

      // Batch Decode APIs
      .def("_DecodeIdsBatch",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& ins_obj, int num_threads,
              py::object thread_pool) {
             std::vector<IntSpanOrVector> ins =
                 CastToVectorIntSpanOrVector(ins_obj);
             std::vector<std::string> outs(ins.size());
             WorkerPool pool(num_threads, thread_pool, ins.size());
             {
               py::gil_scoped_release release;
               auto status = sentencepiece::RunBatch(
                   ins.size(),
                   [&](size_t i) {
                     auto s = CheckIds(ins[i].span(), self.GetPieceSize());
                     if (!s.ok()) return s;
                     return self.Decode(ins[i].span(), &outs[i]);
                   },
                   *pool.get());
               if (!status.ok()) throw status;
             }
             py::list py_outs(outs.size());
             for (size_t i = 0; i < outs.size(); ++i) {
               py_outs[i] = py::str(outs[i]);
             }
             return py_outs;
           })
      .def("_DecodeIdsAsBytesBatch",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& ins_obj, int num_threads,
              py::object thread_pool) {
             std::vector<IntSpanOrVector> ins =
                 CastToVectorIntSpanOrVector(ins_obj);
             std::vector<std::string> outs(ins.size());
             WorkerPool pool(num_threads, thread_pool, ins.size());
             {
               py::gil_scoped_release release;
               auto status = sentencepiece::RunBatch(
                   ins.size(),
                   [&](size_t i) {
                     auto s = CheckIds(ins[i].span(), self.GetPieceSize());
                     if (!s.ok()) return s;
                     return self.Decode(ins[i].span(), &outs[i]);
                   },
                   *pool.get());
               if (!status.ok()) throw status;
             }
             py::list py_outs(outs.size());
             for (size_t i = 0; i < outs.size(); ++i) {
               py_outs[i] = py::bytes(outs[i]);
             }
             return py_outs;
           })
      .def("_DecodeIdsAsSerializedProtoBatch",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& ins_obj, int num_threads,
              py::object thread_pool) {
             std::vector<IntSpanOrVector> ins =
                 CastToVectorIntSpanOrVector(ins_obj);
             std::vector<std::string> outs(ins.size());
             WorkerPool pool(num_threads, thread_pool, ins.size());
             {
               py::gil_scoped_release release;
               auto status = sentencepiece::RunBatch(
                   ins.size(),
                   [&](size_t i) {
                     auto s = CheckIds(ins[i].span(), self.GetPieceSize());
                     if (!s.ok()) return s;
                     sentencepiece::SentencePieceText spt;
                     s = self.Decode(ins[i].span(), &spt);
                     if (!s.ok()) return s;
                     outs[i] = spt.SerializeAsString();
                     return absl::OkStatus();
                   },
                   *pool.get());
               if (!status.ok()) throw status;
             }
             py::list py_outs(outs.size());
             for (size_t i = 0; i < outs.size(); ++i) {
               py_outs[i] = py::bytes(outs[i]);
             }
             return py_outs;
           })
      .def(
          "_DecodePiecesBatch",
          [](const sentencepiece::SentencePieceProcessor& self,
             const py::sequence& ins, int num_threads, py::object thread_pool) {
            if (ins.empty()) return py::list();

            // Empty sequences (e.g. a sentence that encoded to no pieces) are
            // valid and decode to an empty string, so where they appear in the
            // batch must not change the result. Determine the output element
            // type (str/bytes) from the first non-empty sequence before
            // releasing the GIL.
            bool is_bytes = false;
            bool found_non_empty = false;
            std::vector<PyListStringViewVector> C_ins_wrappers;
            C_ins_wrappers.reserve(ins.size());
            std::vector<absl::Span<const absl::string_view>> C_ins(ins.size());
            for (size_t i = 0; i < ins.size(); ++i) {
              C_ins_wrappers.emplace_back(ins[i]);
              if (!found_non_empty && !C_ins_wrappers.back().empty()) {
                is_bytes = C_ins_wrappers.back().is_bytes();
                found_non_empty = true;
              }
              C_ins[i] = C_ins_wrappers.back().views();
            }
            std::vector<std::string> outs(ins.size());
            WorkerPool pool(num_threads, thread_pool, ins.size());
            {
              py::gil_scoped_release release;
              auto status = sentencepiece::RunBatch(
                  ins.size(),
                  [&](size_t i) { return self.Decode(C_ins[i], &outs[i]); },
                  *pool.get());
              if (!status.ok()) throw status;
            }

            py::list py_outs(outs.size());
            for (size_t i = 0; i < outs.size(); ++i) {
              py_outs[i] = ToPyString(outs[i], is_bytes);
            }
            return py_outs;
          })
      .def(
          "_DecodePiecesAsBytesBatch",
          [](const sentencepiece::SentencePieceProcessor& self,
             const py::sequence& ins, int num_threads, py::object thread_pool) {
            if (ins.empty()) return py::list();
            std::vector<PyListStringViewVector> C_ins_wrappers;
            C_ins_wrappers.reserve(ins.size());
            std::vector<absl::Span<const absl::string_view>> C_ins(ins.size());
            for (size_t i = 0; i < ins.size(); ++i) {
              C_ins_wrappers.emplace_back(ins[i]);
              C_ins[i] = C_ins_wrappers.back().views();
            }
            std::vector<std::string> outs(ins.size());
            WorkerPool pool(num_threads, thread_pool, ins.size());
            {
              py::gil_scoped_release release;
              auto status = sentencepiece::RunBatch(
                  ins.size(),
                  [&](size_t i) { return self.Decode(C_ins[i], &outs[i]); },
                  *pool.get());
              if (!status.ok()) throw status;
            }
            py::list py_outs(outs.size());
            for (size_t i = 0; i < outs.size(); ++i) {
              py_outs[i] = py::bytes(outs[i]);
            }
            return py_outs;
          })
      .def(
          "_DecodePiecesAsSerializedProtoBatch",
          [](const sentencepiece::SentencePieceProcessor& self,
             const py::sequence& ins, int num_threads, py::object thread_pool) {
            if (ins.empty()) return py::list();
            std::vector<PyListStringViewVector> C_ins_wrappers;
            C_ins_wrappers.reserve(ins.size());
            std::vector<absl::Span<const absl::string_view>> C_ins(ins.size());
            for (size_t i = 0; i < ins.size(); ++i) {
              C_ins_wrappers.emplace_back(ins[i]);
              C_ins[i] = C_ins_wrappers.back().views();
            }
            std::vector<std::string> outs(ins.size());
            WorkerPool pool(num_threads, thread_pool, ins.size());
            {
              py::gil_scoped_release release;
              auto status = sentencepiece::RunBatch(
                  ins.size(),
                  [&](size_t i) {
                    sentencepiece::SentencePieceText spt;
                    auto s = self.Decode(C_ins[i], &spt);
                    if (!s.ok()) return s;
                    outs[i] = spt.SerializeAsString();
                    return absl::OkStatus();
                  },
                  *pool.get());
              if (!status.ok()) throw status;
            }
            py::list py_outs(outs.size());
            for (size_t i = 0; i < outs.size(); ++i) {
              py_outs[i] = py::bytes(outs[i]);
            }
            return py_outs;
          })

      .def("_DecodeAsOffsetMappingBatch",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& ins_obj, int num_threads,
              py::object thread_pool, bool return_bytes) {
             py::sequence seq_ins;
             try {
               seq_ins = ins_obj.cast<py::sequence>();
             } catch (const py::cast_error&) {
               throw py::type_error(
                   "Batch input must be a sequence of sequences or a 2D "
                   "integer array.");
             }
             if (seq_ins.empty()) return py::list();

             bool is_pieces_batch = false;
             bool detected_bytes = false;

             // A batch may start with empty sequences (e.g. a sentence that
             // encoded to no pieces / no ids). Empty sequences carry no type
             // information, so infer the element type from the first non-empty
             // inner sequence; otherwise a piece batch that starts with an
             // empty sequence would be decoded as an id batch.
             for (size_t i = 0; i < seq_ins.size(); ++i) {
               py::object seq = seq_ins[i];
               if (!py::isinstance<py::list>(seq) &&
                   !py::isinstance<py::tuple>(seq)) {
                 continue;
               }
               py::sequence inner_seq =
                   py::reinterpret_borrow<py::sequence>(seq);
               if (inner_seq.empty()) continue;
               py::object first_item = inner_seq[0];
               if (py::isinstance<py::str>(first_item)) {
                 is_pieces_batch = true;
               } else if (py::isinstance<py::bytes>(first_item)) {
                 is_pieces_batch = true;
                 detected_bytes = true;
               }
               break;
             }

             if (is_pieces_batch) {
               return_bytes = detected_bytes;
             }

             const size_t batch_size = seq_ins.size();
             std::vector<sentencepiece::SentencePieceText> spts(batch_size);
             WorkerPool pool(num_threads, thread_pool, batch_size);

             if (is_pieces_batch) {
               std::vector<PyListStringViewVector> C_ins_wrappers;
               C_ins_wrappers.reserve(batch_size);
               std::vector<absl::Span<const absl::string_view>> C_ins(
                   batch_size);
               for (size_t i = 0; i < batch_size; ++i) {
                 C_ins_wrappers.emplace_back(seq_ins[i]);
                 C_ins[i] = C_ins_wrappers.back().views();
               }
               {
                 py::gil_scoped_release release;
                 auto status = sentencepiece::RunBatch(
                     batch_size,
                     [&](size_t i) { return self.Decode(C_ins[i], &spts[i]); },
                     *pool.get());
                 if (!status.ok()) throw status;
               }
             } else {
               std::vector<IntSpanOrVector> C_ins =
                   CastToVectorIntSpanOrVector(ins_obj);

               {
                 py::gil_scoped_release release;
                 auto status = sentencepiece::RunBatch(
                     batch_size,
                     [&](size_t i) {
                       auto s = CheckIds(C_ins[i].span(), self.GetPieceSize());
                       if (!s.ok()) return s;
                       return self.Decode(C_ins[i].span(), &spts[i]);
                     },
                     *pool.get());
                 if (!status.ok()) throw status;
               }
             }

             py::list py_outs(spts.size());
             for (size_t i = 0; i < spts.size(); ++i) {
               py_outs[i] = ExtractOffsetMapping(spts[i], return_bytes);
             }
             return py_outs;
           })

      // NBest APIs
      .def("_NBestEncodeAsIds",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input, int nbest_size, bool add_bos,
              bool add_eos, bool reverse) {
             PyInputStringView in(input);
             std::vector<std::vector<int>> idss;
             {
               py::gil_scoped_release release;
               auto status = self.NBestEncode(in.value, nbest_size, &idss);
               if (!status.ok()) throw status;
               for (auto& ids : idss) {
                 RewriteIdsThrowException(self, &ids, add_bos, add_eos,
                                          reverse);
               }
             }
             return idss;
           })
      .def("_NBestEncodeAsBuffer",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input, int nbest_size, bool add_bos,
              bool add_eos, bool reverse) {
             PyInputStringView in(input);
             std::vector<std::vector<int>> idss;
             {
               py::gil_scoped_release release;
               auto status = self.NBestEncode(in.value, nbest_size, &idss);
               if (!status.ok()) throw status;
               for (auto& ids : idss) {
                 RewriteIdsThrowException(self, &ids, add_bos, add_eos,
                                          reverse);
               }
             }
             std::vector<VectorBuffer> outs(idss.size());
             for (size_t i = 0; i < idss.size(); ++i) {
               outs[i] = VectorBuffer(std::move(idss[i]));
             }
             return outs;
           })
      .def("_NBestEncodeAsPieces",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input, int nbest_size, bool add_bos,
              bool add_eos, bool reverse, bool emit_unk_piece,
              bool return_bytes) {
             PyInputStringView in(input);
             std::vector<std::vector<std::string>> piecess;
             {
               py::gil_scoped_release release;
               auto status = self.NBestEncode(in.value, nbest_size, &piecess);
               if (!status.ok()) throw status;
               for (auto& pieces : piecess) {
                 RewriteIdsThrowException(self, &pieces, add_bos, add_eos,
                                          reverse, emit_unk_piece);
               }
             }
             py::list py_outs(piecess.size());
             for (size_t i = 0; i < piecess.size(); ++i) {
               py_outs[i] = ToPyStringList(piecess[i], return_bytes);
             }
             return py_outs;
           })
      .def("_NBestEncodeAsSerializedProto",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input, int nbest_size, bool add_bos,
              bool add_eos, bool reverse, bool emit_unk_piece) {
             CheckProtoArgsThrowException(add_bos, add_eos, reverse,
                                          emit_unk_piece);
             PyInputStringView in(input);
             sentencepiece::NBestSentencePieceText nbest_spt;
             {
               py::gil_scoped_release release;
               auto status = self.NBestEncode(in.value, nbest_size, &nbest_spt);
               if (!status.ok()) throw status;
             }
             return py::bytes(nbest_spt.SerializeAsString());
           })

      // Parallel Encode APIs
      .def(
          "_ParallelEncodeAsIds",
          [](const sentencepiece::SentencePieceProcessor& self,
             const py::object& input, int chunk_len, int num_threads,
             py::object thread_pool, bool add_bos, bool add_eos, bool reverse) {
            PyInputStringView in(input);
            std::vector<int> ids;
            WorkerPool pool(num_threads, thread_pool);
            {
              py::gil_scoped_release release;
              auto status =
                  self.ParallelEncode(in.value, chunk_len, *pool.get(), &ids);
              if (!status.ok()) throw status;
              RewriteIdsThrowException(self, &ids, add_bos, add_eos, reverse);
            }
            return ids;
          })
      .def(
          "_ParallelEncodeAsBuffer",
          [](const sentencepiece::SentencePieceProcessor& self,
             const py::object& input, int chunk_len, int num_threads,
             py::object thread_pool, bool add_bos, bool add_eos, bool reverse) {
            PyInputStringView in(input);
            std::vector<int> ids;
            WorkerPool pool(num_threads, thread_pool);
            {
              py::gil_scoped_release release;
              auto status =
                  self.ParallelEncode(in.value, chunk_len, *pool.get(), &ids);
              if (!status.ok()) throw status;
              RewriteIdsThrowException(self, &ids, add_bos, add_eos, reverse);
            }
            return VectorBuffer(std::move(ids));
          })
      .def("_ParallelEncodeAsPieces",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input, int chunk_len, int num_threads,
              py::object thread_pool, bool add_bos, bool add_eos, bool reverse,
              bool emit_unk_piece, bool return_bytes) {
             PyInputStringView in(input);
             std::vector<std::string> pieces;
             WorkerPool pool(num_threads, thread_pool);
             {
               py::gil_scoped_release release;
               auto status = self.ParallelEncode(in.value, chunk_len,
                                                 *pool.get(), &pieces);
               if (!status.ok()) throw status;
               RewriteIdsThrowException(self, &pieces, add_bos, add_eos,
                                        reverse, emit_unk_piece);
             }
             return ToPyStringList(pieces, return_bytes);
           })
      .def("_ParallelEncodeAsSerializedProto",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input, int chunk_len, int num_threads,
              py::object thread_pool, bool add_bos, bool add_eos, bool reverse,
              bool emit_unk_piece) {
             CheckProtoArgsThrowException(add_bos, add_eos, reverse,
                                          emit_unk_piece);
             PyInputStringView in(input);
             sentencepiece::SentencePieceText spt;
             WorkerPool pool(num_threads, thread_pool);
             {
               py::gil_scoped_release release;
               auto status =
                   self.ParallelEncode(in.value, chunk_len, *pool.get(), &spt);
               if (!status.ok()) throw status;
             }
             return py::bytes(spt.SerializeAsString());
           })

      // Normalize APIs
      .def("_Normalize",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input) {
             PyInputStringView in(input);
             std::string norm;
             {
               py::gil_scoped_release release;
               norm = self.Normalize(in.value);
             }
             return ToPyString(norm, in.is_bytes);
           })
      .def("_NormalizeWithOffsets",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::object& input) {
             PyInputStringView in(input);
             std::string norm;
             std::vector<size_t> offsets;
             {
               py::gil_scoped_release release;
               auto status = self.Normalize(in.value, &norm, &offsets);
               if (!status.ok()) throw status;
             }
             if (!in.is_bytes) {
               sentencepiece::ConvertToUnicodeAlignment(in.value, norm,
                                                        &offsets);
             }
             return py::make_tuple(ToPyString(norm, in.is_bytes), offsets);
           })

      // Vocab management
      .def("GetPieceSize", &sentencepiece::SentencePieceProcessor::GetPieceSize)
      .def("PieceToId",
           [](const sentencepiece::SentencePieceProcessor& self,
              std::string_view piece) { return self.PieceToId(piece); })
      .def("PieceToId",
           [](const sentencepiece::SentencePieceProcessor& self,
              const py::sequence& pieces) {
             py::list ids(pieces.size());
             for (size_t i = 0; i < pieces.size(); ++i) {
               try {
                 ids[i] = self.PieceToId(pieces[i].cast<std::string_view>());
               } catch (const py::cast_error&) {
                 throw py::type_error("Sequence elements must be str or bytes");
               }
             }
             return ids;
           }) REGISTER_ID_METHOD(IdToPiece) REGISTER_ID_METHOD(GetScore)
          REGISTER_ID_METHOD(IsUnknown) REGISTER_ID_METHOD(IsControl)
              REGISTER_ID_METHOD(IsUnused) REGISTER_ID_METHOD(IsByte)
      .def("unk_id", &sentencepiece::SentencePieceProcessor::unk_id)
      .def("bos_id", &sentencepiece::SentencePieceProcessor::bos_id)
      .def("eos_id", &sentencepiece::SentencePieceProcessor::eos_id)
      .def("pad_id", &sentencepiece::SentencePieceProcessor::pad_id)

      // Serialization
      .def("serialized_model_proto",
           [](const sentencepiece::SentencePieceProcessor& self) {
             return py::bytes(self.serialized_model_proto());
           });

  // Bind Trainer. Use py::nodelete holder to avoid trying to delete it because
  // destructor is private.
  py::class_<
      sentencepiece::SentencePieceTrainer,
      std::unique_ptr<sentencepiece::SentencePieceTrainer, py::nodelete>>(
      m, "SentencePieceTrainer")
      .def_static("_TrainFromString",
                  [](const std::string& arg) {
                    py::gil_scoped_release release;
                    auto status = sentencepiece::SentencePieceTrainer::Train(
                        arg, sentencepiece::TrainerComponents{});
                    if (!status.ok()) throw status;
                    return true;
                  })
      .def_static(
          "_TrainFromMap",
          [](const std::unordered_map<std::string, std::string>& args,
             py::object sentence_iterator, py::object pretokenizer,
             bool allow_inconsistent_pretokenization,
             bool return_model_proto) -> py::object {
            sentencepiece::TrainerComponents components;
            components.allow_inconsistent_pretokenization =
                allow_inconsistent_pretokenization;
            std::unique_ptr<PySentenceIterator> py_iter;
            if (!sentence_iterator.is_none()) {
              py_iter = std::make_unique<PySentenceIterator>(
                  sentence_iterator.cast<py::iterator>());
              components.sentence_iterator = py_iter.get();
            }
            absl::Mutex pretokenizer_mu;
            std::string pretokenizer_error;
            std::atomic<bool> has_pretokenizer_error{false};
            if (!pretokenizer.is_none()) {
              components.pretokenizer =
                  [&pretokenizer, &pretokenizer_mu, &pretokenizer_error,
                   &has_pretokenizer_error](
                      absl::string_view text) -> std::vector<std::string> {
                if (has_pretokenizer_error.load(std::memory_order_relaxed)) {
                  return {" "};
                }
                py::gil_scoped_acquire acquire;
                try {
                  py::object py_text = py::str(text.data(), text.size());
                  py::object result = pretokenizer(py_text);
                  std::vector<std::string> chunks;
                  for (auto item : result.cast<py::sequence>()) {
                    chunks.push_back(item.cast<std::string>());
                  }
                  return chunks;
                } catch (const std::exception& e) {
                  absl::MutexLock lock(pretokenizer_mu);
                  if (pretokenizer_error.empty()) {
                    pretokenizer_error =
                        std::string("Pretokenizer failed: ") + e.what();
                    has_pretokenizer_error.store(true,
                                                 std::memory_order_relaxed);
                  }
                  return {" "};
                }
              };
            }

            std::string model_proto;
            absl::Status status;
            {
              py::gil_scoped_release release;
              status = sentencepiece::SentencePieceTrainer::Train(
                  args, components,
                  return_model_proto ? &model_proto : nullptr);
            }
            if (!pretokenizer_error.empty()) {
              throw std::runtime_error(pretokenizer_error);
            }
            if (!status.ok()) throw status;

            if (return_model_proto) {
              return py::bytes(model_proto);
            }
            return py::none();
          },
          py::arg("args"), py::arg("sentence_iterator") = py::none(),
          py::arg("pretokenizer") = nullptr,
          py::arg("allow_inconsistent_pretokenization") = false,
          py::arg("return_model_proto") = false);

  // Bind Normalizer
  py::class_<sentencepiece::SentencePieceNormalizer>(m,
                                                     "SentencePieceNormalizer")
      .def(py::init<>())
      .def("LoadFromFile",
           [](sentencepiece::SentencePieceNormalizer& self,
              const std::string& filename) {
             py::gil_scoped_release release;
             auto status = self.Load(filename);
             if (!status.ok()) throw status;
             return true;
           })
      .def("LoadFromSerializedProto",
           [](sentencepiece::SentencePieceNormalizer& self,
              const py::bytes& serialized) {
             std::string_view serialized_view =
                 serialized.cast<std::string_view>();
             py::gil_scoped_release release;
             auto status = self.LoadFromSerializedProto(serialized_view);
             if (!status.ok()) throw status;
             return true;
           })
      .def("LoadFromSerializedNormalizerSpec",
           [](sentencepiece::SentencePieceNormalizer& self,
              const py::bytes& serialized) {
             std::string_view serialized_view =
                 serialized.cast<std::string_view>();
             py::gil_scoped_release release;
             auto status =
                 self.LoadFromSerializedNormalizerSpec(serialized_view);
             if (!status.ok()) throw status;
             return true;
           })
      .def("LoadFromRuleTSV",
           [](sentencepiece::SentencePieceNormalizer& self,
              const std::string& filename) {
             py::gil_scoped_release release;
             auto status = self.LoadFromRuleTSV(filename);
             if (!status.ok()) throw status;
             return true;
           })
      .def("LoadFromRuleName",
           [](sentencepiece::SentencePieceNormalizer& self,
              const std::string& name) {
             py::gil_scoped_release release;
             auto status = self.LoadFromRuleName(name);
             if (!status.ok()) throw status;
             return true;
           })
      .def(
          "LoadFromMap",
          [](sentencepiece::SentencePieceNormalizer& self,
             const std::vector<std::pair<std::string, std::string>>& norm_map) {
            py::gil_scoped_release release;
            auto status = self.LoadFromMap(norm_map);
            if (!status.ok()) throw status;
            return true;
          })
      .def("Decompile",
           [](const sentencepiece::SentencePieceNormalizer& self) {
             std::vector<std::pair<std::string, std::string>> norm_map;
             {
               py::gil_scoped_release release;
               auto status = self.Decompile(&norm_map);
               if (!status.ok()) throw status;
             }
             return norm_map;
           })
      .def("_Normalize",
           [](const sentencepiece::SentencePieceNormalizer& self,
              const py::object& input) {
             PyInputStringView in(input);
             std::string norm;
             {
               py::gil_scoped_release release;
               auto status = self.Normalize(in.value, &norm);
               if (!status.ok()) throw status;
             }
             return ToPyString(norm, in.is_bytes);
           })
      .def("_NormalizeWithOffsets",
           [](const sentencepiece::SentencePieceNormalizer& self,
              const py::object& input) {
             PyInputStringView in(input);
             std::string norm;
             std::vector<size_t> offsets;
             {
               py::gil_scoped_release release;
               auto status = self.Normalize(in.value, &norm, &offsets);
               if (!status.ok()) throw status;
             }
             if (!in.is_bytes) {
               sentencepiece::ConvertToUnicodeAlignment(in.value, norm,
                                                        &offsets);
             }
             return py::make_tuple(ToPyString(norm, in.is_bytes), offsets);
           })
      .def("_SetProtoField",
           [](sentencepiece::SentencePieceNormalizer& self,
              const std::string& name, bool value) {
             auto status = sentencepiece::SentencePieceTrainer::SetProtoField(
                 name, value ? "1" : "0", self.mutable_normalizer_spec());
             if (!status.ok()) throw status;
           })
      .def("serialized_model_proto",
           [](const sentencepiece::SentencePieceNormalizer& self) {
             return py::bytes(self.serialized_model_proto());
           })
      .def("serialized_normalizer_spec",
           [](const sentencepiece::SentencePieceNormalizer& self) {
             return py::bytes(self.serialized_normalizer_spec());
           });
}
