// Copyright 2018 Google Inc.
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

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "sentencepiece_processor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/hash/hash.h"

typedef int int32;
typedef long long int int64;
typedef unsigned long long int uint64;

namespace sentencepiece {
using ::tensorflow::DEVICE_CPU;
using ::tensorflow::Hash64;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

namespace {

// A utility function to convert sentencepiece::util::Status to
// ::tensorflow::Status
::tensorflow::Status ToTFStatus(const sentencepiece::util::Status& s) {
  if (s.ok()) return ::tensorflow::Status();
  return ::tensorflow::Status(static_cast<::tensorflow::error::Code>(s.code()),
                              ::tensorflow::string(s.error_message()));
}

// Global cache to reuse SentencePieceProcessor with the same
// model file or model proto.  The instance is managed with shared_ptr so
// the instance is deleted when no client is using it (refcount is zero).
class SentencePieceProcessorCache {
 public:
  std::shared_ptr<SentencePieceProcessor> get(
      const std::string key, bool is_proto,
      sentencepiece::util::Status* status) {
    std::lock_guard<std::mutex> l(mutex_);

    const uint64 fp = Hash64(key.data(), key.size());
    auto sp = data_[fp].lock();

    if (sp) {
      *status = sp->status();
      return sp;
    }

    sp = std::make_shared<SentencePieceProcessor>();
    *status = is_proto ? sp->LoadFromSerializedProto(key) : sp->Load(key);
    if (!status->ok()) return nullptr;

    data_[fp] = sp;
    return sp;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<uint64, std::weak_ptr<SentencePieceProcessor>> data_;
};

class SentencePieceBaseOp : public OpKernel {
 public:
  explicit SentencePieceBaseOp(OpKernelConstruction* context)
      : OpKernel(context) {
    std::string model_file_attr, model_proto_attr;
    OP_REQUIRES_OK(context, context->GetAttr("model_file", &model_file_attr));
    OP_REQUIRES_OK(context, context->GetAttr("model_proto", &model_proto_attr));

    // Initializes global cache.
    static SentencePieceProcessorCache* cache = new SentencePieceProcessorCache;
    sentencepiece::util::Status status;

    OP_REQUIRES(context,
                ((model_proto_attr.empty() && !model_file_attr.empty()) ||
                 (!model_proto_attr.empty() && model_file_attr.empty())),
                ::tensorflow::errors::InvalidArgument(
                    "Either `model_proto` or `model_file` must be set."));

    if (!model_file_attr.empty()) {
      sentencepiece_processor_ = cache->get(model_file_attr, false, &status);
    } else {
      // Loads serialized sentencepiece model proto to enable embedding the
      // relatively small sentencepiece model proto into the tensorflow graph
      // such that the tensorflow graph is self-contained.
      sentencepiece_processor_ = cache->get(model_proto_attr, true, &status);
    }

    OP_REQUIRES_OK(context, ToTFStatus(status));
    OP_REQUIRES(context, sentencepiece_processor_,
                ::tensorflow::errors::InvalidArgument(
                    "Failed to initialize SentencePieceProcessor"));

    // Sets extra options to add <s>, </s>.
    auto has_attribute = [&context](const std::string& name) {
      bool flag = false;
      context->GetAttr(name, &flag);
      return flag;
    };

    if (has_attribute("add_bos")) {
      bos_id_ = sentencepiece_processor_->bos_id();
      OP_REQUIRES(context, bos_id_ >= 0,
                  ::tensorflow::errors::InvalidArgument(
                      "`bos_id` is not defined in model"));
    }

    if (has_attribute("add_eos")) {
      eos_id_ = sentencepiece_processor_->eos_id();
      OP_REQUIRES(context, eos_id_ >= 0,
                  ::tensorflow::errors::InvalidArgument(
                      "`eos_id` is not defined in model"));
    }

    reverse_ = has_attribute("reverse");

    pad_id_ = sentencepiece_processor_->pad_id();
    if (pad_id_ == -1) pad_id_ = sentencepiece_processor_->unk_id();
  }

 protected:
  void GetPad(int32* pad) const { *pad = pad_id_; }

  void GetPad(std::string* pad) const {
    pad->clear();
    if (sentencepiece_processor_ && pad_id_ >= 0 &&
        pad_id_ != sentencepiece_processor_->unk_id())
      *pad = sentencepiece_processor_->IdToPiece(pad_id_);
  }

  std::shared_ptr<SentencePieceProcessor> sentencepiece_processor_;
  int bos_id_ = -1;
  int eos_id_ = -1;
  int pad_id_ = -1;
  bool reverse_ = false;
};
}  // namespace

class SentencePieceGetPieceSizeOp : public SentencePieceBaseOp {
 public:
  explicit SentencePieceGetPieceSizeOp(OpKernelConstruction* context)
      : SentencePieceBaseOp(context) {}

  void Compute(OpKernelContext* context) override {
    Tensor* vocab_size_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {}, &vocab_size_tensor));
    vocab_size_tensor->scalar<int32>()() =
        sentencepiece_processor_->GetPieceSize();
  }
};

template <typename S, typename T>
class SentencePieceConvertPieceOp : public SentencePieceBaseOp {
 public:
  explicit SentencePieceConvertPieceOp(OpKernelConstruction* context)
      : SentencePieceBaseOp(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor = nullptr;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor->shape(),
                                                     &output_tensor));
    for (int i = 0; i < input_tensor->NumElements(); ++i)
      output_tensor->flat<T>()(i) = Convert(input_tensor->flat<S>()(i));
  }

  int32 Convert(const std::string& piece) const {
    return sentencepiece_processor_->PieceToId(piece);
  }

  std::string Convert(int32 id) const {
    if (id >= 0 && id < sentencepiece_processor_->GetPieceSize()) {
      return sentencepiece_processor_->IdToPiece(id);
    }
    return "";
  }
};

class SentencePieceGetPieceTypeOp : public SentencePieceBaseOp {
 public:
  explicit SentencePieceGetPieceTypeOp(OpKernelConstruction* context)
      : SentencePieceBaseOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("piece_type", &piece_type_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor = nullptr;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor->shape(),
                                                     &output_tensor));

    for (int i = 0; i < input_tensor->NumElements(); ++i) {
      const int id = input_tensor->flat<int32>()(i);
      switch (piece_type_) {
        case 0:
          output_tensor->flat<bool>()(i) =
              sentencepiece_processor_->IsUnknown(id);
          break;
        case 1:
          output_tensor->flat<bool>()(i) =
              sentencepiece_processor_->IsControl(id);
          break;
        case 2:
          output_tensor->flat<bool>()(i) =
              sentencepiece_processor_->IsUnused(id);
          break;
        default:
          break;
      }
    }
  }

 private:
  int piece_type_;
};

template <typename T>
class SentencePieceEncodeOpBase : public SentencePieceBaseOp {
 public:
  explicit SentencePieceEncodeOpBase(OpKernelConstruction* context)
      : SentencePieceBaseOp(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor = nullptr;

    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor->shape()),
                ::tensorflow::errors::InvalidArgument(
                    "`input` must be a vector, got shape: ",
                    input_tensor->shape().DebugString()));
    const auto& input_sentences = input_tensor->vec<std::string>();
    const int64 batch_size = input_sentences.size();

    const Tensor* nbest_size_tensor = nullptr;
    OP_REQUIRES_OK(context, context->input("nbest_size", &nbest_size_tensor));
    OP_REQUIRES(context, nbest_size_tensor->dims() <= 1,
                ::tensorflow::errors::InvalidArgument(
                    "`nbest_size` must be a scalar or vector. got shape: ",
                    nbest_size_tensor->shape().DebugString()));
    if (nbest_size_tensor->dims() == 1) {
      OP_REQUIRES(
          context, batch_size == nbest_size_tensor->dim_size(0),
          ::tensorflow::errors::InvalidArgument(
              "`nbest_size` must have the same batch size as `input`."));
    }

    const Tensor* alpha_tensor = nullptr;
    OP_REQUIRES_OK(context, context->input("alpha", &alpha_tensor));
    OP_REQUIRES(context, alpha_tensor->dims() <= 1,
                ::tensorflow::errors::InvalidArgument(
                    "`alpha` must be a scalar or vector, got shape: ",
                    alpha_tensor->shape().DebugString()));
    if (alpha_tensor->dims() == 1) {
      OP_REQUIRES(context, batch_size == alpha_tensor->dim_size(0),
                  ::tensorflow::errors::InvalidArgument(
                      "`alpha` must have the same batch size as `input`."));
    }

    std::vector<std::vector<T>> pieces(batch_size);

    for (int64 i = 0; i < batch_size; ++i) {
      const int32 nbest_size = nbest_size_tensor->dims() == 1
                                   ? nbest_size_tensor->vec<int32>()(i)
                                   : nbest_size_tensor->scalar<int32>()();
      if (nbest_size == 0 || nbest_size == 1) {
        OP_REQUIRES_OK(context, ToTFStatus(sentencepiece_processor_->Encode(
                                    input_sentences(i), &pieces[i])));
      } else {
        const float alpha = alpha_tensor->dims() == 1
                                ? alpha_tensor->vec<float>()(i)
                                : alpha_tensor->scalar<float>()();
        OP_REQUIRES_OK(context,
                       ToTFStatus(sentencepiece_processor_->SampleEncode(
                           input_sentences(i), nbest_size, alpha, &pieces[i])));
      }
      RewritePieces(&pieces[i]);
    }

    MakeOutputTensor(context, pieces);
  }

 protected:
  void RewritePieces(std::vector<std::string>* pieces) const {
    if (reverse_) std::reverse(pieces->begin(), pieces->end());
    if (bos_id_ > 0)
      pieces->insert(pieces->begin(),
                     sentencepiece_processor_->IdToPiece(bos_id_));
    if (eos_id_ > 0)
      pieces->push_back(sentencepiece_processor_->IdToPiece(eos_id_));
  }

  void RewritePieces(std::vector<int32>* pieces) const {
    if (reverse_) std::reverse(pieces->begin(), pieces->end());
    if (bos_id_ > 0) pieces->insert(pieces->begin(), bos_id_);
    if (eos_id_ > 0) pieces->push_back(eos_id_);
  }

  virtual void MakeOutputTensor(OpKernelContext* context,
                                const std::vector<std::vector<T>>& pieces) = 0;
};

template <typename T>
class SentencePieceEncodeSparseOp : public SentencePieceEncodeOpBase<T> {
 public:
  explicit SentencePieceEncodeSparseOp(OpKernelConstruction* context)
      : SentencePieceEncodeOpBase<T>(context) {}

 protected:
  void MakeOutputTensor(OpKernelContext* context,
                        const std::vector<std::vector<T>>& pieces) override {
    const int64 batch_size = pieces.size();

    int64 max_sequence_length = 0;
    int64 indices_size = 0;
    for (int row = 0; row < batch_size; ++row) {
      const int col_size = pieces[row].size();
      max_sequence_length = std::max<int64>(col_size, max_sequence_length);
      indices_size += col_size;
    }

    // Creates the indices output tensor.
    Tensor* indices_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {indices_size, 2},
                                                     &indices_tensor));

    auto indices_tensor_output = indices_tensor->matrix<int64>();
    int item_idx = 0;
    for (int row = 0; row < batch_size; ++row) {
      for (int col = 0; col < pieces[row].size(); ++col) {
        indices_tensor_output(item_idx, 0) = row;
        indices_tensor_output(item_idx, 1) = col;
        ++item_idx;
      }
    }

    // Creates the values output tensor.
    Tensor* values_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, {indices_size}, &values_tensor));

    auto values_tensor_output = values_tensor->flat<T>();
    item_idx = 0;
    for (int row = 0; row < batch_size; ++row) {
      std::copy(pieces[row].begin(), pieces[row].end(),
                &values_tensor_output(item_idx));
      item_idx += pieces[row].size();
    }

    // Creates the shape output tensor.
    Tensor* shape_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {2}, &shape_tensor));

    auto shape_tensor_output = shape_tensor->flat<int64>();
    shape_tensor_output(0) = batch_size;
    shape_tensor_output(1) = max_sequence_length;
  }
};

template <typename T>
class SentencePieceEncodeDenseOp : public SentencePieceEncodeOpBase<T> {
 public:
  explicit SentencePieceEncodeDenseOp(OpKernelConstruction* context)
      : SentencePieceEncodeOpBase<T>(context) {
    this->GetPad(&pad_);
  }

  // protected:
  void MakeOutputTensor(OpKernelContext* context,
                        const std::vector<std::vector<T>>& pieces) override {
    const int64 batch_size = pieces.size();

    int64 max_sequence_length = 0;
    for (int row = 0; row < batch_size; ++row) {
      max_sequence_length =
          std::max<int64>(pieces[row].size(), max_sequence_length);
    }

    Tensor* values_tensor = nullptr;
    Tensor* length_tensor = nullptr;

    OP_REQUIRES_OK(
        context, context->allocate_output(0, {batch_size, max_sequence_length},
                                          &values_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, {batch_size}, &length_tensor));

    auto values_tensor_output = values_tensor->matrix<T>();
    auto length_tensor_output = length_tensor->vec<int32>();

    for (int row = 0; row < batch_size; ++row) {
      for (int col = 0; col < max_sequence_length; ++col) {
        values_tensor_output(row, col) =
            col < pieces[row].size() ? pieces[row][col] : pad_;
      }
      length_tensor_output(row) = pieces[row].size();
    }
  }

 private:
  T pad_;
};

template <typename T>
class SentencePieceDecodeOp : public SentencePieceBaseOp {
 public:
  explicit SentencePieceDecodeOp(OpKernelConstruction* context)
      : SentencePieceBaseOp(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor = nullptr;
    const Tensor* length_tensor = nullptr;

    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_tensor->shape()),
                ::tensorflow::errors::InvalidArgument(
                    "`input` must be a 2-D matrix. got shape: ",
                    input_tensor->shape().DebugString()));
    OP_REQUIRES_OK(context, context->input("sequence_length", &length_tensor));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(length_tensor->shape()),
                ::tensorflow::errors::InvalidArgument(
                    "`sequence_length` must be a vector. got shape: ",
                    length_tensor->shape().DebugString()));
    OP_REQUIRES(
        context, input_tensor->dim_size(0) == length_tensor->dim_size(0),
        ::tensorflow::errors::InvalidArgument(
            "`sequence_length` must have the same batch size as `input`."));

    const auto& input_sentences = input_tensor->matrix<T>();
    const auto& sequence_length = length_tensor->vec<int32>();
    const int64 batch_size = input_tensor->dim_size(0);
    const int max_sequence_length = input_tensor->dim_size(1);

    Tensor* values_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {batch_size}, &values_tensor));
    auto values_tensor_output = values_tensor->vec<std::string>();

    for (int64 i = 0; i < batch_size; ++i) {
      OP_REQUIRES(context,
                  (sequence_length(i) >= 0 &&
                   sequence_length(i) <= max_sequence_length),
                  ::tensorflow::errors::InvalidArgument(
                      "`sequence_length` is out-of-range."));
      std::vector<T> pieces(&input_sentences(i, 0),
                            &input_sentences(i, 0) + sequence_length(i));
      if (reverse_) std::reverse(pieces.begin(), pieces.end());
      OP_REQUIRES_OK(context, ToTFStatus(sentencepiece_processor_->Decode(
                                  pieces, &values_tensor_output(i))));
    }
  }
};

namespace {
// The snake case of this variables are used as the function names.
constexpr char kGetPieceSizeOpName[] = "SentencepieceGetPieceSize";
constexpr char kPieceToIdOpName[] = "SentencepiecePieceToId";
constexpr char kIdToPieceOpName[] = "SentencepieceIdToPiece";
constexpr char kGetPieceTypeOpName[] = "SentencepieceGetPieceType";
constexpr char kEncodeDenseOpName[] = "SentencepieceEncodeDense";
constexpr char kEncodeSparseOpName[] = "SentencepieceEncodeSparse";
constexpr char kDecodeOpName[] = "SentencepieceDecode";
}  // namespace

REGISTER_OP(kGetPieceSizeOpName)
    .Output("vocab_size: int32")
    .Attr("model_file: string = ''")
    .Attr("model_proto: string = ''")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->MakeShape({}));
      return ::tensorflow::Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name(kGetPieceSizeOpName).Device(DEVICE_CPU),
                        SentencePieceGetPieceSizeOp);

REGISTER_OP(kPieceToIdOpName)
    .Input("input: string")
    .Output("values: int32")
    .Attr("model_file: string = ''")
    .Attr("model_proto: string = ''")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name(kPieceToIdOpName).Device(DEVICE_CPU),
                        SentencePieceConvertPieceOp<std::string, int32>);

REGISTER_OP(kIdToPieceOpName)
    .Input("input: int32")
    .Output("values: string")
    .Attr("model_file: string = ''")
    .Attr("model_proto: string = ''")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name(kIdToPieceOpName).Device(DEVICE_CPU),
                        SentencePieceConvertPieceOp<int32, std::string>);

REGISTER_OP(kGetPieceTypeOpName)
    .Input("input: int32")
    .Output("values: bool")
    .Attr("model_file: string = ''")
    .Attr("model_proto: string = ''")
    .Attr("piece_type: int = 0")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name(kGetPieceTypeOpName).Device(DEVICE_CPU),
                        SentencePieceGetPieceTypeOp);

REGISTER_OP(kEncodeDenseOpName)
    .Attr("out_type: {int32, string} = DT_INT32")
    .Input("input: string")
    .Input("nbest_size: int32")
    .Input("alpha: float")
    .Output("values: out_type")
    .Output("sequence_length: int32")
    .Attr("model_file: string = ''")
    .Attr("model_proto: string = ''")
    .Attr("reverse: bool = false")
    .Attr("add_bos: bool = false")
    .Attr("add_eos: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input, nbest, alpha;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 1, &nbest));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &alpha));
      DimensionHandle batch_size = c->Dim(input, 0);
      if (c->Rank(nbest) == 1)
        TF_RETURN_IF_ERROR(c->Merge(batch_size, c->Dim(nbest, 0), &batch_size));
      if (c->Rank(alpha) == 1)
        TF_RETURN_IF_ERROR(c->Merge(batch_size, c->Dim(alpha, 0), &batch_size));
      c->set_output(0, c->MakeShape({batch_size, c->UnknownDim()}));
      c->set_output(1, c->MakeShape({batch_size}));
      return ::tensorflow::Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name(kEncodeDenseOpName)
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("out_type"),
                        SentencePieceEncodeDenseOp<int32>);

REGISTER_KERNEL_BUILDER(Name(kEncodeDenseOpName)
                            .Device(DEVICE_CPU)
                            .TypeConstraint<std::string>("out_type"),
                        SentencePieceEncodeDenseOp<std::string>);

REGISTER_OP(kEncodeSparseOpName)
    .Attr("out_type: {int32, string} = DT_INT32")
    .Input("input: string")
    .Input("nbest_size: int32")
    .Input("alpha: float")
    .Output("indices: int64")
    .Output("values: out_type")
    .Output("dense_shape: int64")
    .Attr("model_file: string = ''")
    .Attr("model_proto: string = ''")
    .Attr("reverse: bool = false")
    .Attr("add_bos: bool = false")
    .Attr("add_eos: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input, nbest, alpha;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 1, &nbest));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(2), 1, &alpha));
      DimensionHandle batch_size = c->Dim(input, 0);
      if (c->Rank(nbest) == 1)
        TF_RETURN_IF_ERROR(c->Merge(batch_size, c->Dim(nbest, 0), &batch_size));
      if (c->Rank(alpha) == 1)
        TF_RETURN_IF_ERROR(c->Merge(batch_size, c->Dim(alpha, 0), &batch_size));
      c->set_output(0, c->MakeShape({c->UnknownDim(), 2}));
      c->set_output(1, c->MakeShape({c->UnknownDim()}));
      c->set_output(2, c->MakeShape({2}));
      return ::tensorflow::Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name(kEncodeSparseOpName)
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("out_type"),
                        SentencePieceEncodeSparseOp<int32>);

REGISTER_KERNEL_BUILDER(Name(kEncodeSparseOpName)
                            .Device(DEVICE_CPU)
                            .TypeConstraint<std::string>("out_type"),
                        SentencePieceEncodeSparseOp<std::string>);

REGISTER_OP(kDecodeOpName)
    .Attr("T: {int32, string}")
    .Input("input: T")
    .Input("sequence_length: int32")
    .Output("values: string")
    .Attr("model_file: string = ''")
    .Attr("model_proto: string = ''")
    .Attr("reverse: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input, sequence_length;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sequence_length));
      DimensionHandle batch_size = c->Dim(input, 0);
      TF_RETURN_IF_ERROR(
          c->Merge(batch_size, c->Dim(sequence_length, 0), &batch_size));
      c->set_output(0, c->MakeShape({batch_size}));
      return ::tensorflow::Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name(kDecodeOpName).Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    SentencePieceDecodeOp<int32>);

REGISTER_KERNEL_BUILDER(
    Name(kDecodeOpName).Device(DEVICE_CPU).TypeConstraint<std::string>("T"),
    SentencePieceDecodeOp<std::string>);
}  // namespace sentencepiece
