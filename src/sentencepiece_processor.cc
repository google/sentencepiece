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

#include "sentencepiece_processor.h"
#include <random>
#include "common.h"
#include "model_factory.h"
#include "normalizer.h"
#include "sentencepiece.pb.h"
#include "unigram_model.h"
#include "util.h"

namespace sentencepiece {
namespace {

// Replaces white space with U+2581 (LOWER ONE EIGHT BLOCK).
const char kSpaceSymbol[] = "\xe2\x96\x81";

// Encodes <unk> into U+2047 (DOUBLE QUESTION MARK),
// since this character can be useful both for user and
// developer. We can easily figure out that <unk> is emitted.
const char kUnknownSymbol[] = " \xE2\x81\x87 ";
}  // namespace

SentencePieceProcessor::SentencePieceProcessor() {}
SentencePieceProcessor::~SentencePieceProcessor() {}

util::Status SentencePieceProcessor::Load(const std::string &filename) {
  std::ifstream ifs(filename.c_str(), std::ios::binary | std::ios::in);
  if (!ifs) {
    return util::NotFoundError(std::string("Cannot open ") + filename);
  }

  return Load(&ifs);
}

util::Status SentencePieceProcessor::Load(std::istream *is) {
  if (is == nullptr)
    return util::InternalError("input ifstream is null");

  model_proto_ = port::MakeUnique<ModelProto>();
  if (!model_proto_->ParseFromIstream(is)) {
    return util::InternalError("Model file is broken");
  }

  model_ = ModelFactory::Create(*model_proto_);
  normalizer_ =
      port::MakeUnique<normalizer::Normalizer>(model_proto_->normalizer_spec());

  return status();
}

util::Status SentencePieceProcessor::SetEncodeExtraOptions(
    const std::string &extra_options) {
  return ParseExtraOptions(extra_options, &encode_extra_options_);
}

util::Status SentencePieceProcessor::SetDecodeExtraOptions(
    const std::string &extra_options) {
  return ParseExtraOptions(extra_options, &decode_extra_options_);
}

util::Status SentencePieceProcessor::status() const {
  if (model_ == nullptr)
    return util::InternalError("Model is not initialized.");
  if (normalizer_ == nullptr)
    return util::InternalError("Normalizer is not initialized.");
  if (!model_->status().ok()) return model_->status();
  if (!normalizer_->status().ok()) return normalizer_->status();

  return util::OkStatus();
}

#define CHECK_OR_RETURN_STATUS_STL(container)                   \
  RETURN_IF_ERROR(status());                                    \
  if (container == nullptr)                                     \
    return util::InternalError("output container is null");     \
  container->clear();

#define CHECK_OR_RETURN_STATUS_PROTO(proto)             \
  RETURN_IF_ERROR(status());                            \
  if (proto == nullptr)                                 \
    return util::InternalError("output proto is null"); \
  proto->Clear();

//////////////////////////////////////////////////////////////
// Simple API.
util::Status SentencePieceProcessor::Encode(const std::string &input,
                                            std::vector<std::string> *pieces) const {
  CHECK_OR_RETURN_STATUS_STL(pieces);

  SentencePieceText spt;
  RETURN_IF_ERROR(Encode(input, &spt));
  for (const auto &sp : spt.pieces()) {
    pieces->emplace_back(sp.piece());
  }

  return util::OkStatus();
}

util::Status SentencePieceProcessor::Encode(const std::string &input,
                                            std::vector<int> *ids) const {
  CHECK_OR_RETURN_STATUS_STL(ids);

  SentencePieceText spt;
  RETURN_IF_ERROR(Encode(input, &spt));
  for (const auto &sp : spt.pieces()) {
    ids->emplace_back(sp.id());
  }

  return util::OkStatus();
}

util::Status SentencePieceProcessor::Decode(const std::vector<std::string> &pieces,
                                            std::string *detokenized) const {
  CHECK_OR_RETURN_STATUS_STL(detokenized);
  SentencePieceText spt;

  RETURN_IF_ERROR(Decode(pieces, &spt));
  *detokenized = std::move(spt.text());

  return util::OkStatus();
}

util::Status SentencePieceProcessor::Decode(const std::vector<int> &ids,
                                            std::string *detokenized) const {
  CHECK_OR_RETURN_STATUS_STL(detokenized);
  SentencePieceText spt;

  RETURN_IF_ERROR(Decode(ids, &spt));
  *detokenized = std::move(spt.text());

  return util::OkStatus();
}

util::Status SentencePieceProcessor::NBestEncode(
    const std::string &input, int nbest_size,
    std::vector<std::vector<std::string>> *pieces) const {
  CHECK_OR_RETURN_STATUS_STL(pieces);

  NBestSentencePieceText spt;
  RETURN_IF_ERROR(NBestEncode(input, nbest_size, &spt));
  for (const auto &nbest : spt.nbests()) {
    std::vector<std::string> result;
    for (const auto &sp : nbest.pieces()) {
      result.emplace_back(sp.piece());
    }
    pieces->emplace_back(result);
  }

  return util::OkStatus();
}

util::Status SentencePieceProcessor::NBestEncode(
    const std::string &input, int nbest_size,
    std::vector<std::vector<int>> *ids) const {
  CHECK_OR_RETURN_STATUS_STL(ids);

  NBestSentencePieceText spt;
  RETURN_IF_ERROR(NBestEncode(input, nbest_size, &spt));
  for (const auto &nbest : spt.nbests()) {
    std::vector<int> result;
    for (const auto &sp : nbest.pieces()) {
      result.emplace_back(sp.id());
    }
    ids->emplace_back(result);
  }

  return util::OkStatus();
}

util::Status SentencePieceProcessor::SampleEncode(
    const std::string &input, int nbest_size, float alpha,
    std::vector<std::string> *pieces) const {
  CHECK_OR_RETURN_STATUS_STL(pieces);

  SentencePieceText spt;
  RETURN_IF_ERROR(SampleEncode(input, nbest_size, alpha, &spt));
  for (const auto &sp : spt.pieces()) {
    pieces->emplace_back(sp.piece());
  }

  return util::OkStatus();
}

util::Status SentencePieceProcessor::SampleEncode(const std::string &input,
                                                  int nbest_size, float alpha,
                                                  std::vector<int> *ids) const {
  CHECK_OR_RETURN_STATUS_STL(ids);

  SentencePieceText spt;
  RETURN_IF_ERROR(SampleEncode(input, nbest_size, alpha, &spt));
  for (const auto &sp : spt.pieces()) {
    ids->emplace_back(sp.id());
  }

  return util::OkStatus();
}

util::Status SentencePieceProcessor::PopulateSentencePieceText(
    const std::string &input, const std::string &normalized,
    const std::vector<size_t> &norm_to_orig, const EncodeResult &result,
    SentencePieceText *spt) const {
  size_t consumed = 0;
  bool is_prev_unk = false;
  for (const auto &p : result) {
    const StringPiece w = p.first;  // piece
    const int id = p.second;        // id

    if (w.empty()) {
      return util::InternalError("Empty piece is not allowed.");
    }

    const bool is_unk = IsUnknown(id);

    if (IsControl(id)) {
      // Control symbol has no corresponding source surface, so begin == end.
      auto *sp = spt->add_pieces();
      sp->set_piece(w.to_string());
      sp->set_id(id);
      sp->set_begin(norm_to_orig[consumed]);
      sp->set_end(norm_to_orig[consumed]);
    } else {
      const size_t begin = consumed;
      const size_t end = consumed + w.size();
      if (begin >= norm_to_orig.size() || end >= norm_to_orig.size()) {
        return util::OutOfRangeError("consumed index is out-of-range.");
      }
      const size_t orig_begin = norm_to_orig[begin];
      const size_t orig_end = norm_to_orig[end];
      if (orig_begin > input.size() || orig_end > input.size() ||
          orig_begin > orig_end) {
        return util::OutOfRangeError("original index is out-of-range.");
      }
      const auto surface = input.substr(orig_begin, orig_end - orig_begin);
      // Merges continuous run of unknown pieces so that decoder
      // can copy or generate unknown tokens easily.
      // Note that merged tokens are still unknown,
      // since known pieces never consist of unknown characters.
      if (is_prev_unk && is_unk) {
        auto *sp = spt->mutable_pieces(spt->pieces_size() - 1);
        sp->set_piece(sp->piece() + w.to_string());
        sp->set_surface(sp->surface() + surface);
        sp->set_end(orig_end);
      } else {
        auto *sp = spt->add_pieces();
        sp->set_piece(w.to_string());
        sp->set_id(id);
        sp->set_surface(surface);
        sp->set_begin(orig_begin);
        sp->set_end(orig_end);
      }
      consumed += w.size();
    }
    is_prev_unk = is_unk;
  }

  if (consumed != normalized.size()) {
    return util::OutOfRangeError("all normalized characters are not consumed.");
  }

  RETURN_IF_ERROR(ApplyExtraOptions(encode_extra_options_, spt));

  spt->set_text(input);

  return util::OkStatus();
}

util::Status SentencePieceProcessor::Encode(const std::string &input,
                                            SentencePieceText *spt) const {
  CHECK_OR_RETURN_STATUS_PROTO(spt);

  std::string normalized;
  std::vector<size_t> norm_to_orig;
  RETURN_IF_ERROR(normalizer_->Normalize(input, &normalized, &norm_to_orig));

  const auto result = model_->Encode(normalized);
  RETURN_IF_ERROR(PopulateSentencePieceText(input, normalized, norm_to_orig, result, spt));

  return util::OkStatus();
}

util::Status SentencePieceProcessor::NBestEncode(
    const std::string &input, int nbest_size,
    NBestSentencePieceText *nbest_spt) const {
  CHECK_OR_RETURN_STATUS_PROTO(nbest_spt);

  std::string normalized;
  std::vector<size_t> norm_to_orig;
  RETURN_IF_ERROR(normalizer_->Normalize(input, &normalized, &norm_to_orig));

  const auto nbests =
      model_->NBestEncode(normalized, nbest_size);
  if (nbests.empty()) {
    return util::InternalError("NBestEncode returns empty result");
  }

  for (const auto &result : nbests) {
    auto *spt = nbest_spt->add_nbests();
    spt->set_score(result.second);
    RETURN_IF_ERROR(PopulateSentencePieceText(input, normalized, norm_to_orig, result.first,
                                              spt));
  }

  return util::OkStatus();
}

util::Status SentencePieceProcessor::SampleEncode(const std::string &input,
                                                  int nbest_size, float alpha,
                                                  SentencePieceText *spt) const {
  CHECK_OR_RETURN_STATUS_PROTO(spt);

  if (nbest_size > 512 || nbest_size == 0) {
    return util::OutOfRangeError(
        "nbest_size must be 0 < nbest_size <= 512 or nbest_size < 0.");
  }

  std::string normalized;
  std::vector<size_t> norm_to_orig;
  RETURN_IF_ERROR(normalizer_->Normalize(input, &normalized, &norm_to_orig));

  if (nbest_size == 1) {
    const auto result = model_->Encode(normalized);
    RETURN_IF_ERROR(PopulateSentencePieceText(input, normalized, norm_to_orig, result, spt));
  } else if (nbest_size > 1) {
    const auto nbests =
        model_->NBestEncode(normalized, nbest_size);
    if (nbests.empty()) {
      return util::InternalError("NBestEncode returns empty result");
    }

    std::vector<float> probs(nbests.size(), 0.0);
    for (size_t i = 0; i < nbests.size(); ++i) {
      probs[i] = std::exp(alpha * nbests[i].second);
    }

    thread_local static std::mt19937 mt(std::random_device{}());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    RETURN_IF_ERROR(PopulateSentencePieceText(input, normalized, norm_to_orig,
                                              nbests[dist(mt)].first, spt));

  } else if (nbest_size < 0) {
    const auto result = model_->SampleEncode(normalized, alpha);
    RETURN_IF_ERROR(PopulateSentencePieceText(input, normalized, norm_to_orig, result, spt));
  }

  return util::OkStatus();
}

util::Status SentencePieceProcessor::Decode(const std::vector<std::string> &pieces,
                                            SentencePieceText *spt) const {
  CHECK_OR_RETURN_STATUS_PROTO(spt);

  auto DecodeSentencePiece = [&](StringPiece piece, int id,
                                 bool is_bos_ws) -> std::string {
    if (IsControl(id)) {  // <s>, </s>
      return "";          // invisible symbol.
    } else if (IsUnknown(id)) {
      if (IdToPiece(id) == piece) {  // <unk>
        return kUnknownSymbol;
      } else {  // return piece when piece is not <unk>.
        return piece.to_string();
      }
    }

    if (is_bos_ws) {
      // Consume if the current position is bos and
      // piece starts with kSpaceSymbol.
      piece.Consume(kSpaceSymbol);
    }

    return string_util::StringReplace(piece, kSpaceSymbol, " ", true);
  };

  for (const std::string &w : pieces) {
    auto *sp = spt->add_pieces();
    sp->set_piece(w);
    sp->set_id(PieceToId(w));
  }

  RETURN_IF_ERROR(ApplyExtraOptions(decode_extra_options_, spt));

  std::string *text = spt->mutable_text();
  for (auto &sp : *(spt->mutable_pieces())) {
    sp.set_surface(DecodeSentencePiece(sp.piece(), sp.id(), text->empty()));
    sp.set_begin(text->size());
    sp.set_end(text->size() + sp.surface().size());
    *text += sp.surface();
  }

  return util::OkStatus();
}

util::Status SentencePieceProcessor::Decode(const std::vector<int> &ids,
                                            SentencePieceText *spt) const {
  std::vector<std::string> pieces;
  for (const int id : ids) {
    pieces.emplace_back(IdToPiece(id));
  }
  return Decode(pieces, spt);
}

#define CHECK_STATUS_OR_RETURN_DEFAULT(value)                           \
  if (!status().ok()) {                                                 \
    LOG(ERROR) << status().error_message() << "\nReturns default value " << value; \
    return value;                                                       \
  }

int SentencePieceProcessor::GetPieceSize() const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0);
  return model_->GetPieceSize();
}

int SentencePieceProcessor::PieceToId(const std::string &piece) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0);
  return model_->PieceToId(piece);
}

std::string SentencePieceProcessor::IdToPiece(int id) const {
  CHECK_STATUS_OR_RETURN_DEFAULT("");
  return model_->IdToPiece(id);
}

float SentencePieceProcessor::GetScore(int id) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0.0);
  return model_->GetScore(id);
}

bool SentencePieceProcessor::IsControl(int id) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0);
  return model_->IsControl(id);
}

bool SentencePieceProcessor::IsUnknown(int id) const {
  CHECK_STATUS_OR_RETURN_DEFAULT(0);
  return model_->IsUnknown(id);
}

// static
util::Status SentencePieceProcessor::ApplyExtraOptions(
    const std::vector<ExtraOption> &extra_options,
    SentencePieceText *spt) const {
  for (const auto &extra_option : extra_options) {
    switch (extra_option) {
      case REVERSE:
        std::reverse(spt->mutable_pieces()->begin(),
                     spt->mutable_pieces()->end());
        break;
      case EOS: {
        auto *piece = spt->add_pieces();
        piece->set_id(PieceToId("</s>"));
        piece->set_piece("</s>");
      } break;
      case BOS: {
        auto *array = spt->mutable_pieces();
        array->Add();
        for (int i = array->size() - 1; i > 0; --i) {
          array->SwapElements(i - 1, i);
        }
        auto *piece = array->Mutable(0);
        piece->set_id(PieceToId("<s>"));
        piece->set_piece("<s>");
      } break;
      default:
        return util::InternalError("Unknown extra_option type");
    }
  }

  return util::OkStatus();
}

// static
util::Status SentencePieceProcessor::ParseExtraOptions(
    const std::string &extra_option,
    std::vector<SentencePieceProcessor::ExtraOption> *extra_options) {
  extra_options->clear();
  static std::map<std::string, SentencePieceProcessor::ExtraOption>
      extra_option_map = {{"bos", SentencePieceProcessor::BOS},
                          {"eos", SentencePieceProcessor::EOS},
                          {"reverse", SentencePieceProcessor::REVERSE}};
  for (const auto &s : string_util::Split(extra_option, ":")) {
    const auto it = extra_option_map.find(s);
    if (it == extra_option_map.end())
      return util::InternalError(std::string("option ") + s + " is not available.");
    extra_options->push_back(it->second);
  }
  return util::OkStatus();
}

void SentencePieceProcessor::SetModel(std::unique_ptr<ModelInterface> &&model) {
  model_ = std::move(model);
}

void SentencePieceProcessor::SetNormalizer(
    std::unique_ptr<normalizer::Normalizer> &&normalizer) {
  normalizer_ = std::move(normalizer);
}

const ModelProto &SentencePieceProcessor::model_proto() const {
  return *model_proto_;
}
}  // namespace sentencepiece
