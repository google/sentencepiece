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

bool SentencePieceProcessor::Load(const std::string &filename) {
  std::ifstream ifs(filename.c_str(), std::ios::binary | std::ios::in);
  if (!ifs) {
    LOG(WARNING) << "Cannot open " << filename;
    return false;
  }

  return Load(&ifs);
}

bool SentencePieceProcessor::Load(std::istream *is) {
  CHECK_NOTNULL(is);

  model_proto_ = port::MakeUnique<ModelProto>();
  if (!model_proto_->ParseFromIstream(is)) {
    LOG(WARNING) << "Model file is broken";
    return false;
  }

  model_ = ModelFactory::Create(*model_proto_);
  normalizer_ =
      port::MakeUnique<normalizer::Normalizer>(model_proto_->normalizer_spec());

  return true;
}

void SentencePieceProcessor::LoadOrDie(const std::string &filename) {
  CHECK(Load(filename)) << "failed to load model: " << filename;
}

void SentencePieceProcessor::LoadOrDie(std::istream *is) {
  CHECK(Load(is)) << "failed to load model";
}

void SentencePieceProcessor::SetEncodeExtraOptions(
    const std::string &extra_options) {
  encode_extra_options_ = ParseExtraOptions(extra_options);
}

void SentencePieceProcessor::SetDecodeExtraOptions(
    const std::string &extra_options) {
  decode_extra_options_ = ParseExtraOptions(extra_options);
}

//////////////////////////////////////////////////////////////
// Simple API.
void SentencePieceProcessor::Encode(const std::string &input,
                                    std::vector<std::string> *pieces) const {
  CHECK_NOTNULL(pieces)->clear();

  SentencePieceText spt;
  Encode(input, &spt);
  for (const auto &sp : spt.pieces()) {
    pieces->emplace_back(sp.piece());
  }
}

void SentencePieceProcessor::Encode(const std::string &input,
                                    std::vector<int> *ids) const {
  CHECK_NOTNULL(ids)->clear();

  SentencePieceText spt;
  Encode(input, &spt);
  for (const auto &sp : spt.pieces()) {
    ids->emplace_back(sp.id());
  }
}

void SentencePieceProcessor::Decode(const std::vector<std::string> &pieces,
                                    std::string *detokenized) const {
  CHECK_NOTNULL(detokenized);
  SentencePieceText spt;

  Decode(pieces, &spt);
  *detokenized = std::move(spt.text());
}

void SentencePieceProcessor::Decode(const std::vector<int> &ids,
                                    std::string *detokenized) const {
  CHECK_NOTNULL(detokenized);
  SentencePieceText spt;

  Decode(ids, &spt);
  *detokenized = std::move(spt.text());
}

void SentencePieceProcessor::NBestEncode(
    const std::string &input, int nbest_size,
    std::vector<std::vector<std::string>> *pieces) const {
  CHECK_NOTNULL(pieces)->clear();

  NBestSentencePieceText spt;
  NBestEncode(input, nbest_size, &spt);
  for (const auto &nbest : spt.nbests()) {
    std::vector<std::string> result;
    for (const auto &sp : nbest.pieces()) {
      result.emplace_back(sp.piece());
    }
    pieces->emplace_back(result);
  }
}

void SentencePieceProcessor::NBestEncode(
    const std::string &input, int nbest_size,
    std::vector<std::vector<int>> *ids) const {
  CHECK_NOTNULL(ids)->clear();

  NBestSentencePieceText spt;
  NBestEncode(input, nbest_size, &spt);
  for (const auto &nbest : spt.nbests()) {
    std::vector<int> result;
    for (const auto &sp : nbest.pieces()) {
      result.emplace_back(sp.id());
    }
    ids->emplace_back(result);
  }
}

void SentencePieceProcessor::SampleEncode(
    const std::string &input, int nbest_size, float alpha,
    std::vector<std::string> *pieces) const {
  CHECK_NOTNULL(pieces)->clear();

  SentencePieceText spt;
  SampleEncode(input, nbest_size, alpha, &spt);
  for (const auto &sp : spt.pieces()) {
    pieces->emplace_back(sp.piece());
  }
}

void SentencePieceProcessor::SampleEncode(const std::string &input,
                                          int nbest_size, float alpha,
                                          std::vector<int> *ids) const {
  CHECK_NOTNULL(ids)->clear();

  SentencePieceText spt;
  SampleEncode(input, nbest_size, alpha, &spt);
  for (const auto &sp : spt.pieces()) {
    ids->emplace_back(sp.id());
  }
}

void SentencePieceProcessor::PopulateSentencePieceText(
    const std::string &input, const std::string &normalized,
    const std::vector<size_t> &norm_to_orig, const EncodeResult &result,
    SentencePieceText *spt) const {
  size_t consumed = 0;
  bool is_prev_unk = false;
  for (const auto &p : result) {
    const StringPiece w = p.first;  // piece
    const int id = p.second;        // id
    CHECK(!w.empty());
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
      CHECK_LE(begin, norm_to_orig.size());
      CHECK_LE(end, norm_to_orig.size());
      const size_t orig_begin = norm_to_orig[begin];
      const size_t orig_end = norm_to_orig[end];
      CHECK_LE(orig_begin, input.size());
      CHECK_LE(orig_end, input.size());
      CHECK_LE(orig_begin, orig_end);
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
  CHECK_EQ(consumed, normalized.size());

  ApplyExtraOptions(encode_extra_options_, spt);

  spt->set_text(input);
}

void SentencePieceProcessor::Encode(const std::string &input,
                                    SentencePieceText *spt) const {
  CHECK_NOTNULL(spt)->Clear();

  std::string normalized;
  std::vector<size_t> norm_to_orig;
  CHECK_NOTNULL(normalizer_)->Normalize(input, &normalized, &norm_to_orig);

  const auto result = CHECK_NOTNULL(model_)->Encode(normalized);
  PopulateSentencePieceText(input, normalized, norm_to_orig, result, spt);
}

void SentencePieceProcessor::NBestEncode(
    const std::string &input, int nbest_size,
    NBestSentencePieceText *nbest_spt) const {
  CHECK_NOTNULL(nbest_spt)->Clear();

  std::string normalized;
  std::vector<size_t> norm_to_orig;
  CHECK_NOTNULL(normalizer_)->Normalize(input, &normalized, &norm_to_orig);

  const auto nbests =
      CHECK_NOTNULL(model_)->NBestEncode(normalized, nbest_size);
  for (const auto &result : nbests) {
    auto *spt = nbest_spt->add_nbests();
    spt->set_score(result.second);
    PopulateSentencePieceText(input, normalized, norm_to_orig, result.first,
                              spt);
  }
}

void SentencePieceProcessor::SampleEncode(const std::string &input,
                                          int nbest_size, float alpha,
                                          SentencePieceText *spt) const {
  CHECK_NOTNULL(spt)->Clear();

  CHECK_LE(nbest_size, 512)
      << "Too big nbest size. consider using nbest <= 512.";
  CHECK_NE(nbest_size, 0) << "nbest size must not be zero.";

  std::string normalized;
  std::vector<size_t> norm_to_orig;
  CHECK_NOTNULL(normalizer_)->Normalize(input, &normalized, &norm_to_orig);

  if (nbest_size == 1) {
    const auto result = CHECK_NOTNULL(model_)->Encode(normalized);
    PopulateSentencePieceText(input, normalized, norm_to_orig, result, spt);
  } else if (nbest_size > 1) {
    const auto nbests =
        CHECK_NOTNULL(model_)->NBestEncode(normalized, nbest_size);
    CHECK(!nbests.empty());

    std::vector<float> probs(nbests.size(), 0.0);
    for (size_t i = 0; i < nbests.size(); ++i) {
      probs[i] = std::exp(alpha * nbests[i].second);
    }

    thread_local static std::mt19937 mt(std::random_device{}());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    PopulateSentencePieceText(input, normalized, norm_to_orig,
                              nbests[dist(mt)].first, spt);

  } else if (nbest_size < 0) {
    const auto result = CHECK_NOTNULL(model_)->SampleEncode(normalized, alpha);
    PopulateSentencePieceText(input, normalized, norm_to_orig, result, spt);
  }
}

void SentencePieceProcessor::Decode(const std::vector<std::string> &pieces,
                                    SentencePieceText *spt) const {
  CHECK_NOTNULL(spt)->Clear();

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

  ApplyExtraOptions(decode_extra_options_, spt);

  std::string *text = spt->mutable_text();
  for (auto &sp : *(spt->mutable_pieces())) {
    sp.set_surface(DecodeSentencePiece(sp.piece(), sp.id(), text->empty()));
    sp.set_begin(text->size());
    sp.set_end(text->size() + sp.surface().size());
    *text += sp.surface();
  }
}

void SentencePieceProcessor::Decode(const std::vector<int> &ids,
                                    SentencePieceText *spt) const {
  std::vector<std::string> pieces;
  for (const int id : ids) {
    pieces.emplace_back(IdToPiece(id));
  }
  return Decode(pieces, spt);
}

int SentencePieceProcessor::GetPieceSize() const {
  return CHECK_NOTNULL(model_)->GetPieceSize();
}

int SentencePieceProcessor::PieceToId(const std::string &piece) const {
  return CHECK_NOTNULL(model_)->PieceToId(piece);
}

std::string SentencePieceProcessor::IdToPiece(int id) const {
  return CHECK_NOTNULL(model_)->IdToPiece(id);
}

float SentencePieceProcessor::GetScore(int id) const {
  return CHECK_NOTNULL(model_)->GetScore(id);
}

bool SentencePieceProcessor::IsControl(int id) const {
  return CHECK_NOTNULL(model_)->IsControl(id);
}

bool SentencePieceProcessor::IsUnknown(int id) const {
  return CHECK_NOTNULL(model_)->IsUnknown(id);
}

// static
void SentencePieceProcessor::ApplyExtraOptions(
    const std::vector<ExtraOption> &extra_options,
    SentencePieceText *spt) const {
  constexpr int kBOS = 1;
  constexpr int kEOS = 2;

  for (const auto &extra_option : extra_options) {
    switch (extra_option) {
      case REVERSE:
        std::reverse(spt->mutable_pieces()->begin(),
                     spt->mutable_pieces()->end());
        break;
      case EOS: {
        auto *piece = spt->add_pieces();
        piece->set_id(kEOS);
        piece->set_piece(IdToPiece(kEOS));
      } break;
      case BOS: {
        auto *array = spt->mutable_pieces();
        array->Add();
        for (int i = array->size() - 1; i > 0; --i) {
          array->SwapElements(i - 1, i);
        }
        auto *piece = array->Mutable(0);
        piece->set_id(kBOS);
        piece->set_piece(IdToPiece(kBOS));
      } break;
      default:
        LOG(FATAL) << "Unknown extra_option type: "
                   << static_cast<int>(extra_option);
    }
  }
}

// static
std::vector<SentencePieceProcessor::ExtraOption>
SentencePieceProcessor::ParseExtraOptions(const std::string &extra_option) {
  static std::map<std::string, SentencePieceProcessor::ExtraOption>
      extra_option_map = {{"bos", SentencePieceProcessor::BOS},
                          {"eos", SentencePieceProcessor::EOS},
                          {"reverse", SentencePieceProcessor::REVERSE}};
  std::vector<SentencePieceProcessor::ExtraOption> extra_options;
  for (const auto &s : string_util::Split(extra_option, ":")) {
    extra_options.push_back(port::FindOrDie(extra_option_map, s));
  }
  return extra_options;
}

void SentencePieceProcessor::SetModel(std::unique_ptr<ModelInterface> &&model) {
  model_ = std::move(model);
}

void SentencePieceProcessor::SetNormalizer(
    std::unique_ptr<normalizer::Normalizer> &&normalizer) {
  normalizer_ = std::move(normalizer);
}

const ModelProto &SentencePieceProcessor::model_proto() const {
  CHECK_NOTNULL(model_proto_);
  return *model_proto_;
}
}  // namespace sentencepiece
