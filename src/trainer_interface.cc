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

#include "trainer_interface.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "model_factory.h"
#include "normalizer.h"
#include "sentencepiece_processor.h"
#include "unicode_script.h"
#include "util.h"

namespace sentencepiece {

const char32 TrainerInterface::kWSChar = L'\u2581';
const char TrainerInterface::kWSStr[] = "\xe2\x96\x81";

const char32 TrainerInterface::kUNKChar = L'\u2585';
const char TrainerInterface::kUNKStr[] = "\xe2\x96\x85";

const char32 TrainerInterface::kUPPBoundaryChar = L'\u0009';
const char TrainerInterface::kUPPBoundaryStr[] = "\t";

const char TrainerInterface::kUNK[] = "<unk>";
const char TrainerInterface::kBOS[] = "<s>";
const char TrainerInterface::kEOS[] = "</s>";
const char TrainerInterface::kPAD[] = "<pad>";

namespace {
util::Status VerifySpec(const TrainerSpec &trainer_spec) {
  CHECK_OR_RETURN(!trainer_spec.model_prefix().empty());
  CHECK_GT_OR_RETURN(trainer_spec.input().size(), 0);
  CHECK_GT_OR_RETURN(trainer_spec.vocab_size(), 0);

#define CHECK_RANGE(variable, minval, maxval) \
  CHECK_OR_RETURN(variable >= minval && variable <= maxval)

  CHECK_RANGE(trainer_spec.character_coverage(), 0.98, 1.0);
  CHECK_RANGE(trainer_spec.input_sentence_size(), 100, 100000000);
  CHECK_RANGE(trainer_spec.max_sentencepiece_length(), 1, 512);
  CHECK_RANGE(trainer_spec.mining_sentence_size(), 100, 5000000);
  CHECK_RANGE(trainer_spec.num_sub_iterations(), 1, 10);
  CHECK_RANGE(trainer_spec.num_threads(), 1, 128);
  CHECK_RANGE(trainer_spec.seed_sentencepiece_size(), 1000, 5000000);
  CHECK_RANGE(trainer_spec.shrinking_factor(), 0.5, 0.95);
  CHECK_RANGE(trainer_spec.training_sentence_size(), 100, 100000000);
#undef CHECK_RANGE

  return util::OkStatus();
}
}  // namespace

TrainerInterface::TrainerInterface(const TrainerSpec &trainer_spec,
                                   const NormalizerSpec &normalizer_spec)
    : trainer_spec_(trainer_spec), normalizer_spec_(normalizer_spec) {
  status_ = VerifySpec(trainer_spec_);
  if (status_.ok()) status_ = InitMetaPieces();
}

TrainerInterface::~TrainerInterface() {}

bool TrainerInterface::IsValidSentencePiece(
    const string_util::UnicodeText &sentencepiece) const {
  // Returns false if the length of piece is invalid.
  if (sentencepiece.empty() ||
      sentencepiece.size() >
          static_cast<size_t>(trainer_spec_.max_sentencepiece_length())) {
    return false;
  }

  size_t pos = 0;
  unicode_script::ScriptType prev_script =
      static_cast<unicode_script::ScriptType>(-1);
  for (auto it = sentencepiece.begin(); it != sentencepiece.end(); ++it) {
    if (*it == kUNKChar) {  // UNK must not be included
      return false;
    }
    // kUPPBoundaryChar is included when split_by_upp_for_training is true.
    if (*it == kUPPBoundaryChar) {
      return false;
    }
    if (*it == 0x0020) {
      LOG(WARNING) << "space must not be included in normalized string.";
      return false;
    }
    if (*it == kWSChar) {
      // Only allows whitespace to appear as a prefix of piece.
      // When split_by_whitespace is false, we allow whitespaces to
      // appear in the middle, "foo_bar", but do not allow them
      // to appear as suffix, "foo_bar_".
      // Regardless of the setting of split_by_whitespace,
      // whitespace is treated as a prefix/infix of symbol or
      // independent symbol.
      if ((trainer_spec_.split_by_whitespace() && pos > 0) ||
          (!trainer_spec_.split_by_whitespace() && pos > 0 &&
           pos == sentencepiece.size() - 1)) {
        return false;
      }
    } else {
      auto s = unicode_script::GetScript(*it);
      // Merge Hiragana/Katakana into Han.
      if (s == unicode_script::U_Hiragana || s == unicode_script::U_Katakana ||
          *it == 0x30FC) {  // long vowel sound (Katakana) should be Katakana
        s = unicode_script::U_Han;
      }
      // Do not allow a piece to include multiple Unicode scripts
      // when split_by_unicode_script() is true (default = true).
      if (prev_script != -1 && prev_script != s &&
          trainer_spec_.split_by_unicode_script()) {
        return false;
      }
      prev_script = s;
    }
    ++pos;
  }
  return true;
}

util::Status TrainerInterface::LoadSentences() {
  RETURN_IF_ERROR(status());
  CHECK_OR_RETURN(sentences_.empty());
  CHECK_OR_RETURN(required_chars_.empty());

  const normalizer::Normalizer normalizer(normalizer_spec_);

  CHECK_OR_RETURN(trainer_spec_.input_format().empty() ||
                  trainer_spec_.input_format() == "text" ||
                  trainer_spec_.input_format() == "tsv")
      << "Supported formats are 'text' and 'tsv'.";

  const bool is_tsv = trainer_spec_.input_format() == "tsv";

  for (const auto &filename : trainer_spec_.input()) {
    LOG(INFO) << "Loading corpus: " << filename;
    std::string sentence;
    io::InputBuffer input(filename);
    RETURN_IF_ERROR(input.status());
    while (input.ReadLine(&sentence)) {
      int64 freq = 1;
      if (is_tsv) {
        const std::vector<std::string> v = string_util::Split(sentence, "\t");
        CHECK_EQ_OR_RETURN(v.size(), 2)
            << "Input format must be: word <tab> freq. " << sentence;
        sentence = v[0];
        freq = std::atoll(v[1].c_str());
        CHECK_GE_OR_RETURN(freq, 1);
      }

      constexpr int kMaxLines = 2048;
      if (sentence.size() > kMaxLines) {
        continue;
      }
      if (sentence.find(kUNKStr) != std::string::npos) {
        LOG(INFO) << "Reserved chars are found. Skipped: " << sentence;
        continue;
      }

      // Normalizes sentence with Normalizer.
      // whitespaces are replaced with kWSChar.
      const std::string normalized = normalizer.Normalize(sentence);
      if (sentences_.size() % 100000 == 0) {
        LOG(INFO) << "Loading: " << normalized
                  << "\tsize=" << sentences_.size();
      }
      CHECK_OR_RETURN(normalized.find(" ") == std::string::npos)
          << "Normalized string must not include spaces";
      if (normalized.empty()) {
        LOG(WARNING) << "Empty string found. removed";
        continue;
      }

      sentences_.emplace_back(normalized, freq);

      if (sentences_.size() ==
          static_cast<size_t>(trainer_spec_.input_sentence_size())) {
        goto END;
      }
    }
  }

END:
  LOG(INFO) << "Loaded " << sentences_.size() << " sentences";

  // Count character frequencies.
  int64 all_chars_count = 0;
  std::unordered_map<char32, int64> chars_count;
  for (const auto &w : sentences_) {
    for (const char32 c : string_util::UTF8ToUnicodeText(w.first)) {
      if (c == 0x0020) {
        // UTF8ToUnicodeText returns a white space if the text
        // contains an interchange-invalid character.
        CHECK_OR_RETURN(w.first.find(" ") == std::string::npos)
            << "space must not be included in normalized string.";
        continue;
      }
      chars_count[c] += w.second;
      all_chars_count += w.second;
    }
  }
  LOG(INFO) << "all chars count=" << all_chars_count;

  // Determines required_chars which must be included in the vocabulary.
  int64 accumulated_chars_count = 0;
  for (const auto &w : Sorted(chars_count)) {
    const float coverage = 1.0 * accumulated_chars_count / all_chars_count;
    if (coverage >= trainer_spec_.character_coverage()) {
      LOG(INFO) << "Done: " << 100.0 * coverage << "% characters are covered.";
      break;
    }
    accumulated_chars_count += w.second;
    CHECK_NE_OR_RETURN(w.first, 0x0020)
        << "space must not be included in normalized string.";
    required_chars_.insert(w);
  }
  LOG(INFO) << "alphabet size=" << required_chars_.size();

  CHECK_OR_RETURN(!port::ContainsKey(required_chars_, kUNKChar));

  // Replaces rare characters (characters not included in required_chars_)
  // with kUNKChar.
  for (auto &w : sentences_) {
    string_util::UnicodeText uw2;
    for (const char32 c : string_util::UTF8ToUnicodeText(w.first)) {
      if (port::ContainsKey(required_chars_, c)) {
        uw2.push_back(c);
      } else {
        uw2.push_back(kUNKChar);
      }
    }
    w.first = string_util::UnicodeTextToUTF8(uw2);
  }

  // +3 for meta pieces.
  if (trainer_spec_.model_type() != TrainerSpec::WORD &&
      trainer_spec_.model_type() != TrainerSpec::CHAR) {
    CHECK_LT_OR_RETURN(
        static_cast<int>(required_chars_.size() + meta_pieces_.size()),
        trainer_spec_.vocab_size())
        << "Vocabulary size is smaller than required_chars. "
        << trainer_spec_.vocab_size() << " vs "
        << required_chars_.size() + meta_pieces_.size() << ". "
        << "Increase vocab_size or decrease character_coverage with "
        << "--character_coverage option.";
  }

  LOG(INFO) << "Done! " << sentences_.size() << " sentences are loaded";

  return util::OkStatus();
}

void TrainerInterface::SplitSentencesByWhitespace() {
  LOG(INFO) << "Tokenizing input sentences with whitespace: "
            << sentences_.size();
  std::unordered_map<std::string, int64> tokens;
  for (const auto &s : sentences_) {
    for (const auto &w : SplitIntoWords(s.first)) {
      tokens[w.to_string()] += s.second;
    }
  }
  sentences_ = Sorted(tokens);
  LOG(INFO) << "Done! " << sentences_.size();
}

util::Status TrainerInterface::Serialize(ModelProto *model_proto) const {
  // Duplicated sentencepiece is not allowed.
  std::unordered_set<std::string> dup;

#define CHECK_PIECE(piece)         \
  CHECK_OR_RETURN(!piece.empty()); \
  CHECK_OR_RETURN(dup.insert(piece).second) << piece << " is already defined";

  for (const auto &w : meta_pieces_) {
    auto *sp = model_proto->add_pieces();
    sp->set_piece(w.first);
    sp->set_type(w.second);
    sp->set_score(0.0);
    CHECK_NE_OR_RETURN(ModelProto::SentencePiece::NORMAL, sp->type());
    CHECK_PIECE(sp->piece());
  }

  for (const auto &w : final_pieces_) {
    auto *sp = model_proto->add_pieces();
    sp->set_piece(w.first);
    sp->set_score(w.second);
    CHECK_PIECE(sp->piece());
  }

  *(model_proto->mutable_trainer_spec()) = trainer_spec_;
  *(model_proto->mutable_normalizer_spec()) = normalizer_spec_;

  if (!trainer_spec_.hard_vocab_limit() ||
      trainer_spec_.model_type() == TrainerSpec::CHAR) {
    CHECK_GE_OR_RETURN(trainer_spec_.vocab_size(), model_proto->pieces_size());
    CHECK_GE_OR_RETURN(trainer_spec_.vocab_size(),
                       static_cast<int>(dup.size()));
    model_proto->mutable_trainer_spec()->set_vocab_size(
        model_proto->pieces_size());
  } else {
    CHECK_OR_RETURN(trainer_spec_.vocab_size() == model_proto->pieces_size() &&
                    trainer_spec_.vocab_size() == static_cast<int>(dup.size()))
        << "Use --hard_vocab_limit=false to make the vocab size `soft limit`.";
  }

  return util::OkStatus();
}

util::Status TrainerInterface::SaveModel(StringPiece filename) const {
  LOG(INFO) << "Saving model: " << filename;
  ModelProto model_proto;
  RETURN_IF_ERROR(Serialize(&model_proto));
  std::ofstream ofs(filename.data(), OUTPUT_MODE);
  CHECK_OR_RETURN(ofs) << "\"" << filename.data()
                       << "\": " << std::strerror(errno);
  CHECK_OR_RETURN(model_proto.SerializeToOstream(&ofs));
  return util::OkStatus();
}

util::Status TrainerInterface::SaveVocab(StringPiece filename) const {
  LOG(INFO) << "Saving vocabs: " << filename;
  ModelProto model_proto;
  Serialize(&model_proto);
  io::OutputBuffer output(filename);
  RETURN_IF_ERROR(output.status());

  for (const auto &piece : model_proto.pieces()) {
    std::ostringstream os;
    os << piece.piece() << "\t" << piece.score();
    CHECK_OR_RETURN(output.WriteLine(os.str()));
  }

  return util::OkStatus();
}

util::Status TrainerInterface::Save() const {
  RETURN_IF_ERROR(SaveModel(trainer_spec_.model_prefix() + ".model"));
  RETURN_IF_ERROR(SaveVocab(trainer_spec_.model_prefix() + ".vocab"));
  return util::OkStatus();
}

util::Status TrainerInterface::InitMetaPieces() {
  CHECK_OR_RETURN(meta_pieces_.empty());

  std::vector<std::pair<int, std::string>> ids;
  if (trainer_spec_.unk_id() >= 0)
    ids.emplace_back(trainer_spec_.unk_id(), kUNK);
  if (trainer_spec_.bos_id() >= 0)
    ids.emplace_back(trainer_spec_.bos_id(), kBOS);
  if (trainer_spec_.eos_id() >= 0)
    ids.emplace_back(trainer_spec_.eos_id(), kEOS);
  if (trainer_spec_.pad_id() >= 0)
    ids.emplace_back(trainer_spec_.pad_id(), kPAD);

  std::sort(ids.begin(), ids.end());

  int prev_id = -1;
  bool has_unk = false;
  for (const auto &p : ids) {
    CHECK_EQ_OR_RETURN(prev_id + 1, p.first)
        << "ID for `" << p.second << "` must be " << prev_id + 1;
    prev_id = p.first;
    CHECK_EQ_OR_RETURN(static_cast<int>(meta_pieces_.size()), p.first);
    if (p.second == kUNK) has_unk = true;
    meta_pieces_.emplace_back(
        p.second, (p.second == kUNK ? ModelProto::SentencePiece::UNKNOWN
                                    : ModelProto::SentencePiece::CONTROL));
  }

  CHECK_OR_RETURN(has_unk) << kUNK << " must be defined.";

  for (const auto &w : trainer_spec_.control_symbols()) {
    meta_pieces_.emplace_back(w, ModelProto::SentencePiece::CONTROL);
  }

  for (const auto &w : trainer_spec_.user_defined_symbols()) {
    meta_pieces_.emplace_back(w, ModelProto::SentencePiece::USER_DEFINED);
  }

  return util::OkStatus();
}

}  // namespace sentencepiece
