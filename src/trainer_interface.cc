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

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "filesystem.h"
#include "model_factory.h"
#include "model_interface.h"
#include "normalizer.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_trainer.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/container/flat_hash_set.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/random/random.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/match.h"
#include "third_party/absl/strings/numbers.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_format.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/str_replace.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/absl/strings/string_view.h"
#include "unicode_script.h"

// NOTE: The following flags are experimental options for new L1 sparse Unigram
// training. They will eventually be migrated to TrainerSpec
// (trainer_spec.proto).
ABSL_FLAG(bool, use_sparse_pruning, false,
          "Use continuous L1 sparse pruning in Unigram EM");
ABSL_FLAG(bool, auto_character_coverage, false,
          "When true, disables character_coverage mandatory inclusion and rare "
          "character UNK replacement in Unigram EM.");
ABSL_FLAG(float, fixed_sparse_lambda, 0.0f,
          "Fixed L1 regularization parameter lambda for constant exchange rate "
          "pruning. When 0.0 (default), dynamic Quantile Annealing "
          "automatically estimates lambda to match target vocab_size K.");
ABSL_FLAG(
    bool, post_l1_debias, true,
    "When true (default), resets lambda penalty and re-estimates pure Unigram "
    "MLE probabilities on the selected vocabulary (Post-Lasso Debiased Mode).");
ABSL_FLAG(float, seed_piece_length_power, 1.0f,
          "power exponent beta for seed sentencepiece length weighting: score = freq * len^beta (default=1.0)");
ABSL_FLAG(float, min_freq_alpha, 0.0f,
          "dynamic minimum frequency factor alpha derived from Zipf law: min_freq = max(2, alpha * N / (K * ln(K))) (default=0.0)");

namespace sentencepiece {

const char32_t TrainerInterface::kWSChar = U'\u2581';
const char TrainerInterface::kWSStr[] = "\xe2\x96\x81";

const char32_t TrainerInterface::kUNKChar = U'\u2585';
const char TrainerInterface::kUNKStr[] = "\xe2\x96\x85";

const char32_t TrainerInterface::kPretokenizationBoundaryChar = U'\u001f';
const char TrainerInterface::kPretokenizationBoundaryStr[] = "\x1f";

namespace {
absl::Status VerifySpec(const TrainerSpec& trainer_spec,
                        const TrainerComponents& components) {
  RET_CHECK_GT(trainer_spec.vocab_size(), 0);

  if (trainer_spec.model_type() == TrainerSpec::UNIGRAM ||
      trainer_spec.model_type() == TrainerSpec::BPE) {
    RET_CHECK(!trainer_spec.use_all_vocab())
        << "--use_all_vocab=true is valid for WORD/CHAR model.";
  }

  if (!trainer_spec.seed_sentencepieces_file().empty()) {
    RET_CHECK(trainer_spec.model_type() == TrainerSpec::UNIGRAM)
        << "seed_sentencepieces_file is only supported for UNIGRAM model.";
  }

#define CHECK_RANGE(variable, minval, maxval) \
  RET_CHECK(variable >= minval && variable <= maxval)

  CHECK_RANGE(trainer_spec.character_coverage(), 0.98, 1.0);
  CHECK_RANGE(trainer_spec.max_sentencepiece_length(), 1, 512);
  CHECK_RANGE(trainer_spec.num_sub_iterations(), 1, 10);
  CHECK_RANGE(trainer_spec.num_threads(), 1, 1024);
  CHECK_RANGE(trainer_spec.self_test_sample_size(), 0, 1000);
  CHECK_RANGE(trainer_spec.shrinking_factor(), 0.5, 0.95);
  CHECK_RANGE(trainer_spec.max_sentence_length(), 10, 1073741824);
#undef CHECK_RANGE

  RET_CHECK(trainer_spec.input_sentence_size() <= 0 ||
            trainer_spec.input_sentence_size() > 100);

  RET_CHECK(!trainer_spec.unk_piece().empty());
  RET_CHECK(!trainer_spec.bos_piece().empty());
  RET_CHECK(!trainer_spec.eos_piece().empty());
  RET_CHECK(!trainer_spec.pad_piece().empty());

  if (components.pretokenizer != nullptr ||
      !trainer_spec.pretokenization_delimiter().empty()) {
    RET_CHECK(trainer_spec.model_type() == TrainerSpec::UNIGRAM ||
              trainer_spec.model_type() == TrainerSpec::BPE)
        << "PretokenizerForTraining is only supported in UNIGRAM or BPE mode.";
    RET_CHECK(components.pretokenizer == nullptr ||
              trainer_spec.pretokenization_delimiter().empty())
        << "pretokenizer callback and pretokenization_delimiter cannot be set "
           "simultaneously.";
  }

  if (absl::GetFlag(FLAGS_auto_character_coverage)) {
    RET_CHECK(trainer_spec.model_type() == TrainerSpec::UNIGRAM)
        << "--auto_character_coverage is only supported in UNIGRAM model mode.";
    RET_CHECK(trainer_spec.byte_fallback())
        << "--auto_character_coverage requires --byte_fallback=true.";
    RET_CHECK(trainer_spec.required_chars().empty())
        << "--auto_character_coverage cannot be used together with "
           "--required_chars or --required_chars_file.";
  }

  return absl::OkStatus();
}

bool is_unicode_decimal_number(char32_t c) {
  return (c >= 0x30 && c <= 0x39) || (c >= 0xff10 && c <= 0xff19);
}

class SentenceSelector {
 public:
  using Sampler = random::ReservoirSampler<TrainerInterface::Sentence>;

  static constexpr int64_t kTooBigSentencesSize = 1000000;

  SentenceSelector(TrainerInterface::Sentences* sentences,
                   const TrainerSpec& spec)
      : sentences_(sentences), spec_(&spec) {
    if (spec_->input_sentence_size() > 0) {
      if (spec_->shuffle_input_sentence()) {
        constexpr size_t kSeed = 12345678;
        sampler_ = std::make_unique<Sampler>(
            sentences, spec_->input_sentence_size(), kSeed);
      } else {
        LOG(INFO)
            << "First " << spec_->input_sentence_size()
            << " sentences are selected. Remaining sentences are discarded.";
      }
    }
  }

  void Finish() const {
    if (sentences_->size() > kTooBigSentencesSize) {
      LOG(WARNING) << "Too many sentences are loaded! (" << sentences_->size()
                   << "), which may slow down training.";
      LOG(WARNING) << "Consider using "
                      "--input_sentence_size=<size> and "
                      "--shuffle_input_sentence=true.";
      LOG(WARNING) << "They allow to randomly sample <size> sentences from "
                      "the entire corpus.";
    }
  }

  bool Add(const std::pair<std::string, int64_t>& sentence) {
    if (spec_->input_sentence_size() == 0) {
      sentences_->emplace_back(sentence);
    } else {
      if (spec_->shuffle_input_sentence()) {
        sampler_->Add(sentence);
      } else {
        sentences_->emplace_back(sentence);
        if (sentences_->size() >= spec_->input_sentence_size()) {
          return false;
        }
      }
    }

    if (total_size() > 0 && total_size() % kTooBigSentencesSize == 0) {
      LOG(INFO) << "Loaded " << total_size() << " lines";
    }

    return true;
  }

  [[nodiscard]] size_t total_size() const {
    return (sampler_ != nullptr) ? sampler_->total_size() : sentences_->size();
  }

 private:
  TrainerInterface::Sentences* sentences_ = nullptr;
  const TrainerSpec* spec_ = nullptr;
  std::unique_ptr<Sampler> sampler_;
};
}  // namespace

MultiFileSentenceIterator::MultiFileSentenceIterator(
    const std::vector<std::string>& files)
    : files_(files) {
  Next();
}

bool MultiFileSentenceIterator::done() const {
  return (!read_done_ && file_index_ == files_.size());
}

absl::Status MultiFileSentenceIterator::status() const {
  RET_CHECK(fp_);
  return fp_->status();
}

void MultiFileSentenceIterator::Next() {
  TryRead();

  if (!read_done_ && file_index_ < files_.size()) {
    const auto& filename = files_[file_index_++];
    fp_ = filesystem::NewReadableFile(filename);
    LOG(INFO) << "Loading corpus: " << filename;
    if (fp_->status() != absl::OkStatus()) {
      file_index_ = files_.size();
      read_done_ = false;
      return;
    }

    TryRead();
  }
}

void MultiFileSentenceIterator::TryRead() {
  read_done_ = fp_ && fp_->ReadLine(&value_);
}

TrainerInterface::TrainerInterface(TrainerSpec trainer_spec,
                                   NormalizerSpec normalizer_spec,
                                   NormalizerSpec denormalizer_spec)
    : trainer_spec_(std::move(trainer_spec)),
      normalizer_spec_(std::move(normalizer_spec)),
      denormalizer_spec_(std::move(denormalizer_spec)) {
  status_ = VerifySpec(trainer_spec_, components_);
  if (status_.ok()) {
    status_ = InitMetaPieces();
  }
}

TrainerInterface::~TrainerInterface() = default;

bool TrainerInterface::IsValidSentencePiece(
    const string_util::UnicodeText& sentencepiece) const {
  // Returns false if the length of piece is invalid.
  if (sentencepiece.empty() ||
      sentencepiece.size() >
          static_cast<size_t>(trainer_spec_.max_sentencepiece_length())) {
    return false;
  }

  constexpr auto kAnyType = static_cast<unicode_script::ScriptType>(
      std::numeric_limits<char32_t>::max());

  unicode_script::ScriptType prev_script = kAnyType;
  bool all_whitespace_piece =
      std::all_of(sentencepiece.begin(), sentencepiece.end(),
                  [](char32_t c) { return c == kWSChar; });

  for (size_t pos = 0; pos < sentencepiece.size(); ++pos) {
    const char32_t c = sentencepiece[pos];
    if (c == kUNKChar) {  // UNK must not be included
      return false;
    }
    if (c == 0x0000) {  // NULL is not allowed for Darts (TRIE).
      return false;
    }
    // kPretokenizationBoundaryChar is included when pretokenizer is set.
    if (c == kPretokenizationBoundaryChar) {
      return false;
    }
    if (c == 0x0020) {
      LOG(WARNING) << "space must not be included in normalized string.";
      return false;
    }
    if (!string_util::IsValidCodepoint(c)) {
      return false;
    }

    if (c == kWSChar) {
      // Only allows whitespace to appear as a prefix of piece unless
      // allow_whitespace_only_pieces is True.
      // When split_by_whitespace is false, we allow whitespaces to
      // appear in the middle, "foo_bar", but do not allow them
      // to appear as suffix, "foo_bar_".
      // Regardless of the setting of split_by_whitespace,
      // whitespace is treated as a prefix/infix of symbol or
      // independent symbol, unless allow_whitespace_only_pieces() is true,
      // in which case whitespace only pieces can occur.
      if (!trainer_spec_.allow_whitespace_only_pieces() ||
          !all_whitespace_piece) {
        if (trainer_spec_.treat_whitespace_as_suffix()) {
          if ((trainer_spec_.split_by_whitespace() &&
               pos < sentencepiece.size() - 1) ||
              (!trainer_spec_.split_by_whitespace() &&
               pos < sentencepiece.size() - 1 && pos == 0)) {
            return false;
          }
        } else {
          if ((trainer_spec_.split_by_whitespace() && pos > 0) ||
              (!trainer_spec_.split_by_whitespace() && pos > 0 &&
               pos == sentencepiece.size() - 1)) {
            return false;
          }
        }
      }
    } else {
      auto s = unicode_script::GetScript(c);

      // Merge Hiragana/Katakana into Han.
      if (s == unicode_script::U_Hiragana || s == unicode_script::U_Katakana ||
          c == 0x30FC) {  // long vowel sound (Katakana) should be Katakana
        s = unicode_script::U_Han;
      } else if (s == unicode_script::U_Inherited) {
        s = prev_script;
      }

      if (!trainer_spec_.split_by_number() && is_unicode_decimal_number(c)) {
        s = kAnyType;
      }

      if (trainer_spec_.split_digits() && is_unicode_decimal_number(c)) {
        if (sentencepiece.size() > 1) {
          return false;
        }
      }

      // Do not allow a piece to include multiple Unicode scripts
      // when split_by_unicode_script() is true (default = true).
      if (trainer_spec_.split_by_unicode_script() && s != kAnyType &&
          prev_script != kAnyType && prev_script != s) {
        return false;
      }

      prev_script = s;
    }
  }
  return true;
}

absl::Status TrainerInterface::LoadSentences() {
  RETURN_IF_ERROR(status());
  RETURN_IF_ERROR(VerifySpec(trainer_spec_, components_));
  RET_CHECK(sentences_.empty());
  RET_CHECK(required_chars_.empty());
  RET_CHECK(trainer_spec_.input_format().empty() ||
            trainer_spec_.input_format() == "text" ||
            trainer_spec_.input_format() == "tsv")
      << "Supported formats are 'text' and 'tsv'.";

  RET_CHECK((components_.sentence_iterator != nullptr &&
             trainer_spec_.input().empty()) ||
            (components_.sentence_iterator == nullptr &&
             !trainer_spec_.input().empty()))
      << "SentenceIterator and trainer_spec.input() must be exclusive.";

  RET_CHECK(
      (output_model_proto_ != nullptr &&
       trainer_spec_.model_prefix().empty()) ||
      (output_model_proto_ == nullptr && !trainer_spec_.model_prefix().empty()))
      << "ModelProto and trainer_spec.model_prefix() must be exclusive.";

  const bool is_tsv = trainer_spec_.input_format() == "tsv";

  SentenceSelector selector(&sentences_, trainer_spec_);

  int too_long_lines = 0;

  std::unique_ptr<SentenceIterator> sentence_iterator_impl;
  if (components_.sentence_iterator == nullptr) {
    LOG(INFO) << "SentenceIterator is not specified. Using "
                 "MultiFileSentenceIterator.";
    sentence_iterator_impl =
        std::make_unique<MultiFileSentenceIterator>(std::vector<std::string>(
            trainer_spec_.input().begin(), trainer_spec_.input().end()));
    components_.sentence_iterator = sentence_iterator_impl.get();
  }

  for (; !components_.sentence_iterator->done();
       components_.sentence_iterator->Next()) {
    int64_t freq = 1;
    std::string sentence = components_.sentence_iterator->value();

    if (is_tsv) {
      const std::vector<std::string> v = absl::StrSplit(sentence, '\t');
      RET_CHECK_EQ(v.size(), 2)
          << "Input format must be: word <tab> freq. " << sentence;
      sentence = v[0];
      RET_CHECK(absl::SimpleAtoi(v[1], &freq))
          << "Could not parse the frequency";
      RET_CHECK_GE(freq, 1);
    }

    if (sentence.empty()) {
      continue;
    }

    if (static_cast<int>(sentence.size()) >
        trainer_spec_.max_sentence_length()) {
      if (too_long_lines == 0) {
        LOG(WARNING) << "Found too long line (" << sentence.size() << " > "
                     << trainer_spec_.max_sentence_length() << ").";
        LOG(WARNING) << "Too long lines are skipped in the training.";
        LOG(WARNING) << "The maximum length can be changed with "
                        "--max_sentence_length=<size> flag.";
      }
      ++too_long_lines;
      continue;
    }

    if (sentence.find(kUNKStr) != std::string::npos) {
      LOG(INFO) << "Reserved chars are found. Skipped: " << sentence;
      continue;
    }

    if (!selector.Add(std::make_pair(sentence, freq))) {
      goto END;
    }
  }

  RETURN_IF_ERROR(components_.sentence_iterator->status());

END:
  // Emits error message if any.
  selector.Finish();

  if (sentences_.size() == selector.total_size()) {
    LOG(INFO) << "Loaded all " << sentences_.size() << " sentences";
  } else {
    LOG(INFO) << "Sampled " << sentences_.size() << " sentences from "
              << selector.total_size() << " sentences.";
  }

  if (too_long_lines > 0) {
    LOG(INFO) << "Skipped " << too_long_lines << " too long sentences.";
  }

  // Normalize and removes empty string.
  {
    const normalizer::Normalizer normalizer(normalizer_spec_, trainer_spec_);
    std::set<absl::string_view> meta_pieces_set;
    for (const auto& it : meta_pieces_) {
      LOG(INFO) << "Adding meta_piece: " << it.second.first;
      meta_pieces_set.insert(it.second.first);
    }
    const normalizer::PrefixMatcher meta_pieces_matcher(meta_pieces_set);

    LOG(INFO) << "Normalizing sentences...";
    RET_CHECK(!sentences_.empty());

    ThreadPool pool(trainer_spec_.num_threads());
    RETURN_IF_ERROR(RunBatch(
        sentences_.size(),
        [&](size_t i) -> absl::Status {
          auto* s = &sentences_[i].first;
          *s = meta_pieces_matcher.GlobalReplace(normalizer.Normalize(*s),
                                                 kPretokenizationBoundaryStr);
          if (components_.pretokenizer) {
            const auto chunks = components_.pretokenizer(*s);
            if (!components_.allow_inconsistent_pretokenization) {
              const std::string joined = absl::StrJoin(chunks, "");
              if (joined != *s) {
                return absl::InvalidArgumentError(absl::StrCat(
                    "Pretokenized output mismatch at sample ", i, ": joined='",
                    joined, "', original='", *s,
                    "'. Set allow_inconsistent_pretokenization=true in "
                    "TrainerComponents to bypass."));
              }
            }
            *s = absl::StrJoin(chunks, kPretokenizationBoundaryStr);
          } else if (!trainer_spec_.pretokenization_delimiter().empty()) {
            *s = absl::StrReplaceAll(
                *s, {{trainer_spec_.pretokenization_delimiter(),
                      kPretokenizationBoundaryStr}});
          }
          return absl::OkStatus();
        },
        pool));

    for (size_t i = 0; i < sentences_.size(); ++i) {
      auto* s = &sentences_[i].first;
      RET_CHECK(s->find(" ") == std::string::npos)
          << "Normalized string must not include spaces";
      if (s->empty()) {
        std::swap(sentences_[i], sentences_[sentences_.size() - 1]);
        sentences_.resize(sentences_.size() - 1);
      }
    }
  }

  // Generates `required_chars_` with trainer_spec_.character_coverage().
  // required_chars_ are always populated to the final vocab.
  if (!absl::GetFlag(FLAGS_auto_character_coverage)) {
    // Count character frequencies.
    int64_t all_chars_count = 0;
    // A map from a character to {is_required_char, character count}.
    absl::flat_hash_map<char32_t, std::pair<bool, int64_t>> chars_count;
    for (const char32_t c :
         string_util::UTF8ToUnicodeText(trainer_spec_.required_chars())) {
      RET_CHECK(string_util::IsValidCodepoint(c));
      if (c == 0x0000) {
        LOG(INFO) << "Found null character. The required_chars field must be "
                     "encoded in utf-8.";
        continue;
      }
      chars_count[c].first = true;  // is_required_character.
    }

    for (const auto& w : sentences_) {
      for (const char32_t c : string_util::UTF8ToUnicodeText(w.first)) {
        if (!string_util::IsValidCodepoint(c)) {
          continue;
        }
        if (c == 0x0000) {
          LOG(INFO)
              << "Found null character. The corpus must be encoded in utf-8.";
          continue;
        }
        if (c == 0x0020) {
          // UTF8ToUnicodeText returns a white space if the text
          // contains an interchange-invalid character.
          RET_CHECK(w.first.find(" ") == std::string::npos)
              << "space must not be included in normalized string.";
          continue;
        }
        chars_count[c].second += w.second;
        all_chars_count += w.second;
      }
    }
    LOG(INFO) << "all chars count=" << all_chars_count;

    // Determines required_chars which must be included in the vocabulary.
    int64_t accumulated_chars_count = 0;

    // Sorted() sorts the chars_count values in the decsending order of pair<>.
    // I.e. characters are sorted in the order of required characters and then
    // frequent characters.
    for (const auto& w : Sorted(chars_count)) {
      const float coverage = 1.0 * accumulated_chars_count / all_chars_count;
      if (!trainer_spec_.use_all_vocab() &&
          coverage >= trainer_spec_.character_coverage()) {
        LOG(INFO) << "Done: " << 100.0 * coverage
                  << "% characters are covered.";
        break;
      }
      accumulated_chars_count += w.second.second;
      RET_CHECK_NE(w.first, 0x0020)
          << "space must not be included in normalized string.";
      if (w.first == kPretokenizationBoundaryChar) {
        continue;  // Boundary character is not included.
      }
      required_chars_.emplace(w.first, w.second.second);
    }

    LOG(INFO) << "Alphabet size=" << required_chars_.size();
    LOG(INFO) << "Final character coverage="
              << 1.0 * accumulated_chars_count / all_chars_count;

    RET_CHECK(!required_chars_.contains(kUNKChar));

    // Replaces rare characters (characters not included in required_chars_)
    // with kUNKChar.
    for (auto& w : sentences_) {
      string_util::UnicodeText uw2;
      for (const char32_t c : string_util::UTF8ToUnicodeText(w.first)) {
        if (required_chars_.contains(c)) {
          uw2.push_back(c);
        } else {
          uw2.push_back(kUNKChar);
        }
      }
      w.first = string_util::UnicodeTextToUTF8(uw2);
    }

    if (trainer_spec_.model_type() != TrainerSpec::WORD &&
        trainer_spec_.model_type() != TrainerSpec::CHAR) {
      RET_CHECK_LE(
          static_cast<int>(required_chars_.size() + meta_pieces_.size()),
          trainer_spec_.vocab_size())
          << "Vocabulary size is smaller than required_chars. "
          << trainer_spec_.vocab_size() << " vs "
          << required_chars_.size() + meta_pieces_.size() << ". "
          << "Increase vocab_size or decrease character_coverage with "
          << "--character_coverage option.";
    }
  }

  LOG(INFO) << "Done! preprocessed " << sentences_.size() << " sentences.";

  return absl::OkStatus();
}

void TrainerInterface::SplitSentencesByWhitespace() {
  LOG(INFO) << "Tokenizing input sentences with whitespace: "
            << sentences_.size();
  absl::flat_hash_map<std::string, int64_t> tokens;
  for (const auto& s : sentences_) {
    for (const auto& w :
         SplitIntoWords(s.first, trainer_spec_.treat_whitespace_as_suffix(),
                        trainer_spec_.allow_whitespace_only_pieces())) {
      tokens[w] += s.second;
    }
  }
  sentences_ = Sorted(tokens);
  LOG(INFO) << "Done! " << sentences_.size();
}

absl::Status TrainerInterface::Serialize(ModelProto* model_proto) const {
  RETURN_IF_ERROR(status());

  // Duplicated sentencepiece is not allowed.
  absl::flat_hash_set<std::string> dup;

  model_proto->Clear();

#define CHECK_PIECE(piece)                            \
  RET_CHECK(string_util::IsStructurallyValid(piece)); \
  RET_CHECK(!piece.empty());                          \
  RET_CHECK(dup.insert(piece).second) << piece << " is already defined";

  size_t fid = 0;
  for (int id = 0; id < trainer_spec_.vocab_size(); ++id) {
    const auto it = meta_pieces_.find(id);
    if (it != meta_pieces_.end()) {
      auto* sp = model_proto->add_pieces();
      sp->set_piece(it->second.first);
      sp->set_type(it->second.second);
      sp->set_score(0.0);
      RET_CHECK_EQ(model_proto->pieces_size() - 1, it->first);
      RET_CHECK_NE(ModelProto::SentencePiece::NORMAL, sp->type());
      CHECK_PIECE(sp->piece());
    } else if (fid < final_pieces_.size()) {
      const auto& w = final_pieces_[fid++];
      auto* sp = model_proto->add_pieces();
      sp->set_piece(w.first);
      sp->set_score(w.second);
      CHECK_PIECE(sp->piece());
    }
  }

  RET_CHECK_EQ(fid, final_pieces_.size());

  *model_proto->mutable_trainer_spec() = trainer_spec_;
  *model_proto->mutable_normalizer_spec() = normalizer_spec_;

  if (!denormalizer_spec_.normalization_rule_tsv().empty()) {
    *model_proto->mutable_denormalizer_spec() = denormalizer_spec_;
  }

  if (!trainer_spec_.hard_vocab_limit() ||
      trainer_spec_.model_type() == TrainerSpec::CHAR) {
    RET_CHECK_GE(trainer_spec_.vocab_size(), model_proto->pieces_size());
    RET_CHECK_GE(trainer_spec_.vocab_size(), static_cast<int32_t>(dup.size()));
    model_proto->mutable_trainer_spec()->set_vocab_size(
        model_proto->pieces_size());
  } else {
    RET_CHECK_EQ(trainer_spec_.vocab_size(), model_proto->pieces_size())
        << absl::StrFormat(
               "Vocabulary size too high (%d). Please set it to a value <= %d.",
               trainer_spec_.vocab_size(), model_proto->pieces_size());
    RET_CHECK_EQ(trainer_spec_.vocab_size(), static_cast<int32_t>(dup.size()));
  }

  return absl::OkStatus();
}

absl::Status TrainerInterface::SaveModel(absl::string_view filename) const {
  LOG(INFO) << "Saving model: " << filename;
  ModelProto model_proto;

  RETURN_IF_ERROR(Serialize(&model_proto));

  auto output = filesystem::NewWritableFile(filename.data(), true);
  RETURN_IF_ERROR(output->status());
  output->Write(model_proto.SerializeAsString());
  return absl::OkStatus();
}

absl::Status TrainerInterface::SaveVocab(absl::string_view filename) const {
  LOG(INFO) << "Saving vocabs: " << filename;
  ModelProto model_proto;
  RETURN_IF_ERROR(Serialize(&model_proto));
  auto output = filesystem::NewWritableFile(filename);
  RETURN_IF_ERROR(output->status());

  for (const auto& piece : model_proto.pieces()) {
    if (piece.piece().find_first_of(" \t\r\n") != std::string::npos) {
      LOG(WARNING) << "The piece [" << piece.piece()
                   << "] contains escaped characters that break the format of "
                   << filename;
    }
  }

  if (trainer_spec_.vocabulary_output_piece_score()) {
    for (const auto& piece : model_proto.pieces()) {
      RET_CHECK(
          output->WriteLine(absl::StrCat(piece.piece(), "\t", piece.score())));
    }
  } else {
    for (const auto& piece : model_proto.pieces()) {
      RET_CHECK(output->WriteLine(piece.piece()));
    }
  }

  return absl::OkStatus();
}

absl::Status TrainerInterface::Save() const {
  if (output_model_proto_ != nullptr) {
    RETURN_IF_ERROR(Serialize(output_model_proto_));
  } else {
    RETURN_IF_ERROR(SaveModel(trainer_spec_.model_prefix() + ".model"));
    RETURN_IF_ERROR(SaveVocab(trainer_spec_.model_prefix() + ".vocab"));
  }
  return absl::OkStatus();
}

absl::Status TrainerInterface::InitMetaPieces() {
  RET_CHECK(meta_pieces_.empty());
  bool has_unk = false;

  auto insert_id = [&has_unk, this](int id, const std::string& w) -> bool {
    if (id < 0) {
      return true;
    }
    if (id >= trainer_spec_.vocab_size() ||
        meta_pieces_.find(id) != meta_pieces_.end() ||
        (has_unk && w == trainer_spec_.unk_piece())) {
      return false;
    }
    if (w == trainer_spec_.unk_piece()) {
      has_unk = true;
    }
    meta_pieces_[id] = std::make_pair(
        w, w == trainer_spec_.unk_piece() ? ModelProto::SentencePiece::UNKNOWN
                                          : ModelProto::SentencePiece::CONTROL);
    return true;
  };

  RET_CHECK(insert_id(trainer_spec_.unk_id(), trainer_spec_.unk_piece()));
  RET_CHECK(insert_id(trainer_spec_.bos_id(), trainer_spec_.bos_piece()));
  RET_CHECK(insert_id(trainer_spec_.eos_id(), trainer_spec_.eos_piece()));
  RET_CHECK(insert_id(trainer_spec_.pad_id(), trainer_spec_.pad_piece()));

  RET_CHECK(has_unk) << trainer_spec_.unk_piece() << " must be defined.";

  absl::flat_hash_set<std::string> dup;

  int id = 0;
  auto insert_meta_symbol =
      [&id, &dup, this](const std::string& w,
                        ModelProto::SentencePiece::Type type) -> absl::Status {
    if (!dup.insert(w).second) {
      return absl::InternalError(absl::StrCat(
          w, " is already defined. duplicated symbols are not allowed."));
    }

    if (w == trainer_spec_.unk_piece()) {
      return absl::InternalError(
          absl::StrCat(trainer_spec_.unk_piece(),
                       " must not be defined with --control_symbols and "
                       "--user_defined_symbols."));
    }

    if (w == trainer_spec_.bos_piece() && trainer_spec_.bos_id() >= 0) {
      meta_pieces_[trainer_spec_.bos_id()].second = type;
    } else if (w == trainer_spec_.eos_piece() && trainer_spec_.eos_id() >= 0) {
      meta_pieces_[trainer_spec_.eos_id()].second = type;
    } else if (w == trainer_spec_.pad_piece() && trainer_spec_.pad_id() >= 0) {
      meta_pieces_[trainer_spec_.pad_id()].second = type;
    } else {
      while (meta_pieces_.find(id) != meta_pieces_.end()) {
        ++id;
      }
      meta_pieces_[id] = std::make_pair(w, type);
    }

    return absl::OkStatus();
  };

  for (const auto& w : trainer_spec_.control_symbols()) {
    RETURN_IF_ERROR(insert_meta_symbol(w, ModelProto::SentencePiece::CONTROL));
  }

  for (const auto& w : trainer_spec_.user_defined_symbols()) {
    RETURN_IF_ERROR(
        insert_meta_symbol(w, ModelProto::SentencePiece::USER_DEFINED));
    if (trainer_spec_.model_type() == TrainerSpec::WORD &&
        normalizer_spec_.escape_whitespaces() &&
        !absl::StartsWith(w, TrainerInterface::kWSStr)) {
      // WORD model tokens include the escaped whitespace prefix.
      RETURN_IF_ERROR(
          insert_meta_symbol(absl::StrCat(TrainerInterface::kWSStr, w),
                             ModelProto::SentencePiece::USER_DEFINED));
    }
  }

  if (trainer_spec_.byte_fallback()) {
    for (int i = 0; i < 256; ++i) {
      RETURN_IF_ERROR(
          insert_meta_symbol(ByteToPiece(i), ModelProto::SentencePiece::BYTE));
    }
  }

  return absl::OkStatus();
}

}  // namespace sentencepiece
