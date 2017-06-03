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

#include "builder.h"
#include "flags.h"
#include "trainer_factory.h"

using sentencepiece::TrainerSpec;
using sentencepiece::NormalizerSpec;
using sentencepiece::normalizer::Builder;

namespace {
static sentencepiece::TrainerSpec kDefaultTrainerSpec;
static sentencepiece::NormalizerSpec kDefaultNormalizerSpec;
}  // namespace

DEFINE_string(input, "", "comma separated list of input sentences");
DEFINE_string(model_prefix, "", "output model prefix");
DEFINE_string(model_type, "unigram",
              "model algorithm: unigram, bpe, word or char");
DEFINE_int32(vocab_size, kDefaultTrainerSpec.vocab_size(), "vocabulary size");
DEFINE_string(accept_language, "",
              "comma-separated list of languages this model can accept");
DEFINE_double(character_coverage, kDefaultTrainerSpec.character_coverage(),
              "character coverage to determine the minimum symbols");
DEFINE_int32(input_sentence_size, kDefaultTrainerSpec.input_sentence_size(),
             "maximum size of sentences the trainer loads");
DEFINE_int32(mining_sentence_size, kDefaultTrainerSpec.mining_sentence_size(),
             "maximum size of sentences to make seed sentence piece");
DEFINE_int32(training_sentence_size,
             kDefaultTrainerSpec.training_sentence_size(),
             "maximum size of sentences to train sentence pieces");
DEFINE_int32(seed_sentencepiece_size,
             kDefaultTrainerSpec.seed_sentencepiece_size(),
             "the size of seed sentencepieces");
DEFINE_double(shrinking_factor, kDefaultTrainerSpec.shrinking_factor(),
              "Keeps top shrinking_factor pieces with respect to the loss");
DEFINE_int32(num_threads, kDefaultTrainerSpec.num_threads(),
             "number of threads for training");
DEFINE_int32(num_sub_iterations, kDefaultTrainerSpec.num_sub_iterations(),
             "number of EM sub-iterations");
DEFINE_int32(max_sentencepiece_length,
             kDefaultTrainerSpec.max_sentencepiece_length(),
             "maximum length of sentence piece");
DEFINE_bool(split_by_unicode_script,
            kDefaultTrainerSpec.split_by_unicode_script(),
            "use Unicode script to split sentence pieces");
DEFINE_bool(split_by_whitespace, kDefaultTrainerSpec.split_by_whitespace(),
            "use a white space to split sentence pieces");
DEFINE_string(control_symbols, "", "comma separated list of control symbols");
DEFINE_string(user_defined_symbols, "",
              "comma separated list of user defined symbols");
DEFINE_string(normalization_rule_name, "nfkc",
              "Normalization rule name. "
              "Choose from nfkc or identity");
DEFINE_string(normalization_rule_tsv, "", "Normalization rule TSV file. ");
DEFINE_bool(add_dummy_prefix, kDefaultNormalizerSpec.add_dummy_prefix(),
            "Add dummy whitespace at the beginning of text");
DEFINE_bool(remove_extra_whitespaces,
            kDefaultNormalizerSpec.remove_extra_whitespaces(),
            "Removes leading, trailing, and "
            "duplicate internal whitespace");

namespace {
sentencepiece::NormalizerSpec MakeNormalizerSpec() {
  if (!FLAGS_normalization_rule_tsv.empty()) {
    const auto chars_map = sentencepiece::normalizer::Builder::BuildMapFromFile(
        FLAGS_normalization_rule_tsv);
    sentencepiece::NormalizerSpec spec;
    spec.set_name("user_defined");
    spec.set_precompiled_charsmap(
        sentencepiece::normalizer::Builder::CompileCharsMap(chars_map));
    return spec;
  }

  return sentencepiece::normalizer::Builder::GetNormalizerSpec(
      FLAGS_normalization_rule_name);
}
}  // namespace

int main(int argc, char *argv[]) {
  sentencepiece::flags::ParseCommandLineFlags(argc, argv);
  sentencepiece::TrainerSpec trainer_spec;
  sentencepiece::NormalizerSpec normalizer_spec;

  CHECK_OR_HELP(input);
  CHECK_OR_HELP(model_prefix);

// Populates the value from flags to spec.
#define SetTrainerSpecFromFlag(name) trainer_spec.set_##name(FLAGS_##name);

#define SetNormalizerSpecFromFlag(name) \
  normalizer_spec.set_##name(FLAGS_##name);

#define SetRepeatedTrainerSpecFromFlag(name)                     \
  if (!FLAGS_##name.empty()) {                                   \
    for (const auto v :                                          \
         sentencepiece::string_util::Split(FLAGS_##name, ",")) { \
      trainer_spec.add_##name(v);                                \
    }                                                            \
  }

  SetTrainerSpecFromFlag(model_prefix);
  SetTrainerSpecFromFlag(vocab_size);
  SetTrainerSpecFromFlag(character_coverage);
  SetTrainerSpecFromFlag(input_sentence_size);
  SetTrainerSpecFromFlag(mining_sentence_size);
  SetTrainerSpecFromFlag(training_sentence_size);
  SetTrainerSpecFromFlag(seed_sentencepiece_size);
  SetTrainerSpecFromFlag(shrinking_factor);
  SetTrainerSpecFromFlag(num_threads);
  SetTrainerSpecFromFlag(num_sub_iterations);
  SetTrainerSpecFromFlag(max_sentencepiece_length);
  SetTrainerSpecFromFlag(split_by_unicode_script);
  SetTrainerSpecFromFlag(split_by_whitespace);
  SetRepeatedTrainerSpecFromFlag(accept_language);
  SetRepeatedTrainerSpecFromFlag(control_symbols);
  SetRepeatedTrainerSpecFromFlag(user_defined_symbols);

  normalizer_spec = MakeNormalizerSpec();
  SetNormalizerSpecFromFlag(add_dummy_prefix);
  SetNormalizerSpecFromFlag(remove_extra_whitespaces);

  for (const auto &filename :
       sentencepiece::string_util::Split(FLAGS_input, ",")) {
    trainer_spec.add_input(filename);
  }

  const std::map<std::string, TrainerSpec::ModelType> kModelTypeMap = {
      {"unigram", TrainerSpec::UNIGRAM},
      {"bpe", TrainerSpec::BPE},
      {"word", TrainerSpec::WORD},
      {"char", TrainerSpec::CHAR}};
  trainer_spec.set_model_type(
      sentencepiece::port::FindOrDie(kModelTypeMap, FLAGS_model_type));

  auto trainer =
      sentencepiece::TrainerFactory::Create(trainer_spec, normalizer_spec);
  trainer->Train();

  return 0;
}
