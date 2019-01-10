namespace {
inline std::string PrintProto(const TrainerSpec &message) {
  std::ostringstream os;

  os << "TrainerSpec {\n";
  for (const auto &v : message.input())
    os << "  input: " << v << "\n";
  os << "  input_format: " << message.input_format() << "\n";
  os << "  model_prefix: " << message.model_prefix() << "\n";
  static const std::map<TrainerSpec::ModelType, std::string> kModelType_Map = { {TrainerSpec::UNIGRAM, "UNIGRAM"}, {TrainerSpec::BPE, "BPE"}, {TrainerSpec::WORD, "WORD"}, {TrainerSpec::CHAR, "CHAR"},  };
  {
    const auto it = kModelType_Map.find(message.model_type());
    if (it == kModelType_Map.end())
      os << "  model_type: unknown\n";
    else
      os << "  model_type: " << it->second << "\n";
  }
  os << "  vocab_size: " << message.vocab_size() << "\n";
  for (const auto &v : message.accept_language())
    os << "  accept_language: " << v << "\n";
  os << "  self_test_sample_size: " << message.self_test_sample_size() << "\n";
  os << "  character_coverage: " << message.character_coverage() << "\n";
  os << "  input_sentence_size: " << message.input_sentence_size() << "\n";
  os << "  shuffle_input_sentence: " << message.shuffle_input_sentence() << "\n";
  os << "  seed_sentencepiece_size: " << message.seed_sentencepiece_size() << "\n";
  os << "  shrinking_factor: " << message.shrinking_factor() << "\n";
  os << "  max_sentence_length: " << message.max_sentence_length() << "\n";
  os << "  num_threads: " << message.num_threads() << "\n";
  os << "  num_sub_iterations: " << message.num_sub_iterations() << "\n";
  os << "  max_sentencepiece_length: " << message.max_sentencepiece_length() << "\n";
  os << "  split_by_unicode_script: " << message.split_by_unicode_script() << "\n";
  os << "  split_by_number: " << message.split_by_number() << "\n";
  os << "  split_by_whitespace: " << message.split_by_whitespace() << "\n";
  os << "  treat_whitespace_as_suffix: " << message.treat_whitespace_as_suffix() << "\n";
  for (const auto &v : message.control_symbols())
    os << "  control_symbols: " << v << "\n";
  for (const auto &v : message.user_defined_symbols())
    os << "  user_defined_symbols: " << v << "\n";
  os << "  hard_vocab_limit: " << message.hard_vocab_limit() << "\n";
  os << "  use_all_vocab: " << message.use_all_vocab() << "\n";
  os << "  unk_id: " << message.unk_id() << "\n";
  os << "  bos_id: " << message.bos_id() << "\n";
  os << "  eos_id: " << message.eos_id() << "\n";
  os << "  pad_id: " << message.pad_id() << "\n";
  os << "  unk_piece: " << message.unk_piece() << "\n";
  os << "  bos_piece: " << message.bos_piece() << "\n";
  os << "  eos_piece: " << message.eos_piece() << "\n";
  os << "  pad_piece: " << message.pad_piece() << "\n";
  os << "  unk_surface: " << message.unk_surface() << "\n";
  os << "}\n";

  return os.str();
}

inline std::string PrintProto(const NormalizerSpec &message) {
  std::ostringstream os;

  os << "NormalizerSpec {\n";
  os << "  name: " << message.name() << "\n";
  os << "  add_dummy_prefix: " << message.add_dummy_prefix() << "\n";
  os << "  remove_extra_whitespaces: " << message.remove_extra_whitespaces() << "\n";
  os << "  escape_whitespaces: " << message.escape_whitespaces() << "\n";
  os << "  normalization_rule_tsv: " << message.normalization_rule_tsv() << "\n";
  os << "}\n";

  return os.str();
}

}  // namespace

util::Status SentencePieceTrainer::SetProtoField(const std::string& name, const std::string& value, TrainerSpec *message) {
  CHECK_OR_RETURN(message);

  if (name == "input") {
    for (const auto &val : string_util::Split(value, ",")) {
      message->add_input(val);
    }
    return util::OkStatus();
  }

  if (name == "input_format") {
    const auto &val = value;
    message->set_input_format(val);
    return util::OkStatus();
  }

  if (name == "model_prefix") {
    const auto &val = value;
    message->set_model_prefix(val);
    return util::OkStatus();
  }

  static const std::map <std::string, TrainerSpec::ModelType> kModelType_Map = { {"UNIGRAM", TrainerSpec::UNIGRAM}, {"BPE", TrainerSpec::BPE}, {"WORD", TrainerSpec::WORD}, {"CHAR", TrainerSpec::CHAR},  };

  if (name == "model_type") {
    const auto &val = value;
    const auto it = kModelType_Map.find(string_util::ToUpper(val));
    if (it == kModelType_Map.end())
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "unknown enumeration value of \"" << val << "\" as ModelType.";
    message->set_model_type(it->second);
    return util::OkStatus();
  }

  if (name == "vocab_size") {
    const auto &val = value;
    int32 v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as int32.";
    message->set_vocab_size(v);
    return util::OkStatus();
  }

  if (name == "accept_language") {
    for (const auto &val : string_util::Split(value, ",")) {
      message->add_accept_language(val);
    }
    return util::OkStatus();
  }

  if (name == "self_test_sample_size") {
    const auto &val = value;
    int32 v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as int32.";
    message->set_self_test_sample_size(v);
    return util::OkStatus();
  }

  if (name == "character_coverage") {
    const auto &val = value;
    float v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as float.";
    message->set_character_coverage(v);
    return util::OkStatus();
  }

  if (name == "input_sentence_size") {
    const auto &val = value;
    int32 v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as int32.";
    message->set_input_sentence_size(v);
    return util::OkStatus();
  }

  if (name == "shuffle_input_sentence") {
    const auto &val = value;
    bool v;
    if (!string_util::lexical_cast(val.empty() ? "true" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as bool.";
    message->set_shuffle_input_sentence(v);
    return util::OkStatus();
  }

  if (name == "seed_sentencepiece_size") {
    const auto &val = value;
    int32 v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as int32.";
    message->set_seed_sentencepiece_size(v);
    return util::OkStatus();
  }

  if (name == "shrinking_factor") {
    const auto &val = value;
    float v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as float.";
    message->set_shrinking_factor(v);
    return util::OkStatus();
  }

  if (name == "max_sentence_length") {
    const auto &val = value;
    int32 v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as int32.";
    message->set_max_sentence_length(v);
    return util::OkStatus();
  }

  if (name == "num_threads") {
    const auto &val = value;
    int32 v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as int32.";
    message->set_num_threads(v);
    return util::OkStatus();
  }

  if (name == "num_sub_iterations") {
    const auto &val = value;
    int32 v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as int32.";
    message->set_num_sub_iterations(v);
    return util::OkStatus();
  }

  if (name == "max_sentencepiece_length") {
    const auto &val = value;
    int32 v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as int32.";
    message->set_max_sentencepiece_length(v);
    return util::OkStatus();
  }

  if (name == "split_by_unicode_script") {
    const auto &val = value;
    bool v;
    if (!string_util::lexical_cast(val.empty() ? "true" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as bool.";
    message->set_split_by_unicode_script(v);
    return util::OkStatus();
  }

  if (name == "split_by_number") {
    const auto &val = value;
    bool v;
    if (!string_util::lexical_cast(val.empty() ? "true" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as bool.";
    message->set_split_by_number(v);
    return util::OkStatus();
  }

  if (name == "split_by_whitespace") {
    const auto &val = value;
    bool v;
    if (!string_util::lexical_cast(val.empty() ? "true" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as bool.";
    message->set_split_by_whitespace(v);
    return util::OkStatus();
  }

  if (name == "treat_whitespace_as_suffix") {
    const auto &val = value;
    bool v;
    if (!string_util::lexical_cast(val.empty() ? "true" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as bool.";
    message->set_treat_whitespace_as_suffix(v);
    return util::OkStatus();
  }

  if (name == "control_symbols") {
    for (const auto &val : string_util::Split(value, ",")) {
      message->add_control_symbols(val);
    }
    return util::OkStatus();
  }

  if (name == "user_defined_symbols") {
    for (const auto &val : string_util::Split(value, ",")) {
      message->add_user_defined_symbols(val);
    }
    return util::OkStatus();
  }

  if (name == "hard_vocab_limit") {
    const auto &val = value;
    bool v;
    if (!string_util::lexical_cast(val.empty() ? "true" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as bool.";
    message->set_hard_vocab_limit(v);
    return util::OkStatus();
  }

  if (name == "use_all_vocab") {
    const auto &val = value;
    bool v;
    if (!string_util::lexical_cast(val.empty() ? "true" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as bool.";
    message->set_use_all_vocab(v);
    return util::OkStatus();
  }

  if (name == "unk_id") {
    const auto &val = value;
    int32 v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as int32.";
    message->set_unk_id(v);
    return util::OkStatus();
  }

  if (name == "bos_id") {
    const auto &val = value;
    int32 v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as int32.";
    message->set_bos_id(v);
    return util::OkStatus();
  }

  if (name == "eos_id") {
    const auto &val = value;
    int32 v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as int32.";
    message->set_eos_id(v);
    return util::OkStatus();
  }

  if (name == "pad_id") {
    const auto &val = value;
    int32 v;
    if (!string_util::lexical_cast(val.empty() ? "" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as int32.";
    message->set_pad_id(v);
    return util::OkStatus();
  }

  if (name == "unk_piece") {
    const auto &val = value;
    message->set_unk_piece(val);
    return util::OkStatus();
  }

  if (name == "bos_piece") {
    const auto &val = value;
    message->set_bos_piece(val);
    return util::OkStatus();
  }

  if (name == "eos_piece") {
    const auto &val = value;
    message->set_eos_piece(val);
    return util::OkStatus();
  }

  if (name == "pad_piece") {
    const auto &val = value;
    message->set_pad_piece(val);
    return util::OkStatus();
  }

  if (name == "unk_surface") {
    const auto &val = value;
    message->set_unk_surface(val);
    return util::OkStatus();
  }

  return util::StatusBuilder(util::error::NOT_FOUND)
    << "unknown field name \"" << name << "\" in TrainerSpec.";
}

util::Status SentencePieceTrainer::SetProtoField(const std::string& name, const std::string& value, NormalizerSpec *message) {
  CHECK_OR_RETURN(message);

  if (name == "name") {
    const auto &val = value;
    message->set_name(val);
    return util::OkStatus();
  }

  if (name == "precompiled_charsmap") {
    const auto &val = value;
    message->set_precompiled_charsmap(val.data(), val.size());
    return util::OkStatus();
  }

  if (name == "add_dummy_prefix") {
    const auto &val = value;
    bool v;
    if (!string_util::lexical_cast(val.empty() ? "true" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as bool.";
    message->set_add_dummy_prefix(v);
    return util::OkStatus();
  }

  if (name == "remove_extra_whitespaces") {
    const auto &val = value;
    bool v;
    if (!string_util::lexical_cast(val.empty() ? "true" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as bool.";
    message->set_remove_extra_whitespaces(v);
    return util::OkStatus();
  }

  if (name == "escape_whitespaces") {
    const auto &val = value;
    bool v;
    if (!string_util::lexical_cast(val.empty() ? "true" : val, &v))
      return util::StatusBuilder(util::error::INVALID_ARGUMENT) << "cannot parse \"" << val << "\" as bool.";
    message->set_escape_whitespaces(v);
    return util::OkStatus();
  }

  if (name == "normalization_rule_tsv") {
    const auto &val = value;
    message->set_normalization_rule_tsv(val);
    return util::OkStatus();
  }

  return util::StatusBuilder(util::error::NOT_FOUND)
    << "unknown field name \"" << name << "\" in NormalizerSpec.";
}

