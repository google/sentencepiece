// Evaluates exact Raw Log-Likelihood of test corpus given a SentencePiece
// Unigram model.
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "init.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_processor.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/flags/flag.h"
#include "third_party/absl/strings/str_format.h"

ABSL_FLAG(std::string, model, "", "model file name");
ABSL_FLAG(std::string, input, "", "input filename");

int main(int argc, char* argv[]) {
  sentencepiece::ScopedResourceDestructor cleaner;
  sentencepiece::ParseCommandLineFlags(argv[0], &argc, &argv, true);

  const std::string model_file = absl::GetFlag(FLAGS_model);
  const std::string input_file = absl::GetFlag(FLAGS_input);

  if (model_file.empty() || input_file.empty()) {
    std::cerr << "Usage: spm_eval --model=<model> --input=<input>\n";
    return 1;
  }

  sentencepiece::SentencePieceProcessor sp;
  if (!sp.Load(model_file).ok()) {
    std::cerr << "Failed to load model: " << model_file << "\n";
    return 1;
  }

  std::ifstream ifs(input_file);
  if (!ifs) {
    std::cerr << "Failed to open input file: " << input_file << "\n";
    return 1;
  }

  std::string line;
  double total_viterbi_log_likelihood = 0.0;
  uint64_t total_bytes = 0;
  uint64_t total_tokens = 0;
  uint64_t sentence_count = 0;
  absl::flat_hash_map<int, uint64_t> token_counts;

  std::vector<int> ids;
  while (std::getline(ifs, line)) {
    if (line.empty()) continue;

    // Add 1 byte for newline if non-empty to match exact raw text size
    total_bytes += line.size() + 1;
    sentence_count++;

    if (!sp.Encode(line, &ids).ok()) {
      std::cerr << "Failed to encode line: " << line << "\n";
      continue;
    }

    total_tokens += ids.size();
    for (int id : ids) {
      total_viterbi_log_likelihood += sp.GetScore(id);
      token_counts[id]++;
    }
  }

  // Calculate empirical unigram log likelihood: \sum c_i * log(c_i /
  // total_tokens)
  double empirical_log_likelihood = 0.0;
  if (total_tokens > 0) {
    const double N = static_cast<double>(total_tokens);
    for (const auto& kv : token_counts) {
      const double count = static_cast<double>(kv.second);
      empirical_log_likelihood += count * std::log(count / N);
    }
  }

  const bool is_unigram = (sp.model_proto().trainer_spec().model_type() ==
                           sentencepiece::TrainerSpec::UNIGRAM);

  auto safe_div = [](double num, double den) -> double {
    return den > 0.0 ? num / den : 0.0;
  };

  const double eval_ll_nats =
      is_unigram ? total_viterbi_log_likelihood : empirical_log_likelihood;

  const double nats_per_byte = safe_div(eval_ll_nats, total_bytes);
  const double nats_per_token = safe_div(eval_ll_nats, total_tokens);
  const double byte_w_ppl = std::exp(-nats_per_byte);
  const double standard_ppl = std::exp(-nats_per_token);

  const double empirical_nats_per_byte = safe_div(empirical_log_likelihood, total_bytes);
  const double empirical_byte_w_ppl = std::exp(-empirical_nats_per_byte);

  const double bytes_per_token = safe_div(total_bytes, total_tokens);
  const double tokens_per_byte = safe_div(total_tokens, total_bytes);
  const double compression_ratio_pct = tokens_per_byte * 100.0;

  const size_t model_vocab_size = sp.GetPieceSize();
  const size_t unique_vocab_used = token_counts.size();
  const double vocab_coverage_pct = safe_div(unique_vocab_used * 100.0, model_vocab_size);

  std::cout << "=======================================================\n";
  std::cout << "RAW LOG-LIKELIHOOD & COMPRESSION REPORT\n";
  std::cout << "=======================================================\n";
  std::cout << "Model File                       : " << model_file << "\n";
  std::cout << "Input File                       : " << input_file << "\n";
  std::cout << "Total Sentences                  : " << sentence_count << "\n";
  std::cout << "Total Bytes                      : " << total_bytes << "\n";
  std::cout << "Total Tokens                     : " << total_tokens << "\n";
  std::cout << "Model Vocab Size (K)             : " << model_vocab_size << "\n";
  std::cout << "Unique Vocab Used                : " << unique_vocab_used << "\n";
  std::cout << "Vocab Coverage (%)               : "
            << absl::StrFormat("%.2f%%", vocab_coverage_pct) << "\n";
  std::cout << "-------------------------------------------------------\n";
  std::cout << "COMPRESSION DENSITY & RATIO\n";
  std::cout << "Compression Density (Bytes/Tok)  : "
            << absl::StrFormat("%.6f", bytes_per_token) << "\n";
  std::cout << "Compression Rate    (Tok/Byte)  : "
            << absl::StrFormat("%.6f", tokens_per_byte) << "\n";
  std::cout << "Token/Byte Ratio                 : "
            << absl::StrFormat("%.2f%%", compression_ratio_pct) << "\n";
  std::cout << "-------------------------------------------------------\n";
  std::cout << "INFORMATION THEORY & PPL\n";
  if (is_unigram) {
    std::cout << "Viterbi Log-Likelihood (Nats)    : "
              << absl::StrFormat("%.6f", total_viterbi_log_likelihood) << "\n";
  }
  std::cout << "Empirical Log-Likelihood (Nats)  : "
            << absl::StrFormat("%.6f", empirical_log_likelihood) << "\n";
  std::cout << "Log-Likelihood / Byte (Nats)     : "
            << absl::StrFormat("%.6f", nats_per_byte) << "\n";
  std::cout << "Standard Perplexity (PPL)        : "
            << absl::StrFormat("%.6f", standard_ppl) << "\n";
  std::cout << "Byte-Weighted Perplexity (PPL)   : "
            << absl::StrFormat("%.6f", byte_w_ppl) << "\n";
  std::cout << "Empirical Byte-W PPL (All Models): "
            << absl::StrFormat("%.6f", empirical_byte_w_ppl) << "\n";
  std::cout << "=======================================================\n";

  return 0;
}
