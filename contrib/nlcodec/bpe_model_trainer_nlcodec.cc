// Copyright 2024 nlcodec / Thamme Gowda
// Fast BPE merge loop implementation.
// See bpe_model_trainer_nlcodec.h for algorithm description.

#include "contrib/nlcodec/bpe_model_trainer_nlcodec.h"

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "util.h"

// Flag defined here so it's available to all binaries (spm_train, tests, etc.)
ABSL_FLAG(bool, nlcodec_bpe, false,
          "Use nlcodec's fast BPE trainer (~8x faster). Only for "
          "--model_type=bpe.");

namespace sentencepiece {
namespace nlcodec {

auto RunFastBPEMerges(
    const TrainerInterface::Sentences &sentences, int vocab_size,
    std::vector<std::pair<std::string, float>> *final_pieces,
    std::function<bool(const string_util::UnicodeText &)> is_valid)
    -> util::Status {
  // ── Step 1: Build char→ID mapping from sentences ──
  std::unordered_map<std::string, int32_t> char_to_id;
  std::vector<std::string> id_to_str;

  auto get_or_add = [&](const std::string &ch) -> int32_t {
    auto it = char_to_id.find(ch);
    if (it != char_to_id.end()) return it->second;
    int32_t id = static_cast<int32_t>(id_to_str.size());
    char_to_id[ch] = id;
    id_to_str.push_back(ch);
    return id;
  };

  // Convert each word to a sequence of char IDs.
  // Words already have ▁ from SP's SplitSentencesByWhitespace.
  struct WordSeq {
    std::vector<int32_t> ids;
    int64_t freq;
  };
  std::vector<WordSeq> words;

  for (const auto &s : sentences) {
    if (s.first.empty()) continue;
    WordSeq w;
    w.freq = s.second;
    const auto &str = s.first;
    size_t i = 0;
    while (i < str.size()) {
      size_t len = 1;
      auto c = static_cast<unsigned char>(str[i]);
      if (c >= 0xF0)
        len = 4;
      else if (c >= 0xE0)
        len = 3;
      else if (c >= 0xC0)
        len = 2;
      if (i + len > str.size()) len = str.size() - i;
      w.ids.push_back(get_or_add(str.substr(i, len)));
      i += len;
    }
    words.push_back(std::move(w));
  }

  LOG(INFO) << "nlcodec_bpe: " << words.size() << " word types, "
            << id_to_str.size() << " initial chars";

  // ── Step 2: Build linked lists + bigram index ──
  NodeArena arena;
  std::unordered_map<int32_t, int64_t> uni;
  std::unordered_map<Bigram, int64_t, BigramHash> bi;
  BigramIndex bi_ixs;

  for (auto &w : words) {
    auto nodes = arena.from_seq(w.ids, w.freq);
    for (size_t i = 0; i + 1 < w.ids.size(); i++) {
      Bigram bg{w.ids[i], w.ids[i + 1]};
      bi[bg] += w.freq;
      bi_ixs[bg].insert(nodes[i]);
      uni[w.ids[i]] += w.freq;
    }
    if (!w.ids.empty()) uni[w.ids.back()] += w.freq;
  }

  // ── Step 3: Merge loop ──
  MaxHeap heap(bi);
  HeapDirty dirty;

  absl::flat_hash_set<std::string> dup;

  while (static_cast<int>(final_pieces->size()) < vocab_size) {
    if (heap.empty()) break;

    // Pop max, applying lazy corrections
    auto [pair, freq] = heap.pop();
    for (auto dit = dirty.find(pair); dit != dirty.end();
         dit = dirty.find(pair)) {
      int64_t d = dit->second;
      dirty.erase(dit);
      int64_t c = freq + d;
      if (c > 0) heap.push(pair, c);
      if (heap.empty()) {
        LOG(INFO) << "nlcodec_bpe: produced " << final_pieces->size()
                  << " merge pieces";
        return util::OkStatus();
      }
      std::tie(pair, freq) = heap.pop();
    }

    if (freq <= 0) break;

    auto [a, b] = pair;
    const std::string merged = id_to_str[a] + id_to_str[b];

    // Validate: SP requires valid sentence pieces
    const auto ut = string_util::UTF8ToUnicodeText(merged);
    if (!is_valid(ut)) continue;

    // Skip duplicates
    if (!dup.insert(merged).second) continue;

    // Register new merged token
    int32_t new_id = static_cast<int32_t>(id_to_str.size());
    id_to_str.push_back(merged);

    final_pieces->emplace_back(merged,
                               -static_cast<float>(final_pieces->size()));

    if (final_pieces->size() % 20 == 0) {
      LOG(INFO) << "Added: freq=" << freq
                << " size=" << final_pieces->size()
                << " piece=" << merged;
    }

    // Update counts
    uni[new_id] = freq;
    uni[a] -= freq;
    uni[b] -= freq;

    std::unordered_map<Bigram, int64_t, BigramHash> deltas;
    auto update_nodes = std::move(bi_ixs[pair]);
    bi_ixs.erase(pair);

    for (auto *nd : update_nodes) {
      auto *x = nd->left;
      auto *bn = nd->right;

      if (nd->is_unlinked() ||
          (a == b && bn &&
           (nd->val == new_id || bn->val == new_id))) {
        uni[a] += nd->freq;
        uni[b] += nd->freq;
        uni[new_id] -= nd->freq;
        continue;
      }

      if (!bn || nd->val != a || bn->val != b ||
          nd->freq != bn->freq) {
        uni[a] += nd->freq;
        uni[b] += nd->freq;
        uni[new_id] -= nd->freq;
        continue;
      }

      auto *y = bn->right;
      bn->unlink();
      nd->val = new_id;

      if (x) {
        deltas[{x->val, a}] -= x->freq;
        bi_ixs[{x->val, a}].erase(x);
        if (bi_ixs[{x->val, a}].empty()) bi_ixs.erase({x->val, a});
        deltas[{x->val, new_id}] += x->freq;
        bi_ixs[{x->val, new_id}].insert(x);
      }
      if (y) {
        deltas[{b, y->val}] -= bn->freq;
        bi_ixs[{b, y->val}].erase(bn);
        if (bi_ixs[{b, y->val}].empty()) bi_ixs.erase({b, y->val});
        deltas[{new_id, y->val}] += bn->freq;
        bi_ixs[{new_id, y->val}].insert(nd);
      }
    }

    for (auto &[p, d] : deltas) {
      if (d > 0)
        heap.push(p, d);
      else if (d < 0)
        dirty[p] += d;
    }
  }

  LOG(INFO) << "nlcodec_bpe: produced " << final_pieces->size()
            << " merge pieces";
  return util::OkStatus();
}

}  // namespace nlcodec
}  // namespace sentencepiece
