# NLCodec: Fast BPE Training for SentencePiece

A drop-in replacement for SentencePiece's BPE merge loop that achieves **~10× speedup** by using a max-heap with lazy deletion instead of periodic linear scans.

This code lives under `contrib/nlcodec/` and is an **opt-in** feature: it is only compiled when the CMake option `SPM_NLCODEC_BPE` is set to `ON`. When enabled, the `--nlcodec_bpe` flag becomes available in `spm_train`; everything else (sentence loading, normalization, whitespace handling, model serialization) uses SentencePiece's native code paths.

## Building

```bash
mkdir -p build && cd build
cmake .. -DSPM_NLCODEC_BPE=ON
cmake --build . -j$(nproc)
```

Without `-DSPM_NLCODEC_BPE=ON`, none of the files in this directory are compiled and the `--nlcodec_bpe` flag is not registered.

## Algorithm

SentencePiece's default BPE trainer scans an "active set" of bigrams linearly every iteration, with a full rescan every 100 steps — O(N) per merge.

NLCodec uses three data structures to achieve O(log N) per merge:

1. **Max-heap** — finding the best pair is O(log N) pop, not O(N) scan.
2. **Doubly-linked lists** — merging adjacent tokens is O(1), no array shifting.
3. **Lazy deletion ("dirty map")** — frequency decrements are accumulated in a hash map and applied on pop, avoiding O(N) heap search.

Reference: Gowda et al., [*"Many-to-English Machine Translation Tools, Data, and Pretrained Models"*](https://aclanthology.org/2021.acl-demo.37), ACL 2021.

## Usage

```bash
# Train with the fast BPE algorithm
spm_train --input=data.txt --model_prefix=model \
          --vocab_size=32000 --model_type=bpe \
          --nlcodec_bpe
```

The output `.model` and `.vocab` files are identical in format to the default path.

## Files

| File | Description |
|------|-------------|
| `bpe_model_trainer_nlcodec.h` | Data structures (LnNode, NodeArena, MaxHeap, BigramIndex, HeapDirty) and `RunFastBPEMerges()` declaration |
| `bpe_model_trainer_nlcodec.cc` | Implementation of the fast merge loop + `--nlcodec_bpe` flag definition |
| `bpe_model_trainer_nlcodec_test.cc` | 4 test cases: valid model, vocab size match, encode/decode roundtrip, vocab overlap |
| `benchmark.sh` | Self-contained benchmark script (auto-downloads CC-100 data, builds, runs) |

## Benchmark

Run the benchmark (auto-downloads multilingual CC-100 data and builds SentencePiece):

```bash
bash contrib/nlcodec/benchmark.sh              # 200k lines, 32k vocab (default)
bash contrib/nlcodec/benchmark.sh -n 1000000   # 1M lines
bash contrib/nlcodec/benchmark.sh -s            # skip encoding comparison (faster)
bash contrib/nlcodec/benchmark.sh -h            # show all options
```

### Results: 200k multilingual sentences (en, de, zh, ar, hi), 32k vocab --> 10x speedup

```bash
$ bash contrib/nlcodec/benchmark.sh
==============================================
  Default:  149.2s
  Nlcodec:  14.4s
  Speedup:  10.3x
==============================================

Vocab overlap: 31,675 / 32,000 (99.0%)
Token counts:  8,346,614 (default) vs 8,347,032 (nlcodec)
```

The two paths produce nearly identical vocabularies (99% overlap) and equivalent compression. The small differences come from tie-breaking in pair frequency ordering.

### Results: 1M multilingual sentences (en, de, zh, ar, hi), 64k vocab --> 24x speedup

```bash
$ bash contrib/nlcodec/benchmark.sh -n 1000000 -v 64000

==============================================
  BPE Training Benchmark
  Input: train_1000000.txt (1000000 lines)
  Vocab: 64000
==============================================

--- Default BPE ---
trainer_interface.cc(411) LOG(INFO) Loaded all 950754 sentences
trainer_interface.cc(594) LOG(INFO) Done! preprocessed 950754 sentences.
trainer_interface.cc(611) LOG(INFO) Done! 1647091
Time: 1604650ms (1604.7s)

--- Nlcodec BPE (--nlcodec_bpe) ---
trainer_interface.cc(411) LOG(INFO) Loaded all 950754 sentences
trainer_interface.cc(594) LOG(INFO) Done! preprocessed 950754 sentences.
trainer_interface.cc(611) LOG(INFO) Done! 1647091
bpe_model_trainer_nlcodec.cc(67) LOG(INFO) nlcodec_bpe: 1647091 word types, 3288 initial chars
bpe_model_trainer_nlcodec.cc(195) LOG(INFO) nlcodec_bpe: produced 60710 merge pieces
Time: 66393ms (66.4s)

--- Vocab Comparison ---
Default vocab: 64000
Nlcodec vocab: 64000
Overlap: 63698 / 64000 (99.5%)

--- Encoding Comparison ---
Default total tokens: 38044696
Nlcodec total tokens: 38045754
Mean sent len (default): 1056797.11
Mean sent len (nlcodec): 1056826.50

==============================================
  Default:  1604.7s
  Nlcodec:  66.4s
  Speedup:  24.2x
==============================================
```

The two paths produce nearly identical vocabularies (99.5% overlap) and equivalent compression. The small differences come from tie-breaking in pair frequency ordering.

## Tests

Four test cases in `bpe_model_trainer_nlcodec_test.cc` verify correctness:

| Test | What it checks |
|------|---------------|
| `NlcodecBPETest.ProducesValidModel` | Trains a 3k-vocab model, verifies encode/decode works |
| `NlcodecBPETest.VocabSizeMatchesDefault` | Both paths produce identical vocab sizes |
| `NlcodecBPETest.EncodesDecodesCorrectly` | Encode→decode roundtrip on multiple strings |
| `NlcodecBPETest.VocabOverlapsWithDefault` | ≥50% vocab overlap between paths (typically 85%+) |

To build and run:

```bash
mkdir -p build && cd build
cmake .. -DSPM_BUILD_TEST=ON -DSPM_NLCODEC_BPE=ON
cmake --build . -j$(nproc) --target spm_test
./src/spm_test    # runs all tests including nlcodec
```

## Code Style

Code under `contrib/nlcodec/` follows C++ standard library conventions (`snake_case` for methods and variables, trailing return types) rather than the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) used by SentencePiece's `src/` directory. The glue code in `src/bpe_model_trainer.cc` (the `TrainFast()` method and flag declarations) follows Google style to match its surroundings.
