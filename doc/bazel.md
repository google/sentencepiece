# Building SentencePiece with Bazel

In addition to the [CMake build](cli.md), SentencePiece can be built with [Bazel](https://bazel.build) (using Bzlmod).

## Prerequisites

- [Bazel](https://bazel.build) 7.2.1 or later (Bazel 7, 8, 9 supported; installing via [Bazelisk](https://github.com/bazelbuild/bazelisk) is recommended)
- A C++17 compatible compiler (GCC, Clang, or MSVC)

## Building

Build all libraries, tools, and tests:

```bash
bazel build //...
```

### Main Targets

| Target | Description |
| --- | --- |
| `//:sentencepiece` (or `//src:sentencepiece_processor`) | Core runtime library (encode/decode/normalize) |
| `//:sentencepiece_train` (or `//src:sentencepiece_trainer`) | Model training library |
| `//src:libsentencepiece.so` | Standalone shared library (`.so` / `.dylib` / `.dll`) |
| `//src:libsentencepiece_train.so` | Standalone trainer shared library |
| `//:spm_train` | Model trainer CLI |
| `//:spm_encode` | Encoder CLI |
| `//:spm_decode` | Decoder CLI |
| `//:spm_normalize` | Text normalizer CLI |
| `//:spm_export_vocab` | Vocabulary exporter CLI |
| `//:spm_eval` | Evaluation CLI |
| `//:compile_charsmap` | Unicode normalization charsmap compiler CLI |

### Running CLIs Directly with Bazel

You can train and encode directly using `bazel run`:

```bash
# Train a model
bazel run //src:spm_train -- \
  --input=$(pwd)/data/botchan.txt \
  --model_prefix=$(pwd)/m \
  --vocab_size=1000

# Encode text
echo "Hello world." | bazel-bin/src/spm_encode --model=m.model
```

## Running Unit Tests

Run all unit tests in parallel with caching:

```bash
# Run all 22 test suites in parallel
bazel test //...

# Or run the consolidated test suite matching CMake's spm_test
bazel test //src:spm_test --test_output=errors
```

## Using SentencePiece in Another Bazel Project (Bzlmod)

Add SentencePiece to your `MODULE.bazel`:

```starlark
bazel_dep(name = "sentencepiece", version = "0.2.3")
```

Then in your `BUILD.bazel`:

```starlark
cc_binary(
    name = "my_app",
    srcs = ["main.cc"],
    deps = [
        "@sentencepiece//:sentencepiece",
        "@sentencepiece//:sentencepiece_train",  # if training support is needed
    ],
)
```
