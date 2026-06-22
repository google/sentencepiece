# SentencePiece Python Wrapper

Python wrapper for SentencePiece. This API supports the encoding, decoding, and training of SentencePiece models.

For a detailed feature and API comparison with Hugging Face Tokenizers and OpenAI's tiktoken, see the [Tokenizer Comparison Cheat Sheet](tokenizer_comparison_cheat_sheet.md).

## Installation

For Linux (x86_64/aarch64), macOS, and Windows (x64/arm64) environments, you can use the pip command to install the SentencePiece Python module.

```bash
pip install sentencepiece
```

## Basic Usage

The `SentencePieceProcessor` class provides the primary interface for text tokenization (encoding) and detokenization (decoding).

#### Core Methods
*   **`sp.encode(...)`**: Segments input text into token IDs, string pieces, or other formats (like NumPy arrays or Protobuf messages).
*   **`sp.decode(...)`**: Reconstructs the original text from token IDs or string pieces.

#### Input Types & Batch Processing
Both methods support polymorphic inputs and can execute in either single or batch mode:
*   **Single Input**: Pass a single Unicode `str` or raw `bytes` (for encoding), or a single list of IDs/pieces or a 1D NumPy array (for decoding).
*   **Batch Input**: Pass a list of `str` or `bytes` (for encoding), or a list of lists of IDs/pieces or a 2D NumPy array (for decoding). Batch operations automatically release the Python GIL and can run in parallel in C++ (see [Batch Encoding](#batch-encoding-across-document-parallelism)).

Here is a Python example showing basic usage:

```python
import sentencepiece as spm
import numpy as np

# Load the model
sp = spm.SentencePieceProcessor(model_file='test/test_model.model')

# Encode text to IDs (default return_type is int)
print(sp.encode('This is a test'))
# Output: [284, 47, 11, 4, 15, 400]

# Encode batch of texts to IDs
print(sp.encode(['This is a test', 'Hello world'], return_type=int))
# Output: [[284, 47, 11, 4, 15, 400], [151, 88, 21, 887]]

# Alternative method for encoding to IDs
print(sp.encode_as_ids(['This is a test', 'Hello world']))
# Output: [[284, 47, 11, 4, 15, 400], [151, 88, 21, 887]]

# Encode text to pieces
print(sp.encode('This is a test', return_type=str))
# Output: ['▁This', '▁is', '▁a', '▁', 't', 'est']

# Encode batch of texts to pieces
print(sp.encode(['This is a test', 'Hello world'], return_type=str))
# Output: [['▁This', '▁is', '▁a', '▁', 't', 'est'], ['▁He', 'll', 'o', '▁world']]

# Alternative method for encoding to pieces
print(sp.encode_as_pieces(['This is a test', 'Hello world']))
# Output: [['▁This', '▁is', '▁a', '▁', 't', 'est'], ['▁He', 'll', 'o', '▁world']]

# Encode text to pieces as bytes
print(sp.encode('This is a test', return_type=bytes))
# Output: [b'\xe2\x96\x81This', b'\xe2\x96\x81is', b'\xe2\x96\x81a', b'\xe2\x96\x81', b't', b'est']

# Encode to NumPy array (zero-copy)
print(sp.encode('This is a test', return_type='numpy'))
# Output: [284  47  11   4  15 400] (NumPy array)

# Encode to offset mapping
print(sp.encode('This is a test', return_type='offset_mapping'))
# Output: {
#   'ids': [284, 47, 11, 4, 15, 400],
#   'pieces': ['▁This', '▁is', '▁a', '▁', 't', 'est'],
#   'offsets': [(0, 4), (4, 7), (7, 9), (9, 10), (10, 11), (11, 14)]
# }

# Encode to Protobuf message
proto = sp.encode('This is a test', return_type='proto')
from google.protobuf import text_format
print(text_format.MessageToString(proto))
# Output:
# text: "This is a test"
# pieces {
#   piece: "▁This"
#   id: 284
#   surface: "This"
#   begin: 0
#   end: 4
# }
# ... (remaining pieces omitted for brevity)

# Alternative method for encoding to Protobuf
proto2 = sp.encode_as_proto('This is a test')
print(proto2 == proto)
# Output: True

# Encode with sampling (for subword regularization)
for _ in range(3):
  print(sp.encode('This is a test', return_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1))
# Output examples:
# ['▁', 'This', '▁', 'is', '▁a', '▁', 't', 'e', 'st']
# ['▁This', '▁is', '▁a', '▁', 'te', 's', 't']
# ['▁This', '▁is', '▁', 'a', '▁', 't', 'e', 'st']

# N-best encoding
print(sp.nbest_encode('This is a test', nbest_size=3, return_type=str))
# Output:
# [['▁This', '▁is', '▁a', '▁', 't', 'est'],
#  ['▁This', '▁is', '▁a', '▁', 'te', 'st'],
#  ['▁This', '▁is', '▁a', '▁', 'te', 's', 't']]

# Decode IDs to text
print(sp.decode([284, 47, 11, 4, 15, 400]))
# Output: This is a test

# Decode batch of IDs to text
print(sp.decode([[284, 47, 11, 4, 15, 400], [151, 88, 21, 887]]))
# Output: ['This is a test', 'Hello world']

# Decode NumPy array
ids_np = np.array([284, 47, 11, 4, 15, 400], dtype=np.int32)
print(sp.decode(ids_np))
# Output: This is a test

# Decode to bytes
print(sp.decode([284, 47, 11, 4, 15, 400], return_type=bytes))
# Output: b'This is a test'

# Decode with offset mapping
print(sp.decode([284, 47, 11, 4, 15, 400], return_type='offset_mapping'))
# Output: {
#   'text': 'This is a test',
#   'ids': [284, 47, 11, 4, 15, 400],
#   'pieces': ['▁This', '▁is', '▁a', '▁', 't', 'est'],
#   'offsets': [(0, 4), (4, 7), (7, 9), (9, 10), (10, 11), (11, 14)]
# }

# Decode to Protobuf
proto_dec = sp.decode([284, 47, 11, 4, 15, 400], return_type='proto')
print(proto_dec.text)
# Output: This is a test

# Decode pieces to text
print(sp.decode(['▁', 'This', '▁', 'is', '▁a', '▁', 't', 'e', 'st']))
# Output: This is a test

# Decode batch of pieces to text
print(sp.decode([['▁This', '▁is', '▁a', '▁', 't', 'est'], ['▁He', 'll', 'o', '▁world']]))
# Output: ['This is a test', 'Hello world']

# Get vocabulary size
print(sp.get_piece_size())
# Output: 1000

# Piece to ID mapping
print(sp.piece_to_id('<s>'))
# Output: 1
print(sp.piece_to_id(['</s>', '\r', '▁']))
# Output: [2, 3, 4]

# ID to piece mapping
print(sp.id_to_piece(2))
# Output: </s>
print(sp.id_to_piece([2, 3, 4]))
# Output: ['</s>', '\r', '▁']

# Dictionary-like access
print(len(sp))
# Output: 1000
print(sp['</s>'])
# Output: 2

# Special token IDs
print('bos=', sp.bos_id())  # Output: bos= 1
print('eos=', sp.eos_id())  # Output: eos= 2
print('unk=', sp.unk_id())  # Output: unk= 0
print('pad=', sp.pad_id())  # Output: pad= -1 (disabled)

# Get the first 5 vocabulary tokens
print([sp.id_to_piece(i) for i in range(5)])
# Output: ['<unk>', '<s>', '</s>', '\r', '▁']
```
### Offset Mapping

If you want to perform direct character-level or byte-level alignment operations (like text highlighting, substring extraction, or mapping tokens back to original text), you can use `return_type='offset_mapping'`. 

This yields a Python dictionary containing the token IDs, string pieces, and offsets (tuples of `(start, end)`).

#### Unicode Character Offsets (Default)

By default, using `return_type='offset_mapping'` returns **Unicode character offsets** for both encoding and decoding. Slicing Python strings directly using these offsets works out of the box:

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='test/test_model.model')

text = "吾輩は猫である。"

# 1. Encoding text to get offsets in the original text
enc_res = sp.encode(text, return_type='offset_mapping')
print(enc_res['offsets'])  # list[tuple[int, int]] containing Unicode char indices

for piece, (start, end) in zip(enc_res['pieces'], enc_res['offsets']):
  # Note: the first piece might be the dummy prefix (e.g. '▁') which maps to empty surface ''
  original_surface = text[start:end]
  print(f"Piece: {piece} -> Surface in original text: {original_surface}")

# 2. Decoding IDs back to text while retrieving offsets in the reconstructed text
dec_res = sp.decode(enc_res['ids'], return_type='offset_mapping')

print(dec_res['text'])    # "吾輩は猫である。" (the reconstructed text)
print(dec_res['offsets']) # Unicode char indices in `dec_res['text']`

for (start, end), piece in zip(dec_res['offsets'], dec_res['pieces']):
  # Note: piece might contain meta symbols (like '▁'), 
  # but slicing `dec_res['text']` using offsets yields the clean surface text.
  surface = dec_res['text'][start:end]
  print(f"Piece: {piece} -> Surface in reconstructed text: {surface}")
```

#### Raw Byte Offsets

You can retrieve byte-level offsets and raw byte pieces (useful for binary protocols, C++ compatibility, or raw byte slicing). Using `bytes` can also improve performance by bypassing Python's Unicode overhead (see [Optimizing Performance](#optimizing-performance-str-vs-bytes-encode--decode)).

How it is triggered:
*   **Encoding**: Defaults to matching the input type (i.e., `bytes` input triggers byte offsets, `str` input triggers Unicode offsets). You can explicitly override this by passing the **`return_bytes`** parameter in `encode()`:
    *   `return_bytes=True`: Forces raw byte offsets and `bytes` pieces (even with `str` input).
    *   `return_bytes=False`: Forces Unicode offsets and `str` pieces (even with `bytes` input).
    *Note: The `return_bytes` parameter in `encode()` is only allowed when `return_type='offset_mapping'`.*
*   **Decoding**: Used when you explicitly set **`return_bytes=True`** in `decode()` (only allowed when `return_type='offset_mapping'`).

```python
# 1. Encoding bytes input -> returns raw byte offsets and byte pieces
text_bytes = text.encode('utf-8')
enc_res_bytes = sp.encode(text_bytes, return_type='offset_mapping')

print(enc_res_bytes['offsets']) # Byte indices in `text_bytes`
print(enc_res_bytes['pieces'])  # list[bytes] containing raw byte pieces

# Slicing the raw bytes directly works:
for piece, (start, end) in zip(enc_res_bytes['pieces'], enc_res_bytes['offsets']):
  # Note: the first piece is the dummy prefix (b'\xe2\x96\x81') which maps to empty surface b''
  surface_bytes = text_bytes[start:end]
  print(f"Piece: {piece} -> Surface bytes: {surface_bytes}")

# 2. Decoding to raw byte offsets
dec_res_bytes = sp.decode(enc_res_bytes['ids'], return_type='offset_mapping', return_bytes=True)
print(dec_res_bytes['text'])    # b'\xe5\x90\xbe...' (reconstructed text as bytes)
print(dec_res_bytes['offsets']) # Byte indices in `dec_res_bytes['text']`

for (start, end), piece in zip(dec_res_bytes['offsets'], dec_res_bytes['pieces']):
  # piece is returned as bytes when return_bytes=True
  surface_bytes = dec_res_bytes['text'][start:end]
  print(f"Piece: {piece} -> Surface bytes: {surface_bytes}")
```

#### Byte Fallback Behavior

When `byte_fallback=True` is enabled in the model, unknown characters (e.g., emojis or unsupported scripts) are decomposed into a sequence of raw UTF-8 byte tokens (such as `<0xF0>`, `<0x9F>`, etc.). 

In this case, the offsets (whether Unicode character offsets or raw byte offsets) are assigned as follows:
*   **Intermediate Byte Tokens**: The first \(N-1\) byte tokens in the fallback sequence are assigned a zero-width span pointing to the start of the character (i.e., `(start, start)`). Slicing the text with these offsets yields an empty string (`""` or `b""`).
*   **Final Byte Token**: The last byte token in the sequence is assigned the span of the entire character (i.e., `(start, start + 1)` in Unicode character offsets, or `(start, start + byte_length)` in raw byte offsets). Slicing the text with these offsets yields the full reconstructed character.

This ensures that the text can still be cleanly sliced and reconstructed without producing invalid Unicode byte sequences during slicing.

### Protobuf Output

When encoding text with `return_type='proto'` (or `EncodeAsProto`), SentencePiece returns a structured `SentencePieceText` protobuf message. 

#### Benefits of Protobuf Output:
*   **Rich Metadata**: Provides highly detailed information for each token, including alignment offsets, scores (log-likelihoods), and whether tokens represent control characters or raw bytes.
*   **Language-Agnostic Contract**: Because it is a standard serialized protobuf message, it can be passed directly to downstream C++, Go, or Java servers, ensuring consistent deserialization across language boundaries.
*   **Format Flexibility**: Protobuf messages can be easily converted to other popular structures (such as JSON strings) using standard protobuf utilities without manual parsing.

Each token inside the protobuf message contains `begin` and `end` alignment offsets representing **raw byte offsets** in the UTF-8 encoded representation of the original string.

To slice the original text using these offsets, you must first encode the string to UTF-8 bytes, perform the slice, and then decode it back:

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='test/test_model.model')

text = "吾輩は猫である。"
proto = sp.encode(text, return_type='proto')

# Slice the UTF-8 bytes and decode back to string
text_bytes = text.encode('utf-8')
for piece in proto.pieces:
  surface = text_bytes[piece.begin:piece.end].decode('utf-8')
  print(f"Piece: {piece.piece} -> Surface: {surface}")
  assert surface == piece.surface

# Convert the protobuf message to JSON
from google.protobuf.json_format import MessageToJson
json_str = MessageToJson(proto)
print(json_str)
```

### Batch Encoding (Across-Document Parallelism)

When processing a **large collection of documents or sentences** (e.g., training datasets), you can pass a list of strings directly to `encode` (or a list of lists/numpy arrays to `decode`). 

To speed up batch tokenization, you can specify `num_threads=N` to parallelize processing across multiple CPU threads. For optimal performance across multiple batch calls, you should create and reuse a `ThreadPool`.

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='test/test_model.model')

# A large batch of sentences
sentences = ["This is a test sentence."] * 10000

# 1. Basic batch encoding (sequential)
ids_batch = sp.encode(sentences, return_type=int)

# 2. Parallelized batch encoding using multiple threads
# This processes elements of the list in parallel in C++ (releasing the GIL)
ids_batch = sp.encode(sentences, return_type=int, num_threads=4)

# 3. Best practice: Reuse a ThreadPool to avoid thread recreation overhead
pool = spm.ThreadPool(num_threads=4)
for i in range(10):
  # Reuse the same thread pool across multiple calls
  ids_batch = sp.encode(sentences, return_type=int, thread_pool=pool)
```

### Parallel Encoding (Within-Document Parallelism)

When tokenizing a **single, very large document** (e.g., a whole book or a massive text dump), you can use `parallel_encode` to split the document and tokenize it in parallel using multiple CPU threads. 

This is different from passing a list of sentences to `encode(..., num_threads=N)`, which parallelizes *across* the sentences in the list. `parallel_encode` parallelizes *within* a single string.

To use it efficiently, you should create and reuse a `ThreadPool`, and specify a `chunk_len` (in Unicode characters) to define the minimum chunk size processed by each thread.

> [!NOTE]
> Choosing an optimal `chunk_len` is key to performance. A value between **10,000 and 100,000 Unicode characters** is a common starting point, but the best value depends on:
> *   **Input characteristics**: Text size and complexity (e.g., density of multi-byte characters).
> *   **Model**: Complexity of the tokenization (e.g. Unigram vs BPE, vocabulary size).
> *   **Machine specifications**: Available CPU cores, cache sizes, and memory bandwidth.
> 
> Setting `chunk_len` too small can lead to excessive thread-scheduling overhead.

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='test/test_model.model')

# 1. Create a ThreadPool to reuse threads (prevents thread recreation overhead)
pool = spm.ThreadPool(num_threads=4)

# Load a very large document
with open('large_document.txt', 'r') as f:
  large_text = f.read()

# 2. Tokenize the large text in parallel
# chunk_len=1048576 (1M characters) is recommended to minimize thread scheduling overhead
ids = sp.parallel_encode(large_text, chunk_len=1048576, thread_pool=pool, return_type=int)
```

### Text Normalization

You can normalize text using the normalization rules defined inside the model, or by using a standalone `SentencePieceNormalizer` with custom configurations.

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='test/test_model.model')

# Standard normalization (e.g. NFKC and space replacement)
print(sp.normalize("Hello  World."))  # Double space
# Output: ▁Hello▁World.

# Normalization with character-to-byte offset mapping
normalized, offsets = sp.normalize("Hello  World.", with_offsets=True)
print(normalized)
# Output: ▁Hello▁World.
print(offsets)
# Output: [0, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]
# normalized[i] spans from original byte offsets[i] to offsets[i+1]
```

You can also use `SentencePieceNormalizer` to run normalization independently of a model, or with custom parameters:

```python
# Replicate processor's exact normalization behavior:
normalizer = spm.SentencePieceNormalizer(
    model_file='test/test_model.model',
    add_dummy_prefix=True,
    escape_whitespaces=True,
    remove_extra_whitespaces=True
)
print(normalizer.normalize("Hello  World."))
# Output: ▁Hello▁World.

# Initialize from mapping list of tuples (source, target)
norm_map = [
    ('foo', 'bar'),
    ('apple', 'orange'),
]
normalizer = spm.SentencePieceNormalizer(norm_map=norm_map)
print(normalizer.normalize("foo apple"))
# Output: bar orange

# Decompile back to list of tuples
print(normalizer.decompile())
# Output: [('apple', 'orange'), ('foo', 'bar')] (sorted by Unicode code point of source)
```


### Model Training

Training is performed by passing parameters to the `SentencePieceTrainer.train()` function. See the [training options documentation](../doc/options.md) for a full list of supported parameters (which match the command-line flags of [spm_train](https://github.com/google/sentencepiece#train-sentencepiece-model)).

```python
import sentencepiece as spm

# Train the model
spm.SentencePieceTrainer.train(
    input='test/botchan.txt',
    model_prefix='m',
    vocab_size=1000,
    user_defined_symbols=['foo', 'bar']
)
```

Expected output in console (stderr):
```text
sentencepiece_trainer.cc(73) LOG(INFO) Starts training with :
trainer_spec {
  input: test/botchan.txt
  .. snip
unigram_model_trainer.cc(500) LOG(INFO) EM sub_iter=1 size=1188 obj=10.2839 num_tokens=32182 num_tokens/piece=27.0892
unigram_model_trainer.cc(500) LOG(INFO) EM sub_iter=0 size=1100 obj=10.4269 num_tokens=33001 num_tokens/piece=30.0009
unigram_model_trainer.cc(500) LOG(INFO) EM sub_iter=1 size=1100 obj=10.4069 num_tokens=33002 num_tokens/piece=30.0018
trainer_interface.cc(595) LOG(INFO) Saving model: m.model
trainer_interface.cc(619) LOG(INFO) Saving vocabs: m.vocab
```

#### Training with a Custom Normalizer

You can pass a pre-configured `SentencePieceNormalizer` instance to the trainer. This ensures the exact same normalization rules are packaged with the trained model.

```python
# 1. Create and configure a normalizer
norm_map = [('foo', 'bar')]
normalizer = spm.SentencePieceNormalizer(norm_map=norm_map, add_dummy_prefix=False, escape_whitespaces=True)

# 2. Train using the normalizer instance
spm.SentencePieceTrainer.train(
    input='test/botchan.txt',
    model_prefix='m',
    vocab_size=1000,
    normalizer=normalizer
)
```

You can also extend pre-defined normalization rules (like 'nfkc') by decompiling them, appending your custom rules, and loading them back:

```python
# 1. Load pre-defined NFKC rules
base_normalizer = spm.SentencePieceNormalizer(rule_name='nfkc')

# 2. Decompile to get the base mapping list of tuples
norm_map = base_normalizer.decompile()

# 3. Add custom rules to the mapping list
norm_map.append(('foo', 'bar'))
norm_map.append(('apple', 'orange'))

# 4. Create a new normalizer with the extended mapping
# Note: we set escape_whitespaces=True as required by the trainer
extended_normalizer = spm.SentencePieceNormalizer(norm_map=norm_map, add_dummy_prefix=False, escape_whitespaces=True)

# 5. Train using the extended normalizer
spm.SentencePieceTrainer.train(
    input='test/botchan.txt',
    model_prefix='m',
    vocab_size=1000,
    normalizer=extended_normalizer
)
```

> [!IMPORTANT]
> If you specify both `normalizer` and other normalizer-specific arguments (like `add_dummy_prefix`) in `train()`, a `ValueError` will be raised. You must configure these settings directly on the `SentencePieceNormalizer` instance instead of passing them as trainer arguments.

### Training without a Local Filesystem

The SentencePiece trainer can accept any iterable object to feed training sentences. You can also pass a file-like object (any object with a `write()` method) to write the output model to arbitrary devices/buffers. These features are useful for running SentencePiece in environments that have limited access to the local file system (e.g., Google Colab).

```python
import urllib.request
import io
import sentencepiece as spm

# Loads model from URL as iterator and stores the model to BytesIO.
model = io.BytesIO()
with urllib.request.urlopen(
    'https://raw.githubusercontent.com/google/sentencepiece/master/data/botchan.txt'
) as response:
  spm.SentencePieceTrainer.train(
      sentence_iterator=response, model_writer=model, vocab_size=1000)

# Serialize the model as file.
# with open('out.model', 'wb') as f:
#   f.write(model.getvalue())

# Directly load the model from serialized model.
sp = spm.SentencePieceProcessor(model_proto=model.getvalue())
print(sp.encode('this is test'))
```

### Optimizing Performance: `str` vs `bytes` (Encode & Decode)

For maximum throughput (e.g., when processing large datasets or files), you can use Python **`bytes`** for both encoding and decoding to bypass Python's Unicode conversion overhead.

#### Why it is faster:
*   **Zero-Copy Encoding**: When you pass `bytes` to the encoder (e.g., read directly via `open(..., 'rb')`), the wrapper extracts the memory pointer zero-copy to construct `std::string_view` directly. If you pass `str`, Python must cache/convert it to UTF-8, and when reading from files, Python first decodes the file bytes to `str` only for the wrapper to re-encode them.
*   **Fast Decoding (No UTF-8 Validation)**: By default, the decoder converts the reconstructed C++ UTF-8 string into a Python `str`, which requires validating and decoding bytes to Unicode characters in Python. Specifying `return_type=bytes` directly returns a Python `bytes` object, bypassing this validation.

Using `bytes` instead of `str` yields **~1.14x speedup on encoding** and **~1.21x speedup on decoding** (benchmarked on a 30MB Japanese text dataset with 300k lines). Additionally, reading files directly as raw bytes (`open(..., 'rb')`) bypasses Python's native UTF-8 file decoder entirely, yielding further substantial I/O savings.

#### Example Pipeline:
```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="test.model")

# 1. Optimal Encoding: Read and pass raw bytes
with open("large_file.txt", "rb") as f:
  lines_bytes = f.read().splitlines()
ids = sp.encode(lines_bytes, return_type=int)

# 2. Optimal Decoding: Reconstruct directly to bytes
decoded_lines_bytes = sp.decode(ids, return_type=bytes)
with open("decoded_output.txt", "wb") as f:
  f.write(b"\n".join(decoded_lines_bytes) + b"\n")
```

> [!NOTE]
> Type consistency is maintained when using bytes:
> *   `sp.encode(bytes_input, return_type=bytes)` yields a list of Python **`bytes`** objects (raw byte pieces). Specifying `return_type=str` will force-decode the pieces to Python **`str`** objects.
> *   `sp.decode(ids, return_type=bytes)` yields a Python **`bytes`** object (or list of `bytes` in batch mode).


### Model Modification

Modifying an existing SentencePiece model (e.g., adding/removing tokens, editing token scores) in memory is **not officially supported**. The `SentencePieceProcessor` class is designed strictly as a read-only inference engine.

However, since the `.model` file is a serialized Protocol Buffer, you can load and manipulate it in Python using the standard `sentencepiece_model_pb2` definitions included with the package:

> [!WARNING]
> Programmatically modifying model structures is not officially supported and is done **at your own risk**. Modifying vocabulary size, indices, or scores directly can easily break model integrity or corrupt downstream model embeddings.
> Note that the **vocabulary ID of a token is implicitly defined as its index** inside the `pieces` repeated field in the `ModelProto` message (i.e., appending a piece assigns it the next sequential ID, and inserting or deleting pieces will shift the IDs of all subsequent tokens).

```python
from sentencepiece import sentencepiece_model_pb2 as model_pb2
import sentencepiece as spm

# 1. Read the binary model file as a Protobuf message
model_proto = model_pb2.ModelProto()
with open("old.model", "rb") as f:
  model_proto.ParseFromString(f.read())

# 2. Modify vocabulary tokens or scores (at your own risk)
new_piece = model_proto.pieces.add()
new_piece.piece = "<my_special_token>"

# Score meaning depends on the model type:
# - Unigram: Log-probability of the piece (usually negative float). Used to optimize segmentation.
# - BPE: Merge priority. Typically set to 0.0 for high priority, or negative values for lower priority.
new_piece.score = 0.0

# Piece types: NORMAL (1), UNKNOWN (2), CONTROL (3), USER_DEFINED (4), UNUSED (5), BYTE (6)
new_piece.type = model_pb2.ModelProto.SentencePiece.Type.CONTROL

# 3. Load the modified model directly from memory
sp = spm.SentencePieceProcessor(model_proto=model_proto.SerializeToString())

# 4. (Optional) Export/Inspect the modified ModelProto as JSON
from google.protobuf.json_format import MessageToJson
json_model = MessageToJson(model_proto)

# 5. Save the modified model back to a file
with open("new.model", "wb") as f:
  f.write(model_proto.SerializeToString())

# 6. Load the new model from the file
sp2 = spm.SentencePieceProcessor(model_file="new.model")
```


## Free-Threading Support
Experimental support for no-GIL/Free-Threading has been introduced in v0.2.1. For more details, please refer to [this page](https://py-free-threading.github.io/).
This operates similarly to how [NumPy](https://numpy.org/devdocs/reference/thread_safety.html#free-threaded-python) handles it.

The C++ library's `const` and `static` methods (such as `encode()`, `decode()`, and `train()`) are thread-safe and designed to work in a free-threaded (NoGIL) environment.
However, non-const methods like `load()` are not thread-safe and can cause data races; ensure you implement appropriate locks when calling them from multiple threads.

While this limitation might be removed in the future, please note that it's not a simple fix, as it would require additional shared locks in C++.

## Building from Source

Before building SentencePiece from source on Linux, ensure that the following dependencies are installed.

```bash
sudo apt update
sudo apt install -y cmake pkg-config build-essential
```

To build and install the Python wrapper from source, use the following commands to build and install the wheel package.

```bash
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake .. -DSPM_ENABLE_SHARED=OFF -DCMAKE_INSTALL_PREFIX=./root -DSPM_DISABLE_EMBEDDED_DATA=ON
make install
cd ../python
pip install build
python -m build --wheel
pip install dist/sentencepiece*.whl
```

If you don't have write permission to the global site-packages directory or don't want to install into it, you can use the `--user` flag:

```bash
pip install --user dist/sentencepiece*.whl
```

To build and install the Python wrapper from source on Windows using Visual Studio, first ensure you have PowerShell 7 (`pwsh.exe`) installed (which can be installed via `winget install --id Microsoft.Powershell --source winget`). Then, open a "Developer PowerShell for VS 2022" window and run the following commands:

```powershell
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake .. -DSPM_ENABLE_SHARED=OFF -DCMAKE_INSTALL_PREFIX=".\root" -DSPM_DISABLE_EMBEDDED_DATA=ON
cmake --build . --config Release --target install
cd ../python
pip install build
python -m build --wheel
Get-ChildItem .\dist\sentencepiece*.whl | ForEach-Object { pip install $_.FullName }
```

## Migration Notes (New pybind11 API)

Since v0.2.2, the SentencePiece Python wrapper has been migrated from SWIG to **pybind11**, introducing optimized NumPy support and robust Free-Threading (NoGIL) execution. 

This migration introduces a few changes to standard usage and output formats:

### 1. BOS/EOS Behavior for USER_DEFINED Symbols

Historically, `SentencePieceProcessor::bos_id()` and `eos_id()` returned `-1` if the corresponding tokens were defined as `USER_DEFINED` rather than `CONTROL`. However, the C++ `Encode` method with extra options (e.g. `bos`) still prepended the token, while the Python wrapper `encode(..., add_bos=True)` would prepend `-1` or crash.

To resolve this inconsistency:
*   **Behavior Alignment**: The behavior has been aligned to strictly respect the `CONTROL` token type requirement.
*   **Strict Verification & Errors**: If the BOS/EOS tokens are defined as `USER_DEFINED` (which are essentially treated the same as `NORMAL` tokens, with no clear distinction in this context) or any type other than `CONTROL` (or if they are completely missing), trying to add them during encoding will now result in an error:
    *   In C++, `SetEncodeExtraOptions("bos")` or `SetEncodeExtraOptions("eos")` will **fail** (return an error).
    *   In Python, passing `add_bos=True` or `add_eos=True` to `encode()` will **raise a `ValueError`** (instead of silently ignoring it, prepending `-1`, or crashing).

### 2. Unified `return_type` Parameter
The previous parameter `out_type` has been renamed to **`return_type`** to better describe what the functions yield.
*   **Backward Compatibility**: The name `out_type` remains supported as a deprecated keyword argument alias. Passing `out_type=int` will work exactly as before, mapping internally to `return_type=int`. Specifying both `return_type` and `out_type` at the same time is invalid and will raise a `ValueError`.

### 3. Removal of `immutable_proto`
The custom C++ wrapper classes (`ImmutableSentencePieceText`, `ImmutableNBestSentencePieceText`) have been **removed**. 
*   Using `return_type='immutable_proto'` (or `out_type='immutable_proto'`) or helper methods like `EncodeAsImmutableProto` will now raise a `ValueError`.
*   Use `return_type='proto'` or `EncodeAsProto` instead.

### 4. New Classmethod Factories
To align with pythonic best practices and avoid overloaded constructor argument logic, `SentencePieceProcessor` now exposes clean classmethod factories to initialize processors:
*   `SentencePieceProcessor.from_file(model_file, **kwargs)`: Creates and loads a processor from a file path.
*   `SentencePieceProcessor.from_proto(model_proto, **kwargs)`: Creates and loads a processor directly from serialized model bytes in memory.

```python
# 1. Loading from file
sp = spm.SentencePieceProcessor.from_file("m.model", return_type=str)

# 2. Loading from in-memory bytes
sp = spm.SentencePieceProcessor.from_proto(model_bytes, return_type=str)
```

## Further Documentation

For more detailed information on SentencePiece concepts, options, and advanced configurations, please refer to the core documentation:

*   **[Tokenizer Comparison Cheat Sheet](tokenizer_comparison_cheat_sheet.md)**: Comparison of SentencePiece (pybind11), Hugging Face Tokenizers, and tiktoken.
*   **[Training Options](../doc/options.md)**: Detailed guide to all training flags and parameters.
*   **[Special Symbols](../doc/special_symbols.md)**: Guide to special tokens (BOS, EOS, user-defined, control symbols) and how they behave.
*   **[Normalization](../doc/normalization.md)**: Information on text normalization rules and customization.
*   **[Piece Constraints](../doc/piece_constraints.md)**: Details on vocabulary constraints and subword segmentation.
*   **[C++ API Reference](../doc/cpp.md)**: Core C++ API documentation.
