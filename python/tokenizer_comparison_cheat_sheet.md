# Tokenizer Comparison Cheat Sheet

This cheat sheet compares the capabilities, API designs, and performance features of three major tokenizer libraries: **SentencePiece** (new pybind11 API), **Hugging Face Tokenizers**, and OpenAI's **tiktoken**.

---

## 1. Capability Matrix

| Feature / Capability | SentencePiece | Hugging Face `tokenizers` | tiktoken |
| :--- | :--- | :--- | :--- |
| **Compared Version** | `v0.2.2` (New pybind11 API) | `v1.1.1` | `v0.13.0` |
| **GIL Release (Parallelism)** |  ✅ Yes (NoGIL/Free-Threading) |  ✅ Yes |  ✅ Yes |
| **Encode Offset Mapping (Unicode)** |  ✅ Yes (via `return_type='offset_mapping'`) |  ✅ Yes (via `Encoding.offsets`) | ❌ No |
| **Decode Offset Mapping (Unicode)** |  ✅ Yes (via `return_type='offset_mapping'`, includes `text` key) | ❌ No | ❌ No |
| **Raw Byte Offset Mapping** |  ✅ Yes (triggered by bytes input or `return_bytes=True` for both encode and decode) | ❌ No | ❌ No |
| **Direct UTF-8 Bytes I/O** |  ✅ Yes (accepts bytes for encode, returns bytes for decode) | ❌ No (requires str) |  ⚠️ Partial (requires str for encode, returns bytes via `decode_bytes`) |
| **NumPy Array Output** |  ✅ Yes (Direct to NumPy, zero-copy) | ❌ No (Transformers wrapper only) | ❌ No |
| **Batch Encoding Parallelism** |  ✅ Yes (via `num_threads` or `ThreadPool`) |  ✅ Yes (automatic Rayon multi-threading) |  ✅ Yes (via `num_threads`) |
| **Within-Doc Parallelism** |  ✅ Yes (`parallel_encode`) | ❌ No (sequential per document) | ❌ No (sequential per document) |
| **Training from Iterator** |  ✅ Yes (`sentence_iterator`) |  ✅ Yes (`train_from_iterator`) | ❌ No |
| **In-Memory Add Tokens** | ⚠️ Hard (via `protobuf` rewrite) |  ✅ Yes (Easy, via `add_tokens`) | ❌ No |
| **Pre-Tokenization (Modular)** | ❌ No (Intentional design: parses raw Unicode stream without pre-splitting) |  ✅ Yes (fully modular: `tokenizer.pre_tokenizer = ...`) | ⚠️ Partial (static via regex `pat_str` in constructor) |
| **Modular Post-Processing** | ❌ No (BOS/EOS toggles only) |  ✅ Yes (template post-processor) | ❌ No |
| **Text Normalization Support** |  ✅ Yes (baked into model, highly optimized precompiled rules) |  ✅ Yes (fully modular pipeline, e.g. Lowercase, NFKC) | ❌ No (Intentional design: tokenizes raw input exactly as-is) |
| **Custom Normalizers** |  ✅ Yes (defined at training time via TSV mapping) |  ✅ Yes (custom Python/Rust functions or sequences) | ❌ No |
| **Special Token Security Policy** |  ✅ Yes (static per-token: `control_symbols` [safe] vs `user_defined_symbols` [unsafe] in same model) | ⚠️ Partial (global toggle `split_special_tokens`, cannot mix per-token) | ⚠️ Partial (global call-time toggle `allowed_special`/`disallowed_special`, cannot mix per-token) |
| **Token/Char Alignment Helpers** | ❌ No (returns raw offsets only) |  ✅ Yes (`char_to_token()`, `token_to_word()`, etc.) | ❌ No |

---

## 2. Usage & Code Comparison

Below is a side-by-side comparison of common tasks across the three libraries.

### 2.1. Instantiate

*   **SentencePiece**:
    ```python
    import sentencepiece as spm
    
    # Load from file
    sp = spm.SentencePieceProcessor.from_file("m.model")
    
    # Load from in-memory bytes
    sp = spm.SentencePieceProcessor.from_proto(proto_bytes)
    ```
*   **Hugging Face `tokenizers`**:
    ```python
    from tokenizers import Tokenizer
    
    tokenizer = Tokenizer.from_file("t.json")
    ```
*   **tiktoken**:
    ```python
    import tiktoken
    
    enc = tiktoken.get_encoding("cl100k_base")
    # or:
    enc = tiktoken.encoding_for_model("gpt-4")
    ```

### 2.2. Encode (IDs)

*   **SentencePiece**:
    ```python
    ids = sp.encode("Hello world", return_type=int)
    ```
*   **Hugging Face `tokenizers`**:
    ```python
    ids = tokenizer.encode("Hello world").ids
    ```
*   **tiktoken**:
    ```python
    ids = enc.encode("Hello world")
    ```

### 2.3. Encode (Pieces)

*   **SentencePiece**:
    ```python
    pieces = sp.encode("Hello world", return_type=str)
    # Returns list[str]: ['▁Hello', '▁world']
    
    # Or to get raw bytes pieces directly:
    pieces_bytes = sp.encode("Hello world", return_type=bytes)
    # Returns list[bytes]: [b'\xe2\x96\x81Hello', b'\xe2\x96\x81world']
    ```
*   **Hugging Face `tokenizers`**:
    ```python
    pieces = tokenizer.encode("Hello world").tokens
    # Returns list[str]
    ```
*   **tiktoken**:
    ```python
    # tiktoken does not support direct token-to-string piece representation
    # without decoding each ID individually to bytes:
    pieces = [enc.decode_single_token_bytes(t) for t in ids]
    # Returns list[bytes]
    ```

### 2.4. Decode (Str)

*   **SentencePiece**:
    ```python
    text = sp.decode(ids)
    ```
*   **Hugging Face `tokenizers`**:
    ```python
    text = tokenizer.decode(ids)
    ```
*   **tiktoken**:
    ```python
    text = enc.decode(ids)
    ```

### 2.5. Decode (Bytes)

*   **SentencePiece**:
    ```python
    text_bytes = sp.decode(ids, return_type=bytes)
    ```
*   **Hugging Face `tokenizers`**:
    *   *Not Supported* (must manually decode string to bytes)
*   **tiktoken**:
    ```python
    text_bytes = enc.decode_bytes(ids)
    ```

### 2.6. Offset Mapping (Unicode)

*   **SentencePiece**:
    ```python
    # Encode with Unicode character offsets:
    res = sp.encode("text", return_type='offset_mapping')
    
    # Or force Unicode offsets on bytes input explicitly:
    res = sp.encode(b"text", return_type='offset_mapping', return_bytes=False)
    # res['offsets'] -> list[tuple[int, int]]
    
    # Decode with Unicode character offsets:
    res = sp.decode(ids, return_type='offset_mapping')
    # res['text'] -> str, res['offsets'] -> list[tuple[int, int]]
    ```
*   **Hugging Face `tokenizers`**:
    ```python
    # Encode with Unicode character offsets:
    offsets = tokenizer.encode("text").offsets
    # offsets -> list[tuple[int, int]]
    
    # Decode with offsets:
    # *Not Supported*
    ```
*   **tiktoken**:
    *   *Not Supported*

### 2.7. Offset Mapping (Bytes)

*   **SentencePiece**:
    ```python
    # Encode with raw byte offsets (triggered by bytes input):
    res = sp.encode(b"text", return_type='offset_mapping')
    
    # Or force raw byte offsets on str input explicitly:
    res = sp.encode("text", return_type='offset_mapping', return_bytes=True)
    # res['offsets'] -> list[tuple[int, int]] (byte offsets)
    
    # Decode with raw byte offsets (explicit return_bytes=True):
    res = sp.decode(ids, return_type='offset_mapping', return_bytes=True)
    # res['text'] -> bytes, res['offsets'] -> list[tuple[int, int]] (byte offsets)
    ```
*   **Hugging Face `tokenizers`**:
    *   *Not Supported*
*   **tiktoken**:
    *   *Not Supported*

### 2.8. NumPy Support

*   **SentencePiece**:
    ```python
    # Direct encoding to NumPy array (zero-copy):
    arr = sp.encode("text", return_type='numpy')
    
    # Decoding directly from NumPy array:
    text = sp.decode(arr)
    ```
*   **Hugging Face `tokenizers`**:
    ```python
    import numpy as np
    # Requires manual conversion:
    arr = np.array(tokenizer.encode("text").ids)
    ```
*   **tiktoken**:
    *   *Not Supported* (requires manual conversion)

### 2.9. Encode Batch

*   **SentencePiece**:
    ```python
    # Parallel processing in C++ (releasing GIL):
    ids_list = sp.encode(texts, num_threads=4)
    ```
*   **Hugging Face `tokenizers`**:
    ```python
    # Parallel processing in Rust (Rayon):
    outputs = tokenizer.encode_batch(texts)
    ids_list = [o.ids for o in outputs]
    ```
*   **tiktoken**:
    ```python
    # Parallel processing in Rust:
    ids_list = enc.encode_batch(texts, num_threads=4)
    ```

### 2.10. Parallel (Within-Doc)

*   **SentencePiece**:
    ```python
    # Parallelize encoding of a single large document:
    pool = spm.ThreadPool(num_threads=4)
    ids = sp.parallel_encode(large_text, chunk_len=1048576, thread_pool=pool, return_type=int)
    ```
*   **Hugging Face `tokenizers`**:
    *   *Not Supported* (encodes single document sequentially on one thread)
*   **tiktoken**:
    *   *Not Supported* (encodes single document sequentially on one thread)

### 2.11. Thread Pool

*   **SentencePiece**:
    ```python
    # Explicit thread pool sharing across multiple batch calls:
    pool = spm.ThreadPool(num_threads=8)
    ids = sp.encode(batch, thread_pool=pool)
    ```
*   **Hugging Face `tokenizers`**:
    *   *Managed internally* in Rust via global Rayon thread pool.
*   **tiktoken**:
    *   *Managed internally* in Rust thread pool.

### 2.12. Train from Iterator

*   **SentencePiece**:
    ```python
    spm.SentencePieceTrainer.train(sentence_iterator=my_iter, model_prefix='m', vocab_size=1000)
    ```
*   **Hugging Face `tokenizers`**:
    ```python
    from tokenizers.trainers import BpeTrainer
    tokenizer.train_from_iterator(my_iter, trainer=BpeTrainer())
    ```
*   **tiktoken**:
    *   *Not Supported* (training is not exposed in public Python API)

### 2.13. Model Modification (In-Memory)

*   **SentencePiece**:
    ```python
    # Edit the underlying protobuf (unsupported/no guarantee):
    import sentencepiece_model_pb2 as pb
    proto = pb.ModelProto()
    proto.ParseFromString(open("m.model", "rb").read())
    # ... modify proto (e.g. proto.pieces.add()) ...
    sp = spm.SentencePieceProcessor.from_proto(proto.SerializeToString())
    ```
*   **Hugging Face `tokenizers`**:
    ```python
    # Dynamically add tokens to loaded model:
    tokenizer.add_tokens(["<|my_token|>"])
    tokenizer.add_special_tokens(["<|special|>"])
    ```
*   **tiktoken**:
    *   *Not Supported*

### 2.14. Configure Pre-tokenizer

*   **SentencePiece**:
    *   *Not Supported dynamically* (pre-tokenization is baked into model normalization at training time)
*   **Hugging Face `tokenizers`**:
    ```python
    from tokenizers import pre_tokenizers
    # Configure modular pre-tokenizer:
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    ```
*   **tiktoken**:
    *   *Partial* (static regex pattern passed during custom class initialization)

### 2.15. Special Token Security

*   **SentencePiece**:
    ```python
    # Defined statically at training time:
    spm.train(..., control_symbols=['<c>'], user_defined_symbols=['<u>'])
    # <c> is safe (never split from raw text), <u> is parsed from text.
    ```
*   **Hugging Face `tokenizers`**:
    ```python
    # Configure behavior at load/call time:
    tokenizer = AutoTokenizer.from_pretrained(..., split_special_tokens=True)
    ```
*   **tiktoken**:
    ```python
    # Enforced at call time:
    enc.encode("text <|endoftext|>", allowed_special=set(), disallowed_special="all")
    # Raises ValueError on unexpected special token injection
    ```

### 2.16. Alignment Helpers

*   **SentencePiece**:
    *   *Not Supported* (returns raw offsets, but no high-level helper APIs to map char to token)
*   **Hugging Face `tokenizers`**:
    ```python
    output = tokenizer.encode("text")
    token_id = output.char_to_token(5)
    span = output.token_to_chars(token_id)
    ```
*   **tiktoken**:
    *   *Not Supported*

### 2.17. Configure Normalizer

*   **SentencePiece**:
    *   *Training-time only* (normalization rule defined via `normalization_rule_name` or `normalization_rule_tsv` during training).
*   **Hugging Face `tokenizers`**:
    ```python
    from tokenizers import normalizers
    # Configure normalizer pipeline dynamically:
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    ```
*   **tiktoken**:
    *   *Not Supported* (tokenizes raw input exactly as-is)

### 2.18. Independent Normalization

*   **SentencePiece**:
    ```python
    # Normalize text independently of tokenization:
    norm = spm.SentencePieceNormalizer(model_file='m.model')
    print(norm.normalize("text"))
    # Or via processor:
    print(sp.normalize("text"))
    ```
*   **Hugging Face `tokenizers`**:
    ```python
    # Normalize text independently:
    print(tokenizer.normalizer.normalize_str("text"))
    ```
*   **tiktoken**:
    *   *Not Supported*

### 2.19. Normalization Offsets

*   **SentencePiece**:
    ```python
    # Retrieve character-to-byte offset mapping after normalization:
    normalized, offsets = sp.normalize("text", with_offsets=True)
    ```
*   **Hugging Face `tokenizers`**:
    *   *Not Supported directly* on normalizer (offsets are computed during full encoding only)
*   **tiktoken**:
    *   *Not Supported*
