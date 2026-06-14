# SentencePiece Python Module Extra Usage Examples

This document describes additional usage examples for the SentencePiece Python module, focusing on advanced API features.

## Loading model from byte stream

SentencePiece can load models directly from a serialized byte stream in memory instead of requiring a local file path. This is useful when loading models from remote databases or cloud storage (e.g., GCS, S3) without writing them to disk.

```python
import sentencepiece as spm

# Load serialized model bytes
with open('m.model', 'rb') as f:
    serialized_model_proto = f.read()

sp = spm.SentencePieceProcessor(model_proto=serialized_model_proto)

print(sp.encode('this is a test', out_type=str))
```

## Prepending/Appending Special Tokens

You can retrieve the IDs of default special tokens (`<unk>`, `<s>`, `</s>`, `<pad>`) using API methods, allowing you to manually insert them (e.g., prepending BOS or appending EOS).

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='m.model')

# Get special token IDs
print('bos=', sp.bos_id())
print('eos=', sp.eos_id())
print('unk=', sp.unk_id())
print('pad=', sp.pad_id())  # returns -1 if disabled/undefined

# Prepend BOS and append EOS IDs programmatically
encoded_ids = sp.encode('Hello world')
final_ids = [sp.bos_id()] + encoded_ids + [sp.eos_id()]
print(final_ids)
```
