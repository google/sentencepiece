# SentencePiece TensorFlow module

## WARNING
tf_sentencepiece is going to be deprecated in tensorflow 2.3.0. tf_sentencepiece for tensorflow 2.2.0x is the last release of tf_sentencepiece. Use [tensoflow_text](https://github.com/tensorflow/text) to run sentencepiece on tensorflow.

Example
```Python
import tensorflow as tf
import tensorflow_text as text

model = open('test_model.model', 'rb').read()
s1 = text.SentencepieceTokenizer(model=model)
print(s1.tokenize(['hello world']))
print(s1.tokenize_with_offsets(['hello world']))

s2 = text.SentencepieceTokenizer(model=model, out_type=tf.dtypes.string)
print(s2.tokenize(['hello world']))
print(s2.tokenize_with_offsets(['hello world']))
```

## Introduction

SentencePiece TensorFlow module implements the encode (text to id/piece) and decode (id/piece to text) operations which are executed lazily on top of TensorFlow's Session mechanism. This module allows to make an end-to-end training/inference computatation graph by directly feeding raw sentences with the tf.placeholder. 
The SentencePiece model (model proto) is passed as an attribute of the TensorFlow operation
and embedded into the TensorFlow graph so the model and graph become purely self-contained.

## Build and Install SentencePiece
For Linux (x64), macOS environment:

```
% pip install tf_sentencepiece
```

## Usage
Use pydoc to see the usage instruction
```
% pydoc sentencepiece_processor_ops
```

[Sample code](https://colab.research.google.com/drive/1rQ0tgXmHv02sMO6VdTO0yYaTvc1Yv1yP)
