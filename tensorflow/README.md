# SentencePiece TensorFlow module

SentencePiece TensorFlow module implements the encode (text to id/piece) and decode (id/piece to text) functions
of SentencePiece in a native tensorflow operations which are executed lazily on top of TensorFlow's Session mechanism.
This module allows to make an end-to-end training/inference graph by feeding raw sentences to the graph with the
TensorFlow's placeholder. The SentencePiece model (model proto) is passed as an attribute of the TensorFlow operation
and embedded into the TensorFlow graph so the model and graph become purely self-contained.

## Build and Install SentencePiece
For Linux (x64), macOS environment:

```
% pip install tf_sentencepiece
```

## Usage

```
% pydoc sentencepiece_processor_ops
```

[Sample code](https://colab.research.google.com/drive/1rQ0tgXmHv02sMO6VdTO0yYaTvc1Yv1yP?authuser=2#scrollTo=i_44FzFC0qwe)
