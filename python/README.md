# SentencePiece Python Wrapper

Python wrapper for SentencePiece with SWIG. This module wraps sentencepiece::SentencePieceProcessor class with the following modifications:
* Encode and Decode methods are re-defined as EncodeAsIds, EncodeAsPieces, DecodeIds and DecodePieces respectevely.
* Support model training with SentencePieceTrainer.Train method.
* SentencePieceText proto is not supported.
* Added __len__ and __getitem__ methods. len(obj) and obj[key] returns vocab size and vocab id respectively.

## Build and Install SentencePiece
For Linux (x64/i686) environment, you can simply use pip comand to install SentencePiece python module.

```
% pip install sentencepiece
```

Note that binary wheel package is not avaialble for non-Linux environment, including macOS, Windows, and Linux (arm).
You need to install [SentencePiece C++](https://github.com/google/sentencepiece#c-from-source) library in advance.

To build and install the Python wrapper manually, please install [SentencePiece C++](https://github.com/google/sentencepiece#c-from-source) and try the following commands:
```
% python setup.py build
% sudo python setup.py install
```

If you don’t have write permission to the global site-packages directory or don’t want to install into it, please try:
```
% python setup.py install --user
```

## Usage

### Segmentation
```
% python
>>> import sentencepiece as spm
>>> sp = spm.SentencePieceProcessor()
>>> sp.Load("test/test_model.model")
True
>>> sp.EncodeAsPieces("This is a test")
['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'est']
>>> sp.EncodeAsIds("This is a test")
[284, 47, 11, 4, 15, 400]
>>> sp.DecodePieces(['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'est'])
'This is a test'
>>> sp.NBestEncodeAsPieces("This is a test", 5)
[['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'est'], ['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 'te', 'st'], ['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 'te', 's', 't'], ['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 'st'], ['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'es', 't']]
>>> for x in range(10):
...     sp.SampleEncodeAsPieces("This is a test", -1, 0.1)
...
['\xe2\x96\x81', 'T', 'h', 'i', 's', '\xe2\x96\x81', 'is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 's', 't']
['\xe2\x96\x81T', 'h', 'is', '\xe2\x96\x81is', '\xe2\x96\x81', 'a', '\xe2\x96\x81', 't', 'est']
['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81', 'a', '\xe2\x96\x81', 't', 'e', 'st']
['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 'st']
['\xe2\x96\x81This', '\xe2\x96\x81is', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 's', 't']
['\xe2\x96\x81T', 'h', 'is', '\xe2\x96\x81', 'i', 's', '\xe2\x96\x81a', '\xe2\x96\x81', 'te', 's', 't']
['\xe2\x96\x81This', '\xe2\x96\x81', 'is', '\xe2\x96\x81a', '\xe2\x96\x81', 'te', 's', 't']
['\xe2\x96\x81This', '\xe2\x96\x81', 'i', 's', '\xe2\x96\x81a', '\xe2\x96\x81', 't', 'e', 'st']
['\xe2\x96\x81This', '\xe2\x96\x81', 'is', '\xe2\x96\x81', 'a', '\xe2\x96\x81', 't', 'e', 'st']
['\xe2\x96\x81This', '\xe2\x96\x81', 'i', 's', '\xe2\x96\x81', 'a', '\xe2\x96\x81', 'te', 's', 't']
>>> sp.DecodeIds([284, 47, 11, 4, 15, 400])
'This is a test'
>>> sp.GetPieceSize()
1000
>>> sp.IdToPiece(2)
'</s>'
>>> sp.PieceToId('</s>')
2
>>> len(sp)
1000
>>> sp['</s>']
2
```

### Model Training
Training is peformed by passing parameters of [spm_train](https://github.com/google/sentencepiece#train-sentencepiece-model) to  SentencePieceTrainer.Train() function.

```
>>> import sentencepiece as spm
>>> spm.SentencePieceTrainer.Train('--input=test/botchan.txt --model_prefix=m --vocab_size=1000')
unigram_model_trainer.cc(494) LOG(INFO) Starts training with : 
input: "test/botchan.txt"
model_prefix: "m"
model_type: UNIGRAM
..snip..
unigram_model_trainer.cc(529) LOG(INFO) EM sub_iter=0 size=1239 obj=10.4055 num_tokens=36256 num_tokens/piece=29.2623
unigram_model_trainer.cc(529) LOG(INFO) EM sub_iter=1 size=1239 obj=10.3187 num_tokens=36256 num_tokens/piece=29.2623
unigram_model_trainer.cc(529) LOG(INFO) EM sub_iter=0 size=1100 obj=10.5285 num_tokens=37633 num_tokens/piece=34.2118
unigram_model_trainer.cc(529) LOG(INFO) EM sub_iter=1 size=1100 obj=10.4973 num_tokens=37630 num_tokens/piece=34.2091
trainer_interface.cc(284) LOG(INFO) Saving model: m.model
trainer_interface.cc(293) LOG(INFO) Saving vocabs: m.vocab
>>>
```

## Python2/3 String/Unicode compatibility
Sentencepiece python wrapper accepts both Unicode string and legacy byte string.
The output string type is determined by the input string type.
The output type of IdToPiece/DecodeIds methods is *str*, but note that it is a legacy byte string in Python2 and Unicode string in Python3 respectively.

* Python2:
```
>>> sp.EncodeAsPieces('吾輩は猫である')
['\xe2\x96\x81', '\xe5\x90\xbe', '\xe8\xbc\xa9', '\xe3\x81\xaf', '\xe7\x8c\xab', '\xe3\x81\xa7\xe3\x81\x82\xe3\x82\x8b']
>>> sp.EncodeAsPieces(u'吾輩は猫である')
[u'\u2581', u'\u543e', u'\u8f29', u'\u306f', u'\u732b', u'\u3067\u3042\u308b']
>>> sp.EncodeAsPieces(u'吾輩は猫である'.encode('utf-8'))
['\xe2\x96\x81', '\xe5\x90\xbe', '\xe8\xbc\xa9', '\xe3\x81\xaf', '\xe7\x8c\xab', '\xe3\x81\xa7\xe3\x81\x82\xe3\x82\x8b']
>>> sp.IdToPiece(10)
'\xe3\x81\xab'
>>> type(sp.IdToPiece(10))
<type 'str'>
```

* Python3:
```
>>> sp.EncodeAsPieces('吾輩は猫である')
['▁', '吾', '輩', 'は', '猫', 'である']
>>> sp.EncodeAsPieces('吾輩は猫である'.encode('utf-8'))
[b'\xe2\x96\x81', b'\xe5\x90\xbe', b'\xe8\xbc\xa9', b'\xe3\x81\xaf', b'\xe7\x8c\xab', b'\xe3\x81\xa7\xe3\x81\x82\xe3\x82\x8b']
>>>
>>> sp.IdToPiece(10)
'に'
>>> type(sp.IdToPiece(10))
<class 'str'>
```
