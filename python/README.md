# SentencePiece Python Wrapper

Python wrapper for SentencePiece with SWIG. This module wraps sentencepiece::SentencePieceProcessor class with the following modifications:
* Encode and Decode methods are re-defined as EncodeAsIds, EncodeAsPieces, DecodeIds and DecodePieces respectevely.
* SentencePieceText proto is not supported.
* Added __len__ and __getitem__ methods. len(obj) and obj[key] returns vocab size and vocab id respectively.

## Build and Install SentencePiece
You need to install SentencePiece before installing this python wrapper.

You can simply use pip comand to install SentencePiece python module.

```
% pip install sentencepiece
```

To install the wrapper manually, try the following commands:
```
% python setup.py build
% sudo python setup.py install
```

If you don’t have write permission to the global site-packages directory or don’t want to install into it, please try:
```
% python setup.py install --user
```

## Usage

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

## Python2/3 String/Unicode compatibility
Sentencepiece python wrapper accepts both Unicode string and legacy byte string.
The output string type is determined by the input string type.
The output type of IdToPiece/DecodeIds methods is *str*, but note that it is a legacy byte string in Python2 and Unicode string in Python3 respectively.

* Python2:
```
>>> sp.Encode('吾輩は猫である')
['\xe2\x96\x81', '\xe5\x90\xbe', '\xe8\xbc\xa9', '\xe3\x81\xaf', '\xe7\x8c\xab', '\xe3\x81\xa7\xe3\x81\x82\xe3\x82\x8b']
>>> sp.Encode(u'吾輩は猫である')
[u'\u2581', u'\u543e', u'\u8f29', u'\u306f', u'\u732b', u'\u3067\u3042\u308b']
>>> sp.Encode(u'吾輩は猫である'.encode('utf-8'))
['\xe2\x96\x81', '\xe5\x90\xbe', '\xe8\xbc\xa9', '\xe3\x81\xaf', '\xe7\x8c\xab', '\xe3\x81\xa7\xe3\x81\x82\xe3\x82\x8b']
>>> sp.IdToPiece(10)
'\xe3\x81\xab'
>>> type(sp.IdToPiece(10))
<type 'str'>
```

* Python3:
```
>>> sp.Encode('吾輩は猫である')
['▁', '吾', '輩', 'は', '猫', 'である']
>>> sp.Encode('吾輩は猫である'.encode('utf-8'))
[b'\xe2\x96\x81', b'\xe5\x90\xbe', b'\xe8\xbc\xa9', b'\xe3\x81\xaf', b'\xe7\x8c\xab', b'\xe3\x81\xa7\xe3\x81\x82\xe3\x82\x8b']
>>>
>>> sp.IdToPiece(10)
'に'
>>> type(sp.IdToPiece(10))
<class 'str'>
```
