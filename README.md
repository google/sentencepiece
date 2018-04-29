# SentencePiece

[![Build Status](https://travis-ci.org/google/sentencepiece.svg?branch=master)](https://travis-ci.org/google/sentencepiece) [![Coverage Status](https://coveralls.io/repos/github/google/sentencepiece/badge.svg?branch=master)](https://coveralls.io/github/google/sentencepiece?branch=master)

SentencePiece is an unsupervised text tokenizer and detokenizer mainly for
Neural Network-based text generation systems where the vocabulary size
is predetermined prior to the neural model training. SentencePiece implements
**sub-word units** (also known as **wordpieces** [[Wu et al.](https://arxiv.org/pdf/1609.08144.pdf)]
[[Schuster et al.](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)]
and **byte-pair-encoding (BPE)** [[Sennrich et al.](http://www.aclweb.org/anthology/P16-1162)]) with the extension of direct
training from raw sentences. SentencePiece allows us to make a purely end-to-end
system that does not depend on language-specific pre/postprocessing.

**This is not an official Google product.**

## Technical highlights
- **Purely data driven**: SentencePiece trains tokenization and detokenization
  models from only raw sentences. No pre-tokenization ([Moses tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl)/[MeCab](http://taku910.github.io/mecab/)/[KyTea](http://www.phontron.com/kytea/)) is required.
- **Language independent**: SentencePiece treats the sentences just as sequences of Unicode characters. There is no language-dependent logic.
- **Fast and lightweight**: Segmentation speed is around 50k sentences/sec, and memory footprint is around 6MB.
- **Self-contained**: The same tokenization/detokenization is obtained as long as the same model file is used.
- **Direct vocabulary id generation**: SentencePiece manages vocabulary to id mapping and can directly generate vocabulary id sequences from raw sentences.
- **NFKC-based normalization**: SentencePiece performs NFKC-based text normalization.

## Overview
### What is SentencePiece?
SentencePiece is an unsupervised text tokenizer and detokenizer designed mainly for Neural Network-based text generation, for example Neural Network Machine Translation. SentencePiece is a re-implementation of **sub-word units** (also known as **wordpieces** [[Wu et al.](https://arxiv.org/pdf/1609.08144.pdf)][[Schuster et al.](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)] and **byte-pair-encoding (BPE)** [[Sennrich et al.](http://www.aclweb.org/anthology/P16-1162)]). Unlike previous sub-word approaches that train tokenizers from pretokenized sentences, SentencePiece directly trains the tokenizer and detokenizer from raw sentences.
SentencePiece might seem like a sort of unsupervised word segmentation, but there are several differences and constraints in SentencePiece.

#### The number of unique tokens is predetermined
Neural Machine Translation models typically operate with a fixed
vocabulary. Unlike most unsupervised word segmentation algorithms, which
assume an infinite vocabulary, SentencePiece trains the segmentation model such
that the final vocabulary size is fixed, e.g., 8k, 16k, or 32k.

#### Whitespace is treated as a basic symbol
The first step of Natural Language processing is text tokenization. For
example, a standard English tokenizer would segment the text "Hello world." into the
following three tokens.

> [Hello] [World] [.]

One observation is that the original input and tokenized sequence are **NOT
reversibly convertible**. For instance, the information that is no space between
“World” and “.” is dropped from the tokenized sequence, since e.g., `Tokenize(“World.”) == Tokenize(“World .”)`

SentencePiece treats the input text just as a sequence of Unicode characters. Whitespace is also handled as a normal symbol. To handle the whitespace as a basic token explicitly, SentencePiece first escapes the whitespace with a meta symbol "▁" (U+2581) as follows.

> Hello▁World.

Then, this text is segmented into small pieces, for example:

> [Hello] [▁Wor] [ld] [.]

Since the whitespace is preserved in the segmented text, we can detokenize the text without any ambiguities.

```
  detokenized = ''.join(pieces).replace('_', ' ')
```

This feature makes it possible to perform detokenization without relying on language-specific resources.

Note that we cannot apply the same lossless conversions when splitting the
sentence with standard word segmenters, since they treat the whitespace as a
special symbol. Tokenized sequences do not preserve the necessary information to restore the original sentence.

* (en) Hello world.   → [Hello] [World] [.]   \(A space between Hello and World\)
* (ja) こんにちは世界。  → [こんにちは] [世界] [。] \(No space between こんにちは and 世界\)

## Python module
SentencePiece provides Python wrapper that supports both SentencePiece training and segmentation.
For Linux (x64) environment, you can install Python binary package of SentencePiece with.

```
% pip install sentencepiece
```

For more detail, see [Python module](python/README.md)


## Required packages (C++)
The following tools and libraries are required to build SentencePiece:

* GNU autotools (autoconf automake libtool)
* C++11 compiler
* [protobuf](https://github.com/google/protobuf) library

On Ubuntu, autotools can be installed with apt-get:
```
% sudo apt-get install autoconf automake libtool pkg-config libprotobuf9v5 protobuf-compiler libprotobuf-dev
```
The name of the protobuf library is different between ubuntu distros. Please enter appropriate command for your Ubuntu version.

On ubuntu 14.04 LTS (Trusty Tahr):
```
% sudo apt-get install libprotobuf8
```

On ubuntu 16.04 LTS (Xenial Xerus):
```
% sudo apt-get install libprotobuf9v5
```

On ubuntu 17.10 (Artful Aardvark) and Later:
```
% sudo apt-get install libprotobuf10
```

On OSX, you can use brew:
```
% brew install protobuf autoconf automake libtool
```

If want to use self-prepared protobuf library, setup below environment variables before build:
```
% export PROTOBUF=<path_to_protobuf>
% export PROTOC="$PROTOBUF/bin/protoc"
% export PROTOBUF_LIBS="-L$PROTOBUF/lib -lprotobuf -D_THREAD_SAFE"
% export PROTOBUF_CFLAGS="-I$PROTOBUF/include -D_THREAD_SAFE" 
```

## Build and Install SentencePiece
```
% cd /path/to/sentencepiece
% ./autogen.sh
% ./configure
% make
% make check
% sudo make install
$ sudo ldconfig -v
```
## Train SentencePiece Model
```
% spm_train --input=<input> --model_prefix=<model_name> --vocab_size=8000 --model_type=<type>
```
* `--input`: one-sentence-per-line **raw** corpus file. No need to run
  tokenizer, normalizer or preprocessor. By default, SentencePiece normalizes
  the input with Unicode NFKC. You can pass a comma-separated list of files.
* `--model_prefix`: output model name prefix. `<model_name>.model` and `<model_name>.vocab` are generated.
* `--vocab_size`: vocabulary size, e.g., 8000, 16000, or 32000
* `--model_type`: model type. Choose from `unigram` (default), `bpe`, `char`, or `word`. The input sentence must be pretokenized when using `word` type.

Note that `spm_train` loads only the first `--input_sentence_size` sentences (default value is 10M).

Use `--help` flag to display all parameters for training.

## Encode raw text into sentence pieces/ids
```
% spm_encode --model=<model_file> --output_format=piece < input > output
% spm_encode --model=<model_file> --output_format=id < input > output
```

Use `--extra_options` flag to insert the BOS/EOS markers or reverse the input sequence.
```
% spm_encode --extra_options=eos (add </s> only)
% spm_encode --extra_options=bos:eos (add <s> and </s>)
% spm_encode --extra_options=reverse:bos:eos (reverse input and add <s> and </s>)
```

SentencePiece supports nbest segmentation and segmentation sampling with `--output_format=(nbest|sample)_(piece|id)` flags.
```
% spm_encode --model=<model_file> --output_format=sample_piece --nbest_size=-1 --alpha=0.5 < input > output
% spm_encode --model=<model_file> --output_format=nbest_id --nbest_size=10 < input > output
```

## Decode sentence pieces/ids into raw text
```
% spm_decode --model=<model_file> --input_format=piece < input > output
% spm_decode --model=<model_file> --input_format=id < input > output
```
Use `--extra_options` flag to decode the text in reverse order.
```
% spm_decode --extra_options=reverse < input > output
```

## End-to-End Example
```
% spm_train --input=data/botchan.txt --model_prefix=m --vocab_size=1000
unigram_model_trainer.cc(494) LOG(INFO) Starts training with :
input: "../data/botchan.txt"
... <snip>
unigram_model_trainer.cc(529) LOG(INFO) EM sub_iter=1 size=1100 obj=10.4973 num_tokens=37630 num_tokens/piece=34.2091
trainer_interface.cc(272) LOG(INFO) Saving model: m.model
trainer_interface.cc(281) LOG(INFO) Saving vocabs: m.vocab

% echo "I saw a girl with a telescope." | spm_encode --model=m.model
▁I ▁saw ▁a ▁girl ▁with ▁a ▁ te le s c o pe .

% echo "I saw a girl with a telescope." | spm_encode --model=m.model --output_format=id
9 459 11 939 44 11 4 142 82 8 28 21 132 6

% echo "9 459 11 939 44 11 4 142 82 8 28 21 132 6" | spm_decode --model=m.model --input_format=id
I saw a girl with a telescope.
```
You can find that the original input sentence is restored from the vocabulary id sequence.

## Export vocabulary list
```
% spm_export_vocab --model=<model_file> --output=<output file>
```
```<output file>``` stores a list of vocabulary and emission log probabilities. The vocabulary id corresponds to the line number in this file.

## Redefine special meta tokens
  By default, SentencePiece uses Unknown (&lt;unk&gt;), BOS (&lt;s&gt;) and EOS (&lt;/s&gt;) tokens which have the ids of 0, 1, and 2 respectively. We can redefine this mapping in training phase as follows.
  
```
% spm_train --bos_id=0 --eos_id=1 --unk_id=2 --input=... --model_prefix=...
```
When setting -1 id e.g., ```bos_id=-1```, this special token is disabled. Note that the unknow id cannot be removed. In addition, these ids must start with 0 and be continous. We can define an id for padding (&lt;pad&gt;). Padding id is disabled by default. You can assign an id as ```--pad_id=3```.  

If you want to assign another special tokens, please see [Use custom symbols](doc/special_symbols.md).

## Advanced topics

* [SentencePiece Experiments](doc/experiments.md)
* [SentencePieceProcessor C++ API](doc/api.md)
* [Use custom text normalization rules](doc/normalization.md)
* [Use custom symbols](doc/special_symbols.md)
* [Segmentation and training algorithms in detail]
