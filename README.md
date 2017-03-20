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

## Required packages
The following tools and libraries are required to build SentencePiece:

* GNU autotools (autoconf automake libtool)
* C++11 compiler
* [protobuf](https://github.com/google/protobuf) library

On Ubuntu, autotools and protobuf library can be install with apt-get:
```
% sudo apt-get install autoconf automake libtool libprotobuf9v5 protobuf-compiler libprotobuf-dev
```
(If `libprotobuf9v5` is not found, try `libprotobuf-c++` instead.)

## Build and Install SentencePiece
```
% cd /path/to/sentencepiece
% ./autogen.sh
% ./configure
% make
% make check
% sudo make install
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

## Experiments
### Experimental settings
We have evaluated SentencePiece segmentation with the following configurations.

*   Segmentation algorithms:
    *   **BPE** (Byte Pair
        Encoding) [[Sennrich et al.](http://www.aclweb.org/anthology/P16-1162)]] (`--model_type=bpe`)
    *   **Unigram**. Language-model based segmentation. (`--model_type=unigram`)

*   pretokenization methods:
    *   **NoPretok**: No pretokenization. We train SentencePiece directly from
        raw sentences (`--split_by_whitespace=false`).
    *   **WsPretok**: Trains SentencePiece model from the sentences tokenized by
        whitespaces (`--split_by_whitespace=true`). When handling CJK, this setting is almost equivalent to **NoPretok**.
    *   **MosesPretok**: Trains SentencePiece model from sentences tokenized
        by [Moses tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl). We used [KyTea](http://www.phontron.com/kytea/) for
        Japanese and in-house segmenters for Korean and Chinese respectively.

*   NMT parameters: ([Google’s Neural Machine Translation System](https://arxiv.org/pdf/1609.08144.pdf) is applied for all experiments.)
    *   16k shared vocabulary (Shares the same vocabulary for source and
        target. We train single SentencePiece model by concatenating raw source
        and target sentences.)
    *   Dropout prob: 0.2
    *   num nodes: 512
    *   num lstms: 8

*   Evaluation metrics:
    *   Case-sensitive BLEU on detokenized text with NIST scorer.
    *   For CJK, the same word segmenters are applied prior to NIST scorer.
    *   No detokenizer is applied for **NoPretok** and **WsPretok**, which can
        directly emit detokenized sentences.
    *   Applied [Moses detokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/detokenizer.perl) and in-house rule-based detokenizer (CJK) for **MosesPretok**.

*   Data sets:
    *   [KFTT](http://www.phontron.com/kftt/index.html)
    *   [MultiUN](http://opus.lingfil.uu.se/MultiUN.php) (First 5M and next
        5k/5k sentences are used for training and development/testing respectively.)
    *   [WMT16](http://www.statmt.org/WMT16/)
    *   In-house: (Used 5M parallel sentences for training)

**NoPretok** and **WsPretok** do not use any language-dependent resources.
**BPE**+**MosePretok** is almost the same configuration used in [[Sennrich et al.](http://www.aclweb.org/anthology/P16-1162)] and [[Wu et al.](https://arxiv.org/pdf/1609.08144.pdf)].

### Results (BLEU scores)
|Language Pair|BPE(NoPretok)|BPE(WsPretok)|BPE(MosesPretok)|Unigram(NoPretok)|Unigram(WsPretok)|Unigram(MosesPretok)
|---|---|---|---|---|---|---|
|KFTT en-ja|	0.2796|	0.281|	0.286|	0.2806|	0.280|	0.2871|
|KFTT ja-en|	0.1943|	0.208|	0.1967|	0.1985|	0.2148|	0.198|
|MultiUN ar-en|	0.5268|	0.5414|	0.5381|	0.5317|	0.5449|	0.5401|
|MultiUN en-ar|	0.4039|	0.4147|	0.4012|	0.4084|	0.4172|	0.3991|
|MultiUN en-zh|	0.4155|	0.4186|	0.395|	0.4214|	0.4165|	0.399|
|MultiUN zh-en|	0.46|	0.4716|	0.4806|	0.4644|	0.4711|	0.4759|
|In house en-ko|	0.178|	0.1851|	0.1893|	0.1846|	0.1872|	0.1890|
|In house ko-en|	0.1786|	0.1954|	0.1994|	0.1845|	0.1956|	0.2015|
|WMT16 cs-en|	0.1987|	0.2252|	0.2231|	0.2164|	0.2228|	0.2238|
|WMT16 de-en|	0.3194|	0.3348|	0.3374|	0.3261|	0.3375|	0.3398|
|WMT16 en-cs|	0.1607|	0.1827|	0.1812|	0.1722|	0.1778|	0.179|
|WMT16 en-de|	0.2847|	0.3029|	0.3013|	0.2946|	0.3000|	0.3053|
|WMT16 en-fi|	0.1434|	0.1528|	0.1499|	0.1472|	0.1568|	0.1517|
|WMT16 en-ru|	0.1884|	0.1973|	0.1989|	0.19|	0.1982|	0.1903|
|WMT16 fi-en|	0.1775|	0.1867|	0.1877|	0.182|	0.1882|	0.1865|
|WMT16 ru-en|	0.2042|	0.2229|	0.2194|	0.2087|	0.2201|	0.2155|

*   **MosesPretok** does not always improve BLEU scores. Comparable
    accuracy can be obtained without using language-dependent resources in many
    language pairs.
*   Whitespace pretokenization is a reasonable choice. It does not use language-specific resources.
*   **NoPretok** shows poor BLEU scores. Unigrams are more robust than BPE when no pretokenizer is applied.

## Advanced topics

* [SentencePieceProcessor C++ API](doc/api.md)
* [Use custom text normalization rules](doc/normalization.md)
* [Use custom symbols](doc/special_symbols.md)
* [Segmentation and training algorithms in detail]
