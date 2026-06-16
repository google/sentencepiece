# SentencePieceProcessor C++ API

## Load SentencePiece model
To start working with the SentencePiece model, you will want to include the `sentencepiece_processor.h` header file.
Then instantiate the `sentencepiece::SentencePieceProcessor` class and call the `Load` method to load the model using a file path or `std::istream`.

```C++
#include <sentencepiece_processor.h>

sentencepiece::SentencePieceProcessor processor;
const auto status = processor.Load("//path/to/model.model");
if (!status.ok()) {
   std::cerr << status.ToString() << std::endl;
   // error
}

// You can also load a serialized model from std::string.
// const std::string str = // Load blob contents from a file.
// auto status = processor.LoadFromSerializedProto(str);
```

## Tokenize text (preprocessing)
Call the `SentencePieceProcessor::Encode` method to tokenize text.

```C++
std::vector<std::string> pieces;
processor.Encode("This is a test.", &pieces);
for (const std::string &token : pieces) {
  std::cout << token << std::endl;
}
```

You will obtain the sequence of vocabulary IDs as follows:

```C++
std::vector<int> ids;
processor.Encode("This is a test.", &ids);
for (const int id : ids) {
  std::cout << id << std::endl;
}
```

## Detokenize text (postprocessing)
Call the `SentencePieceProcessor::Decode` method to detokenize a sequence of pieces or IDs into text. In general, it is guaranteed that the detokenization is an inverse operation of Encode, i.e., `Decode(Encode(Normalize(input))) == Normalize(input)`.

```C++
std::vector<std::string> pieces = { "▁This", "▁is", "▁a", "▁", "te", "st", "." };   // sequence of pieces
std::string text;
processor.Decode(pieces, &text);
std::cout << text << std::endl;

std::vector<int> ids = { 451, 26, 20, 3, 158, 128, 12  };   // sequence of ids
processor.Decode(ids, &text);
std::cout << text << std::endl;
```

## Sampling (subword regularization)
Call the `SentencePieceProcessor::SampleEncode` method to sample one segmentation.

```C++
std::vector<std::string> pieces;
processor.SampleEncode("This is a test.", &pieces, -1, 0.2);

std::vector<int> ids;
processor.SampleEncode("This is a test.", &ids, -1, 0.2);
```
SampleEncode has two sampling parameters, `nbest_size` and `alpha`, which correspond to `l` and `alpha` in the [original paper](https://arxiv.org/abs/1804.10959). When `nbest_size` is -1, one segmentation is sampled from all hypotheses with forward-filtering and backward sampling algorithm.

## Training
Call the `SentencePieceTrainer::Train` function to train a SentencePiece model.

You can pass training parameters as a single command-line-like string:

```C++
#include <sentencepiece_trainer.h>

sentencepiece::SentencePieceTrainer::Train("--input=test/botchan.txt --model_prefix=m --vocab_size=1000");
```

Alternatively, you can pass parameters as a `std::unordered_map<std::string, std::string>`:

```C++
#include <sentencepiece_trainer.h>

sentencepiece::SentencePieceTrainer::Train({
  {"input", "test/botchan.txt"},
  {"model_prefix", "m"},
  {"vocab_size", "1000"}
});
```



## Vocabulary management
Use the following methods to convert between IDs and pieces.

```C++
processor.GetPieceSize();   // returns the vocabulary size.
processor.PieceToId("foo");  // returns the vocab id of "foo"
processor.IdToPiece(10);     // returns the string representation of id 10.
processor.IsUnknown(0);      // returns true if the given id is an unknown token. e.g., <unk>
processor.IsControl(10);     // returns true if the given id is a control token. e.g., <s>, </s>
```

