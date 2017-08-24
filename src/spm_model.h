#include <string>
#include "util.h"
#include "sentencepiece_processor.h"

class SentencePieceModel {
  sentencepiece::SentencePieceProcessor sp;
public:
  SentencePieceModel(std::string filename);
  std::string encode(std::string line);
  std::string decode(std::string line);
};

SentencePieceModel::SentencePieceModel(std::string filename) {
  sp.LoadOrDie(filename);
}

std::string SentencePieceModel::encode(std::string line) {
  std::vector<std::string> sps;
  sp.Encode(line, &sps);
  return sentencepiece::string_util::Join(sps, " ");  
}

std::string SentencePieceModel::decode(std::string line) {
  std::vector<std::string> sps = sentencepiece::string_util::Split(line, " ");
  std::string detok;
  sp.Decode(sps, &detok);
  return detok;
}
