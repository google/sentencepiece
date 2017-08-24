#include <iostream>
#include "spm_model.h"

int main(int argc, char *argv[]) {
  std::string filename(argv[1]);
  SentencePieceModel* spm = new SentencePieceModel(filename);
  std::string encoded = spm->encode("this is a test");
  std::cout << encoded << std::endl;
  std::cout << spm->decode(encoded) << std::endl;
  return 0;
}
