//----------------------- API interface for use in .NET -----------------------//
#include "sentencepiece_processor.h"

#ifdef linux
#define API extern "C" 
#else
#define API extern "C" __declspec(dllexport)
#endif

#define CALLING_CONV //__stdcall

API sentencepiece::SentencePieceProcessor* CALLING_CONV __SP_Init(char* modelFilename, char* vocabFilename, int threshold);
API void CALLING_CONV __SP_Finalize(sentencepiece::SentencePieceProcessor* sp);
API char* CALLING_CONV __SP_Encode(sentencepiece::SentencePieceProcessor* sp, char* input, int len);
API char* CALLING_CONV __SP_Decode(sentencepiece::SentencePieceProcessor* sp, char* input, int len);
API void CALLING_CONV __SP_Free(char* result);