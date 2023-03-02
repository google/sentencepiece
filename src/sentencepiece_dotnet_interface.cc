//----------------------- API interface for use in .NET -----------------------//
#include "sentencepiece_dotnet_interface.h"

#include "common.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/str_split.h"
#include "third_party/absl/strings/string_view.h"

API sentencepiece::SentencePieceProcessor* CALLING_CONV __SP_Init(char* modelFilename, char* vocabFilename, int threshold)
{
    sentencepiece::SentencePieceProcessor* sp = new sentencepiece::SentencePieceProcessor();

    absl::string_view model_filename((char*)modelFilename, strlen(modelFilename));
    CHECK_OK(sp->Load(model_filename));

    if (vocabFilename) {
        absl::string_view vocab_filename((char*)vocabFilename, strlen(vocabFilename));
        CHECK_OK(sp->LoadVocabulary(vocab_filename, threshold));
    }
    return (sp);
}

API void CALLING_CONV __SP_Finalize(sentencepiece::SentencePieceProcessor* sp)
{
    delete sp;
}

API char* CALLING_CONV __SP_Encode(sentencepiece::SentencePieceProcessor* sp, char* input, int len)
{
    std::vector<std::string> sps;
    absl::string_view line(input, len);
    CHECK_OK(sp->Encode(line, &sps));
    std::string s = absl::StrJoin(sps, " ");
    char* result = new char[s.size() + 1];
    memcpy(result, (void*)s.c_str(), s.size() + 1);
    return (result);
}
API char* CALLING_CONV __SP_Decode(sentencepiece::SentencePieceProcessor* sp, char* input, int len)
{
    absl::string_view line(input, len);
    std::vector<std::string> pieces = absl::StrSplit(line, " ");

    std::string s;
    CHECK_OK(sp->Decode(pieces, &s));
    char* result = new char[s.size() + 1];
    memcpy(result, (void*)s.c_str(), s.size() + 1);
    return (result);
}

API void CALLING_CONV __SP_Free(char* result)
{
    delete result;
}