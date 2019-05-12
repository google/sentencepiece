#include "../src/sentencepiece_processor.h"
#include "com_github_google_sentencepiece_SentencePieceJNI.h"

using sentencepiece::SentencePieceProcessor;
using sentencepiece::util::Status;
using sentencepiece::util::min_string_view;

int throwStatus(JNIEnv *env, const Status &status) {
    if (!status.ok()) {
        jclass cls = env->FindClass("com/github/google/sentencepiece/SentencePieceException");
        env->ThrowNew(cls, status.ToString().c_str());
    }
    return status.code();
}

jbyteArray stringToJbyteArray(JNIEnv *env, const std::string &str) {
    jbyteArray array = env->NewByteArray(str.length());
    env->SetByteArrayRegion(array, 0, str.length(), reinterpret_cast<const jbyte *>(str.c_str()));
    return array;
}

std::string jbyteArrayToString(JNIEnv *env, jbyteArray array) {
    jsize len = env->GetArrayLength(array);

    void *str = env->GetPrimitiveArrayCritical(array, nullptr);
    std::string s((char *) str, len);
    env->ReleasePrimitiveArrayCritical(array, str, JNI_ABORT);

    return s;
}

jobjectArray vectorStringToJobjectArrayByteArray(JNIEnv *env, const std::vector<std::string> &vec) {
    jobjectArray array = env->NewObjectArray(vec.size(), env->FindClass("[B"), nullptr);
    for (int i = 0; i < vec.size(); ++i) {
        env->SetObjectArrayElement(array, i, stringToJbyteArray(env, vec[i]));
    }
    return array;
}

jobjectArray vectorVectorStringToJobjectArrayObjectArrayByteArray(JNIEnv *env, const std::vector<std::vector<std::string>> &vec) {
    jobjectArray array = env->NewObjectArray(vec.size(), env->FindClass("[[B"), nullptr);
    for (int i = 0; i < vec.size(); ++i) {
        env->SetObjectArrayElement(array, i, vectorStringToJobjectArrayByteArray(env, vec[i]));
    }
    return array;
}

void jobjectArrayByteArrayToVectorString(JNIEnv *env, jobjectArray array, std::vector<std::string>* vec) {
    jsize len = env->GetArrayLength(array);
    vec->resize(len);
    for (int i = 0; i < len; ++i) {
        (*vec)[i] = jbyteArrayToString(env, (jbyteArray) env->GetObjectArrayElement(array, i));
    }
}

jintArray vectorIntToJintArray(JNIEnv *env, const std::vector<int> &vec) {
    jintArray array = env->NewIntArray(vec.size());
    env->SetIntArrayRegion(array, 0, vec.size(), vec.data());
    return array;
}

jobjectArray vectorVectorIntToJobjectArrayIntArray(JNIEnv *env, const std::vector<std::vector<int>> &vec) {
    jobjectArray array = env->NewObjectArray(vec.size(), env->FindClass("[I"), nullptr);
    for (int i = 0; i < vec.size(); ++i) {
        env->SetObjectArrayElement(array, i, vectorIntToJintArray(env, vec[i]));
    }
    return array;
}

std::vector<int> jintArrayToVectorInt(JNIEnv *env, jintArray array) {
    jsize len = env->GetArrayLength(array);

    void *data = env->GetPrimitiveArrayCritical(array, nullptr);
    std::vector<int> vec((int*) data, ((int*) data) + len);
    env->ReleasePrimitiveArrayCritical(array, data, JNI_ABORT);

    return vec;
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppCtor
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppCtor
        (JNIEnv *env, jclass cls) {
    auto* spp = new SentencePieceProcessor();
    return (jlong) spp;
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppDtor
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppDtor
        (JNIEnv *env, jclass cls, jlong ptr) {
    auto* spp = (SentencePieceProcessor*) ptr;
    delete spp;
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppLoad
 * Signature: (J[B)V
 */
JNIEXPORT void JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppLoad
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray filename) {
    auto* spp = (SentencePieceProcessor*) ptr;

    jsize len = env->GetArrayLength(filename);

    void* str = env->GetPrimitiveArrayCritical(filename, nullptr);
    Status status = spp->Load(min_string_view(static_cast<const char *>(str), len));
    env->ReleasePrimitiveArrayCritical(filename, str, JNI_ABORT);

    throwStatus(env, status);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppLoadOrDie
 * Signature: (J[B)V
 */
JNIEXPORT void JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppLoadOrDie
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray filename) {
    auto* spp = (SentencePieceProcessor*) ptr;

    jsize len = env->GetArrayLength(filename);

    void* str = env->GetPrimitiveArrayCritical(filename, nullptr);
    spp->LoadOrDie(min_string_view(static_cast<const char *>(str), len));
    env->ReleasePrimitiveArrayCritical(filename, str, JNI_ABORT);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppLoadFromSerializedProto
 * Signature: (J[B)V
 */
JNIEXPORT void JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppLoadFromSerializedProto
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray serialized) {
    auto* spp = (SentencePieceProcessor*) ptr;

    jsize len = env->GetArrayLength(serialized);

    void* str = env->GetPrimitiveArrayCritical(serialized, nullptr);
    Status status = spp->LoadFromSerializedProto(min_string_view(static_cast<const char *>(str), len));
    env->ReleasePrimitiveArrayCritical(serialized, str, JNI_ABORT);

    throwStatus(env, status);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppSetEncodeExtraOptions
 * Signature: (J[B)V
 */
JNIEXPORT void JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppSetEncodeExtraOptions
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray extra_option) {
    auto* spp = (SentencePieceProcessor*) ptr;

    jsize len = env->GetArrayLength(extra_option);

    void* str = env->GetPrimitiveArrayCritical(extra_option, nullptr);
    Status status = spp->SetEncodeExtraOptions(min_string_view(static_cast<const char *>(str), len));
    env->ReleasePrimitiveArrayCritical(extra_option, str, JNI_ABORT);

    throwStatus(env, status);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppSetDecodeExtraOptions
 * Signature: (J[B)V
 */
JNIEXPORT void JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppSetDecodeExtraOptions
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray extra_option) {
    auto* spp = (SentencePieceProcessor*) ptr;

    jsize len = env->GetArrayLength(extra_option);

    void* str = env->GetPrimitiveArrayCritical(extra_option, nullptr);
    Status status = spp->SetDecodeExtraOptions(min_string_view(static_cast<const char *>(str), len));
    env->ReleasePrimitiveArrayCritical(extra_option, str, JNI_ABORT);

    throwStatus(env, status);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppSetVocabulary
 * Signature: (J[[B)V
 */
JNIEXPORT void JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppSetVocabulary
        (JNIEnv *env, jclass cls, jlong ptr, jobjectArray array) {
    auto* spp = (SentencePieceProcessor*) ptr;

    std::vector<std::string> valid_vocab;
    jobjectArrayByteArrayToVectorString(env, array, &valid_vocab);

    Status status = spp->SetVocabulary(valid_vocab);
    throwStatus(env, status);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppResetVocabulary
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppResetVocabulary
        (JNIEnv *env, jclass cls, jlong ptr) {
    auto* spp = (SentencePieceProcessor*) ptr;

    Status status = spp->ResetVocabulary();
    throwStatus(env, status);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppLoadVocabulary
 * Signature: (J[BI)V
 */
JNIEXPORT void JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppLoadVocabulary
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray filename, jint threshold) {
    auto* spp = (SentencePieceProcessor*) ptr;

    jsize len = env->GetArrayLength(filename);

    void* str = env->GetPrimitiveArrayCritical(filename, nullptr);
    Status status = spp->LoadVocabulary(min_string_view(static_cast<const char *>(str), len), threshold);
    env->ReleasePrimitiveArrayCritical(filename, str, JNI_ABORT);

    throwStatus(env, status);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppEncodeAsPieces
 * Signature: (J[B)[[B
 */
JNIEXPORT jobjectArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppEncodeAsPieces
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray input) {
    auto* spp = (SentencePieceProcessor*) ptr;

    std::vector<std::string> pieces;
    jsize len = env->GetArrayLength(input);

    void* str = env->GetPrimitiveArrayCritical(input, nullptr);
    Status status = spp->Encode(min_string_view(static_cast<const char *>(str), len), &pieces);
    env->ReleasePrimitiveArrayCritical(input, str, JNI_ABORT);

    if (throwStatus(env, status)) {
        return nullptr;
    }
    return vectorStringToJobjectArrayByteArray(env, pieces);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppEncodeAsIds
 * Signature: (J[B)[I
 */
JNIEXPORT jintArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppEncodeAsIds
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray input) {
    auto* spp = (SentencePieceProcessor*) ptr;

    std::vector<int> ids;
    jsize len = env->GetArrayLength(input);

    void* str = env->GetPrimitiveArrayCritical(input, nullptr);
    Status status = spp->Encode(min_string_view(static_cast<const char *>(str), len), &ids);
    env->ReleasePrimitiveArrayCritical(input, str, JNI_ABORT);

    if (throwStatus(env, status)) {
        return nullptr;
    }
    return vectorIntToJintArray(env, ids);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppDecodePieces
 * Signature: (J[[B)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppDecodePieces
        (JNIEnv *env, jclass cls, jlong ptr, jobjectArray array) {
    auto* spp = (SentencePieceProcessor*) ptr;

    std::vector<std::string> pieces;
    jobjectArrayByteArrayToVectorString(env, array, &pieces);

    std::string detokenized;
    Status status = spp->Decode(pieces, &detokenized);

    if (throwStatus(env, status)) {
        return nullptr;
    }
    return stringToJbyteArray(env, detokenized);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppDecodeIds
 * Signature: (J[I)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppDecodeIds
        (JNIEnv *env, jclass cls, jlong ptr, jintArray array) {
    auto* spp = (SentencePieceProcessor*) ptr;

    std::vector<int> ids = jintArrayToVectorInt(env, array);

    std::string detokenized;
    Status status = spp->Decode(ids, &detokenized);

    if (throwStatus(env, status)) {
        return nullptr;
    }
    return stringToJbyteArray(env, detokenized);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppNBestEncodeAsPieces
 * Signature: (J[BI)[[[B
 */
JNIEXPORT jobjectArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppNBestEncodeAsPieces
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray input, jint nbest_size) {
    auto* spp = (SentencePieceProcessor*) ptr;

    std::vector<std::vector<std::string>> pieces;
    jsize len = env->GetArrayLength(input);

    void* str = env->GetPrimitiveArrayCritical(input, nullptr);
    Status status = spp->NBestEncode(min_string_view(static_cast<const char *>(str), len), nbest_size, &pieces);
    env->ReleasePrimitiveArrayCritical(input, str, JNI_ABORT);

    if (throwStatus(env, status)) {
        return nullptr;
    }
    return vectorVectorStringToJobjectArrayObjectArrayByteArray(env, pieces);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppNBestEncodeAsIds
 * Signature: (J[BI)[[I
 */
JNIEXPORT jobjectArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppNBestEncodeAsIds
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray input, jint nbest_size) {
    auto* spp = (SentencePieceProcessor*) ptr;

    std::vector<std::vector<int>> ids;
    jsize len = env->GetArrayLength(input);

    void* str = env->GetPrimitiveArrayCritical(input, nullptr);
    Status status = spp->NBestEncode(min_string_view(static_cast<const char *>(str), len), nbest_size, &ids);
    env->ReleasePrimitiveArrayCritical(input, str, JNI_ABORT);

    if (throwStatus(env, status)) {
        return nullptr;
    }
    return vectorVectorIntToJobjectArrayIntArray(env, ids);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppSampleEncodeAsPieces
 * Signature: (J[BIF)[[B
 */
JNIEXPORT jobjectArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppSampleEncodeAsPieces
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray input, jint nbest_size, jfloat alpha) {
    auto* spp = (SentencePieceProcessor*) ptr;

    std::vector<std::string> pieces;
    jsize len = env->GetArrayLength(input);

    void* str = env->GetPrimitiveArrayCritical(input, nullptr);
    Status status = spp->SampleEncode(min_string_view(static_cast<const char *>(str), len), nbest_size, alpha, &pieces);
    env->ReleasePrimitiveArrayCritical(input, str, JNI_ABORT);

    if (throwStatus(env, status)) {
        return nullptr;
    }
    return vectorStringToJobjectArrayByteArray(env, pieces);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppSampleEncodeAsIds
 * Signature: (J[BIF)[I
 */
JNIEXPORT jintArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppSampleEncodeAsIds
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray input, jint nbest_size, jfloat alpha) {
    auto* spp = (SentencePieceProcessor*) ptr;

    std::vector<int> ids;
    jsize len = env->GetArrayLength(input);

    void* str = env->GetPrimitiveArrayCritical(input, nullptr);
    Status status = spp->SampleEncode(min_string_view(static_cast<const char *>(str), len), nbest_size, alpha, &ids);
    env->ReleasePrimitiveArrayCritical(input, str, JNI_ABORT);

    if (throwStatus(env, status)) {
        return nullptr;
    }
    return vectorIntToJintArray(env, ids);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppEncodeAsSerializedProto
 * Signature: (J[B)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppEncodeAsSerializedProto
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray input) {
    auto* spp = (SentencePieceProcessor*) ptr;

    jsize len = env->GetArrayLength(input);

    void* str = env->GetPrimitiveArrayCritical(input, nullptr);
    sentencepiece::util::bytes bytes = spp->EncodeAsSerializedProto(min_string_view(static_cast<const char *>(str), len));
    env->ReleasePrimitiveArrayCritical(input, str, JNI_ABORT);

    return stringToJbyteArray(env, bytes);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppSampleEncodeAsSerializedProto
 * Signature: (J[BIF)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppSampleEncodeAsSerializedProto
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray input, jint nbest_size, jfloat alpha) {
    auto* spp = (SentencePieceProcessor*) ptr;

    jsize len = env->GetArrayLength(input);

    void* str = env->GetPrimitiveArrayCritical(input, nullptr);
    sentencepiece::util::bytes bytes = spp->SampleEncodeAsSerializedProto(min_string_view(static_cast<const char *>(str), len), nbest_size, alpha);
    env->ReleasePrimitiveArrayCritical(input, str, JNI_ABORT);

    return stringToJbyteArray(env, bytes);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppNBestEncodeAsSerializedProto
 * Signature: (J[BI)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppNBestEncodeAsSerializedProto
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray input, jint nbest_size) {
    auto* spp = (SentencePieceProcessor*) ptr;

    jsize len = env->GetArrayLength(input);

    void* str = env->GetPrimitiveArrayCritical(input, nullptr);
    sentencepiece::util::bytes bytes = spp->NBestEncodeAsSerializedProto(min_string_view(static_cast<const char *>(str), len), nbest_size);
    env->ReleasePrimitiveArrayCritical(input, str, JNI_ABORT);

    return stringToJbyteArray(env, bytes);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppDecodePiecesAsSerializedProto
 * Signature: (J[[B)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppDecodePiecesAsSerializedProto
        (JNIEnv *env, jclass cls, jlong ptr, jobjectArray array) {
    auto* spp = (SentencePieceProcessor*) ptr;

    std::vector<std::string> pieces;
    jobjectArrayByteArrayToVectorString(env, array, &pieces);

    sentencepiece::util::bytes bytes = spp->DecodePiecesAsSerializedProto(pieces);
    return stringToJbyteArray(env, bytes);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppDecodeIdsAsSerializedProto
 * Signature: (J[I)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppDecodeIdsAsSerializedProto
        (JNIEnv *env, jclass cls, jlong ptr, jintArray array) {
    auto* spp = (SentencePieceProcessor*) ptr;

    std::vector<int> ids = jintArrayToVectorInt(env, array);

    sentencepiece::util::bytes bytes = spp->DecodeIdsAsSerializedProto(ids);
    return stringToJbyteArray(env, bytes);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppGetPieceSize
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppGetPieceSize
        (JNIEnv *env, jclass cls, jlong ptr) {
    auto* spp = (SentencePieceProcessor*) ptr;
    return spp->GetPieceSize();
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppPieceToId
 * Signature: (J[B)I
 */
JNIEXPORT jint JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppPieceToId
        (JNIEnv *env, jclass cls, jlong ptr, jbyteArray piece) {
    auto* spp = (SentencePieceProcessor*) ptr;

    jsize len = env->GetArrayLength(piece);

    void* str = env->GetPrimitiveArrayCritical(piece, nullptr);
    int id = spp->PieceToId(min_string_view(static_cast<const char *>(str), len));
    env->ReleasePrimitiveArrayCritical(piece, str, JNI_ABORT);

    return id;
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppIdToPiece
 * Signature: (JI)[B
 */
JNIEXPORT jbyteArray JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppIdToPiece
        (JNIEnv *env, jclass cls, jlong ptr, jint id) {
    auto* spp = (SentencePieceProcessor*) ptr;

    const std::string &piece = spp->IdToPiece(id);
    return stringToJbyteArray(env, piece);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppGetScore
 * Signature: (JI)F
 */
JNIEXPORT jfloat JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppGetScore
        (JNIEnv *env, jclass cls, jlong ptr, jint id) {
    auto* spp = (SentencePieceProcessor*) ptr;
    return spp->GetScore(id);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppIsUnknown
 * Signature: (JI)Z
 */
JNIEXPORT jboolean JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppIsUnknown
        (JNIEnv *env, jclass cls, jlong ptr, jint id) {
    auto* spp = (SentencePieceProcessor*) ptr;
    return spp->IsUnknown(id);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppIsControl
 * Signature: (JI)Z
 */
JNIEXPORT jboolean JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppIsControl
        (JNIEnv *env, jclass cls, jlong ptr, jint id) {
    auto* spp = (SentencePieceProcessor*) ptr;
    return spp->IsControl(id);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppIsUnused
 * Signature: (JI)Z
 */
JNIEXPORT jboolean JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppIsUnused
        (JNIEnv *env, jclass cls, jlong ptr, jint id) {
    auto* spp = (SentencePieceProcessor*) ptr;
    return spp->IsUnused(id);
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppUnkId
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppUnkId
        (JNIEnv *env, jclass cls, jlong ptr) {
    auto* spp = (SentencePieceProcessor*) ptr;
    return spp->unk_id();
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppBosId
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppBosId
        (JNIEnv *env, jclass cls, jlong ptr) {
    auto* spp = (SentencePieceProcessor*) ptr;
    return spp->bos_id();
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppEosId
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppEosId
        (JNIEnv *env, jclass cls, jlong ptr) {
    auto* spp = (SentencePieceProcessor*) ptr;
    return spp->eos_id();
}

/*
 * Class:     com_github_google_sentencepiece_SentencePieceJNI
 * Method:    sppPadId
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_github_google_sentencepiece_SentencePieceJNI_sppPadId
        (JNIEnv *env, jclass cls, jlong ptr) {
    auto* spp = (SentencePieceProcessor*) ptr;
    return spp->pad_id();
}
