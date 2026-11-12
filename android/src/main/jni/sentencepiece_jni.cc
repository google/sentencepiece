// Copyright 2026 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <jni.h>
#include <string>
#include <vector>

#include "sentencepiece_processor.h"

namespace {

// Converts a Java jstring to a std::string. Returns empty string on failure.
std::string JStringToStdString(JNIEnv* env, jstring jstr) {
  if (jstr == nullptr) return "";
  const jboolean is_copy = JNI_FALSE;
  const jchar* chars = env->GetStringChars(jstr, &is_copy);
  if (chars == nullptr) return "";
  jsize len = env->GetStringLength(jstr);
  std::string result(reinterpret_cast<const char*>(chars),
                     static_cast<size_t>(len * 2));
  env->ReleaseStringChars(jstr, chars);
  // UTF-16 to UTF-8 conversion.
  std::string utf8;
  utf8.reserve(result.size());
  for (size_t i = 0; i < result.size();) {
    unsigned int codepoint = 0;
    unsigned char lead = static_cast<unsigned char>(result[i]);
    if (lead < 0x80) {
      codepoint = lead;
      i += 1;
    } else if ((lead & 0xE0) == 0xC0) {
      codepoint = (lead & 0x1F) << 6 | (result[i + 1] & 0x3F);
      i += 2;
    } else if ((lead & 0xF0) == 0xE0) {
      codepoint = (lead & 0x0F) << 12 | (result[i + 1] & 0x3F) << 6 |
                  (result[i + 2] & 0x3F);
      i += 3;
    } else if ((lead & 0xF8) == 0xF0) {
      codepoint = (lead & 0x07) << 18 | (result[i + 1] & 0x3F) << 12 |
                  (result[i + 2] & 0x3F) << 6 | (result[i + 3] & 0x3F);
      i += 4;
    } else {
      i += 1;
      continue;
    }
    if (codepoint < 0x80) {
      utf8 += static_cast<char>(codepoint);
    } else if (codepoint < 0x800) {
      utf8 += static_cast<char>(0xC0 | (codepoint >> 6));
      utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
    } else if (codepoint < 0x10000) {
      utf8 += static_cast<char>(0xE0 | (codepoint >> 12));
      utf8 += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
      utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
    } else {
      utf8 += static_cast<char>(0xF0 | (codepoint >> 18));
      utf8 += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
      utf8 += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
      utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
  }
  return utf8;
}

// Converts a std::string to a Java jstring (UTF-8 input).
jstring StdStringToJString(JNIEnv* env, const std::string& str) {
  // Convert UTF-8 to UTF-16.
  std::vector<jchar> utf16;
  utf16.reserve(str.size());
  for (size_t i = 0; i < str.size();) {
    unsigned int codepoint = 0;
    unsigned char lead = static_cast<unsigned char>(str[i]);
    if (lead < 0x80) {
      codepoint = lead;
      i += 1;
    } else if ((lead & 0xE0) == 0xC0) {
      codepoint = ((lead & 0x1F) << 6) | (str[i + 1] & 0x3F);
      i += 2;
    } else if ((lead & 0xF0) == 0xE0) {
      codepoint = ((lead & 0x0F) << 12) | ((str[i + 1] & 0x3F) << 6) |
                  (str[i + 2] & 0x3F);
      i += 3;
    } else if ((lead & 0xF8) == 0xF0) {
      codepoint = ((lead & 0x07) << 18) | ((str[i + 1] & 0x3F) << 12) |
                  ((str[i + 2] & 0x3F) << 6) | (str[i + 3] & 0x3F);
      i += 4;
    } else {
      i += 1;
      continue;
    }
    if (codepoint < 0x10000) {
      utf16.push_back(static_cast<jchar>(codepoint));
    } else {
      codepoint -= 0x10000;
      utf16.push_back(static_cast<jchar>(0xD800 | (codepoint >> 10)));
      utf16.push_back(static_cast<jchar>(0xDC00 | (codepoint & 0x3FF)));
    }
  }
  return env->NewString(utf16.data(), static_cast<jsize>(utf16.size()));
}

// Returns a SentencePieceProcessor* stored as a jlong field.
sentencepiece::SentencePieceProcessor* GetProcessor(JNIEnv* env, jobject thiz,
                                                    jfieldID ptr_field) {
  jlong ptr = env->GetLongField(thiz, ptr_field);
  return reinterpret_cast<sentencepiece::SentencePieceProcessor*>(ptr);
}

}  // namespace

// ---------------------------------------------------------------------------
// SentencePieceProcessor JNI
// ---------------------------------------------------------------------------

extern "C" JNIEXPORT jlong JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeCreate(
    JNIEnv* env, jobject thiz) {
  auto* processor = new sentencepiece::SentencePieceProcessor();
  return reinterpret_cast<jlong>(processor);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeLoad(
    JNIEnv* env, jobject thiz, jlong native_ptr, jstring model_path) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  if (processor == nullptr) return -1;
  std::string path = JStringToStdString(env, model_path);
  sentencepiece::absl::Status status = processor->Load(path);
  return status.ok() ? 0 : -1;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeLoadFromBytes(
    JNIEnv* env, jobject thiz, jlong native_ptr, jbyteArray model_bytes) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  if (processor == nullptr) return -1;
  if (model_bytes == nullptr) return -1;
  jsize len = env->GetArrayLength(model_bytes);
  jbyte* bytes = env->GetByteArrayElements(model_bytes, nullptr);
  std::string serialized(reinterpret_cast<const char*>(bytes),
                         static_cast<size_t>(len));
  env->ReleaseByteArrayElements(model_bytes, bytes, JNI_ABORT);
  sentencepiece::absl::Status status =
      processor->LoadFromSerializedProto(serialized);
  return status.ok() ? 0 : -1;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeEncodeAsPieces(
    JNIEnv* env, jobject thiz, jlong native_ptr, jstring text) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  jclass string_class = env->FindClass("java/lang/String");
  if (processor == nullptr) {
    return env->NewObjectArray(0, string_class, nullptr);
  }
  std::string input = JStringToStdString(env, text);
  std::vector<std::string> pieces;
  sentencepiece::absl::Status status = processor->Encode(input, &pieces);
  if (!status.ok()) {
    return env->NewObjectArray(0, string_class, nullptr);
  }
  jobjectArray result =
      env->NewObjectArray(static_cast<jsize>(pieces.size()), string_class,
                          nullptr);
  for (size_t i = 0; i < pieces.size(); ++i) {
    env->SetObjectArrayElement(result, static_cast<jsize>(i),
                               StdStringToJString(env, pieces[i]));
  }
  return result;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeEncodeAsIds(
    JNIEnv* env, jobject thiz, jlong native_ptr, jstring text) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  if (processor == nullptr) return nullptr;
  std::string input = JStringToStdString(env, text);
  std::vector<int> ids;
  sentencepiece::absl::Status status = processor->Encode(input, &ids);
  if (!status.ok()) return nullptr;
  jintArray result =
      env->NewIntArray(static_cast<jsize>(ids.size()));
  env->SetIntArrayRegion(result, 0, static_cast<jsize>(ids.size()),
                         ids.data());
  return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeDecodePieces(
    JNIEnv* env, jobject thiz, jlong native_ptr, jobjectArray pieces) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  if (processor == nullptr) return nullptr;
  jsize len = env->GetArrayLength(pieces);
  std::vector<std::string> piece_vec;
  piece_vec.reserve(static_cast<size_t>(len));
  for (jsize i = 0; i < len; ++i) {
    jstring piece = static_cast<jstring>(
        env->GetObjectArrayElement(pieces, i));
    piece_vec.push_back(JStringToStdString(env, piece));
    env->DeleteLocalRef(piece);
  }
  std::string detokenized;
  sentencepiece::absl::Status status =
      processor->Decode(sentencepiece::absl::MakeSpan(piece_vec),
                        &detokenized);
  if (!status.ok()) return nullptr;
  return StdStringToJString(env, detokenized);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeDecodeIds(
    JNIEnv* env, jobject thiz, jlong native_ptr, jintArray ids) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  if (processor == nullptr) return nullptr;
  jsize len = env->GetArrayLength(ids);
  jint* ids_ptr = env->GetIntArrayElements(ids, nullptr);
  std::vector<int> id_vec(ids_ptr, ids_ptr + len);
  env->ReleaseIntArrayElements(ids, ids_ptr, JNI_ABORT);
  std::string detokenized;
  sentencepiece::absl::Status status =
      processor->Decode(sentencepiece::absl::MakeSpan(id_vec),
                        &detokenized);
  if (!status.ok()) return nullptr;
  return StdStringToJString(env, detokenized);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeGetPieceSize(
    JNIEnv* env, jobject thiz, jlong native_ptr) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  if (processor == nullptr) return 0;
  return processor->GetPieceSize();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativePieceToId(
    JNIEnv* env, jobject thiz, jlong native_ptr, jstring piece) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  if (processor == nullptr) return -1;
  std::string p = JStringToStdString(env, piece);
  return processor->PieceToId(p);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeIdToPiece(
    JNIEnv* env, jobject thiz, jlong native_ptr, jint id) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  if (processor == nullptr) return nullptr;
  const std::string& piece = processor->IdToPiece(id);
  return StdStringToJString(env, piece);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeUnkId(
    JNIEnv* env, jobject thiz, jlong native_ptr) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  if (processor == nullptr) return -1;
  return processor->unk_id();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeBosId(
    JNIEnv* env, jobject thiz, jlong native_ptr) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  if (processor == nullptr) return -1;
  return processor->bos_id();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeEosId(
    JNIEnv* env, jobject thiz, jlong native_ptr) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  if (processor == nullptr) return -1;
  return processor->eos_id();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativePadId(
    JNIEnv* env, jobject thiz, jlong native_ptr) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  if (processor == nullptr) return -1;
  return processor->pad_id();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeNormalize(
    JNIEnv* env, jobject thiz, jlong native_ptr, jstring text) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  if (processor == nullptr) return nullptr;
  std::string input = JStringToStdString(env, text);
  std::string normalized;
  sentencepiece::absl::Status status = processor->Normalize(input, &normalized);
  if (!status.ok()) return nullptr;
  return StdStringToJString(env, normalized);
}

extern "C" JNIEXPORT void JNICALL
Java_com_google_sentencepiece_SentencePieceProcessor_nativeClose(
    JNIEnv* env, jobject thiz, jlong native_ptr) {
  auto* processor = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(
      native_ptr);
  delete processor;
}
