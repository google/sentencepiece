# sentencepiece-android

Official Android / Kotlin bindings for [SentencePiece](https://github.com/google/sentencepiece).

This module provides an idiomatic Kotlin API wrapping the native C++
`SentencePieceProcessor` via JNI, enabling on-device tokenization and
de-tokenization in Android apps without shipping a Python interpreter.

## Contents

- `src/main/jni/sentencepiece_jni.cc` — JNI bridge to the C++ library.
- `src/main/kotlin/com/google/sentencepiece/SentencePieceProcessor.kt` — Kotlin API.
- `CMakeLists.txt` — NDK build that reuses the sentencepiece C++ target.
- `build.gradle.kts` — Android library module (AGP + Kotlin).

## Requirements

- Android NDK (r25+)
- CMake 3.22+
- Kotlin 1.9+
- Android Gradle Plugin 8.0+

## Usage

```kotlin
import com.google.sentencepiece.SentencePieceProcessor

// Load a model from assets
val processor = SentencePieceProcessor.loadFromAssets(assets, "spm.model")

// Encode
val pieces = processor.encodeAsPieces("hello world")      // List<String>
val ids = processor.encodeAsIds("hello world")            // List<Int>

// Decode
val text = processor.decodeIds(ids)                        // String

// Vocabulary info
val size = processor.getPieceSize()
val unk = processor.unkId()

processor.close()
```

## Building

From the sentencepiece repository root:

```bash
cd android
./gradlew assembleRelease
```

The produced AAR can be consumed by any Android application module.

## License

Apache 2.0 — see the [LICENSE](../LICENSE) file in the sentencepiece repository root.
