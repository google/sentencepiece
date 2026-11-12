package com.google.sentencepiece

import java.io.Closeable
import java.io.File
import java.io.InputStream

/**
 * SentencePieceProcessor is a Kotlin wrapper around the native C++
 * [SentencePieceProcessor](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_processor.h).
 *
 * It provides tokenization (encode) and de-tokenization (decode) using a
 * pre-trained SentencePiece model on Android.
 *
 * Example:
 *
 * ```kotlin
 * val processor = SentencePieceProcessor.loadFromAssets(assets, "spm.model")
 * val pieces = processor.encodeAsPieces("hello world")
 * val ids = processor.encodeAsIds("hello world")
 * val text = processor.decodeIds(ids)
 * processor.close()
 * ```
 */
class SentencePieceProcessor internal constructor(private val nativePtr: Long) : Closeable {

    private var closed = false

    companion object {
        /**
         * Loads a SentencePiece model from a file path on the device.
         *
         * @param modelPath absolute path to a `.model` file.
         * @return a new [SentencePieceProcessor] instance.
         * @throws RuntimeException if the model cannot be loaded.
         */
        fun loadFromFile(modelPath: String): SentencePieceProcessor {
            val ptr = nativeCreate()
            val status = nativeLoad(ptr, modelPath)
            if (status != 0) {
                nativeClose(ptr)
                throw RuntimeException("Failed to load SentencePiece model from $modelPath")
            }
            return SentencePieceProcessor(ptr)
        }

        /**
         * Loads a SentencePiece model from the Android `assets` directory.
         *
         * @param assetManager the [android.content.res.AssetManager].
         * @param assetName name of the `.model` file inside `src/main/assets`.
         * @return a new [SentencePieceProcessor] instance.
         * @throws RuntimeException if the asset cannot be read or the model cannot be loaded.
         */
        fun loadFromAssets(assetManager: android.content.res.AssetManager, assetName: String): SentencePieceProcessor {
            val bytes = assetManager.open(assetName).use { it.readBytes() }
            val ptr = nativeCreate()
            val status = nativeLoadFromBytes(ptr, bytes)
            if (status != 0) {
                nativeClose(ptr)
                throw RuntimeException("Failed to load SentencePiece model from assets/$assetName")
            }
            return SentencePieceProcessor(ptr)
        }

        /**
         * Loads a SentencePiece model from a raw byte array (e.g. from a serialized proto).
         *
         * @param modelBytes the serialized model proto bytes.
         * @return a new [SentencePieceProcessor] instance.
         * @throws RuntimeException if the model cannot be loaded.
         */
        fun loadFromBytes(modelBytes: ByteArray): SentencePieceProcessor {
            val ptr = nativeCreate()
            val status = nativeLoadFromBytes(ptr, modelBytes)
            if (status != 0) {
                nativeClose(ptr)
                throw RuntimeException("Failed to load SentencePiece model from bytes")
            }
            return SentencePieceProcessor(ptr)
        }
    }

    /**
     * Encodes a string into a list of sentence pieces (strings).
     */
    fun encodeAsPieces(text: String): List<String> {
        checkNotClosed()
        return nativeEncodeAsPieces(nativePtr, text)
    }

    /**
     * Encodes a string into a list of vocabulary ids (ints).
     */
    fun encodeAsIds(text: String): List<Int> {
        checkNotClosed()
        return nativeEncodeAsIds(nativePtr, text)
    }

    /**
     * Decodes a list of sentence pieces into a de-tokenized string.
     */
    fun decodePieces(pieces: List<String>): String {
        checkNotClosed()
        return nativeDecodePieces(nativePtr, pieces.toTypedArray())
    }

    /**
     * Decodes a list of vocabulary ids into a de-tokenized string.
     */
    fun decodeIds(ids: List<Int>): String {
        checkNotClosed()
        return nativeDecodeIds(nativePtr, ids.toIntArray())
    }

    /**
     * Returns the size of the sentence piece vocabulary.
     */
    fun getPieceSize(): Int {
        checkNotClosed()
        return nativeGetPieceSize(nativePtr)
    }

    /**
     * Returns the vocabulary id of a given piece. Returns [unkId] if unknown.
     */
    fun pieceToId(piece: String): Int {
        checkNotClosed()
        return nativePieceToId(nativePtr, piece)
    }

    /**
     * Returns the string representation of the vocabulary entry for [id].
     */
    fun idToPiece(id: Int): String {
        checkNotClosed()
        return nativeIdToPiece(nativePtr, id)
    }

    /** Returns the unknown token id. */
    fun unkId(): Int {
        checkNotClosed()
        return nativeUnkId(nativePtr)
    }

    /** Returns the beginning-of-sentence token id, or -1 if undefined. */
    fun bosId(): Int {
        checkNotClosed()
        return nativeBosId(nativePtr)
    }

    /** Returns the end-of-sentence token id, or -1 if undefined. */
    fun eosId(): Int {
        checkNotClosed()
        return nativeEosId(nativePtr)
    }

    /** Returns the padding token id, or -1 if undefined. */
    fun padId(): Int {
        checkNotClosed()
        return nativePadId(nativePtr)
    }

    /**
     * Normalizes a string using the model's normalizer.
     */
    fun normalize(text: String): String {
        checkNotClosed()
        return nativeNormalize(nativePtr, text)
    }

    override fun close() {
        if (!closed) {
            nativeClose(nativePtr)
            closed = true
        }
    }

    private fun checkNotClosed() {
        if (closed) throw IllegalStateException("SentencePieceProcessor has been closed")
    }

    // JNI declarations
    private external fun nativeCreate(): Long
    private external fun nativeLoad(nativePtr: Long, modelPath: String): Int
    private external fun nativeLoadFromBytes(nativePtr: Long, modelBytes: ByteArray): Int
    private external fun nativeEncodeAsPieces(nativePtr: Long, text: String): List<String>
    private external fun nativeEncodeAsIds(nativePtr: Long, text: String): List<Int>
    private external fun nativeDecodePieces(nativePtr: Long, pieces: Array<String>): String
    private external fun nativeDecodeIds(nativePtr: Long, ids: IntArray): String
    private external fun nativeGetPieceSize(nativePtr: Long): Int
    private external fun nativePieceToId(nativePtr: Long, piece: String): Int
    private external fun nativeIdToPiece(nativePtr: Long, id: Int): String
    private external fun nativeUnkId(nativePtr: Long): Int
    private external fun nativeBosId(nativePtr: Long): Int
    private external fun nativeEosId(nativePtr: Long): Int
    private external fun nativePadId(nativePtr: Long): Int
    private external fun nativeNormalize(nativePtr: Long, text: String): String
    private external fun nativeClose(nativePtr: Long)

    init {
        System.loadLibrary("sentencepiece_jni")
    }
}
