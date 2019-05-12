package com.github.google.sentencepiece;

import java.io.IOException;

class SentencePieceJNI {

    static {
        try {
            System.load(NativeLibLoader.createTempFileFromResource("/" + System.mapLibraryName("sentencepiece_java")));
        } catch (IOException e) {
            throw new UnsatisfiedLinkError(e.getMessage());
        }
    }

    static native long sppCtor();
    static native void sppDtor(long spp);
    static native void sppLoad(long spp, byte[] filename) throws SentencePieceException;
    static native void sppLoadOrDie(long spp, byte[] filename);
    static native void sppLoadFromSerializedProto(long spp, byte[] serialized) throws SentencePieceException;
    static native void sppSetEncodeExtraOptions(long spp, byte[] extra_option) throws SentencePieceException;
    static native void sppSetDecodeExtraOptions(long spp, byte[] extra_option) throws SentencePieceException;

    static native void sppSetVocabulary(long spp, byte[][] valid_vocab) throws SentencePieceException;
    static native void sppResetVocabulary(long spp) throws SentencePieceException;
    static native void sppLoadVocabulary(long spp, byte[] filename, int threshold) throws SentencePieceException;

    static native byte[][] sppEncodeAsPieces(long spp, byte[] input) throws SentencePieceException;
    static native int[] sppEncodeAsIds(long spp, byte[] input) throws SentencePieceException;
    static native byte[] sppDecodePieces(long spp, byte[][] pieces) throws SentencePieceException;
    static native byte[] sppDecodeIds(long spp, int[] ids) throws SentencePieceException;

    static native byte[][][] sppNBestEncodeAsPieces(long spp, byte[] input, int nbest_size) throws SentencePieceException;
    static native int[][] sppNBestEncodeAsIds(long spp, byte[] input, int nbest_size) throws SentencePieceException;

    static native byte[][] sppSampleEncodeAsPieces(long spp, byte[] input, int nbest_size, float alpha) throws SentencePieceException;
    static native int[] sppSampleEncodeAsIds(long spp, byte[] input, int nbest_size, float alpha) throws SentencePieceException;

    static native byte[] sppEncodeAsSerializedProto(long spp, byte[] input);
    static native byte[] sppSampleEncodeAsSerializedProto(long spp, byte[] input, int nbest_size, float alpha);
    static native byte[] sppNBestEncodeAsSerializedProto(long spp, byte[] input, int nbest_size);
    static native byte[] sppDecodePiecesAsSerializedProto(long spp, byte[][] pieces);
    static native byte[] sppDecodeIdsAsSerializedProto(long spp, int[] ids);

    static native int sppGetPieceSize(long spp);
    static native int sppPieceToId(long spp, byte[] piece);
    static native byte[] sppIdToPiece(long spp, int id);
    static native float sppGetScore(long spp, int id);
    static native boolean sppIsUnknown(long spp, int id);
    static native boolean sppIsControl(long spp, int id);
    static native boolean sppIsUnused(long spp, int id);
    static native int sppUnkId(long spp);
    static native int sppBosId(long spp);
    static native int sppEosId(long spp);
    static native int sppPadId(long spp);
}
