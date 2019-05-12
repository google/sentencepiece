package com.github.google.sentencepiece;

import com.google.protobuf.InvalidProtocolBufferException;
import sentencepiece.Sentencepiece;
import sentencepiece.SentencepieceModel;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class SentencePieceProcessor implements AutoCloseable {

    private final long rawPtr;

    public SentencePieceProcessor() {
        rawPtr = SentencePieceJNI.sppCtor();
    }

    @Override
    public void close() {
        SentencePieceJNI.sppDtor(rawPtr);
    }

    public void load(String filename) throws SentencePieceException {
        SentencePieceJNI.sppLoad(rawPtr, filename.getBytes(StandardCharsets.UTF_8));
    }

    public void loadOrDie(String filename) {
        SentencePieceJNI.sppLoadOrDie(rawPtr, filename.getBytes(StandardCharsets.UTF_8));
    }

    public void loadFromSerializedProto(byte[] serialized) throws SentencePieceException {
        SentencePieceJNI.sppLoadFromSerializedProto(rawPtr, serialized);
    }

    public void loadFromSerializedProto(SentencepieceModel.ModelProto serialized) throws SentencePieceException {
        loadFromSerializedProto(serialized.toByteArray());
    }

    public void setEncodeExtraOptions(String extraOption) throws SentencePieceException {
        SentencePieceJNI.sppSetEncodeExtraOptions(rawPtr, extraOption.getBytes(StandardCharsets.UTF_8));
    }

    public void setDecodeExtraOptions(String extraOption) throws SentencePieceException {
        SentencePieceJNI.sppSetDecodeExtraOptions(rawPtr, extraOption.getBytes(StandardCharsets.UTF_8));
    }

    public void setVocabulary(List<String> validVocab) throws SentencePieceException {
        byte[][] bytes = validVocab.stream()
                .map(str -> str.getBytes(StandardCharsets.UTF_8))
                .toArray(byte[][]::new);
        SentencePieceJNI.sppSetVocabulary(rawPtr, bytes);
    }

    public void resetVocabulary() throws SentencePieceException {
        SentencePieceJNI.sppResetVocabulary(rawPtr);
    }

    public void loadVocabulary(String filename, int threshold) throws SentencePieceException {
        SentencePieceJNI.sppLoadVocabulary(rawPtr, filename.getBytes(StandardCharsets.UTF_8), threshold);
    }

    public List<String> encodeAsPieces(String input) throws SentencePieceException {
        byte[][] pieces = SentencePieceJNI.sppEncodeAsPieces(rawPtr, input.getBytes(StandardCharsets.UTF_8));
        return Arrays.stream(pieces)
                .map(bytes -> new String(bytes, StandardCharsets.UTF_8))
                .collect(Collectors.toList());
    }

    public int[] encodeAsIds(String input) throws SentencePieceException {
        return SentencePieceJNI.sppEncodeAsIds(rawPtr, input.getBytes(StandardCharsets.UTF_8));
    }

    public String decodePieces(List<String> pieces) throws SentencePieceException {
        byte[][] bytes = pieces.stream()
                .map(str -> str.getBytes(StandardCharsets.UTF_8))
                .toArray(byte[][]::new);
        return new String(SentencePieceJNI.sppDecodePieces(rawPtr, bytes), StandardCharsets.UTF_8);
    }

    public String decodeIds(int... ids) throws SentencePieceException {
        return new String(SentencePieceJNI.sppDecodeIds(rawPtr, ids), StandardCharsets.UTF_8);
    }

    public List<List<String>> nbestEncodeAsPieces(String input, int nbestSize) throws SentencePieceException {
        byte[][][] pieces = SentencePieceJNI.sppNBestEncodeAsPieces(rawPtr, input.getBytes(StandardCharsets.UTF_8), nbestSize);
        return Arrays.stream(pieces)
                .map(bytes -> Arrays.stream(bytes)
                        .map(b -> new String(b, StandardCharsets.UTF_8))
                        .collect(Collectors.toList()))
                .collect(Collectors.toList());
    }

    public int[][] nbestEncodeAsIds(String input, int nbestSize) throws SentencePieceException {
        return SentencePieceJNI.sppNBestEncodeAsIds(rawPtr, input.getBytes(StandardCharsets.UTF_8), nbestSize);
    }

    public List<String> sampleEncodeAsPieces(String input, int nbestSize, float alpha) throws SentencePieceException {
        byte[][] pieces = SentencePieceJNI.sppSampleEncodeAsPieces(rawPtr, input.getBytes(StandardCharsets.UTF_8), nbestSize, alpha);
        return Arrays.stream(pieces)
                .map(bytes -> new String(bytes, StandardCharsets.UTF_8))
                .collect(Collectors.toList());
    }

    public int[] sampleEncodeAsIds(String input, int nbestSize, float alpha) throws SentencePieceException {
        return SentencePieceJNI.sppSampleEncodeAsIds(rawPtr, input.getBytes(StandardCharsets.UTF_8), nbestSize, alpha);
    }

    public Sentencepiece.SentencePieceText encodeAsSerializedProto(String input) throws InvalidProtocolBufferException {
        return Sentencepiece.SentencePieceText.parseFrom(
                SentencePieceJNI.sppEncodeAsSerializedProto(rawPtr, input.getBytes(StandardCharsets.UTF_8)));
    }

    public Sentencepiece.SentencePieceText sampleEncodeAsSerializedProto(String input, int nbestSize, float alpha) throws InvalidProtocolBufferException {
        return Sentencepiece.SentencePieceText.parseFrom(
                SentencePieceJNI.sppSampleEncodeAsSerializedProto(rawPtr, input.getBytes(StandardCharsets.UTF_8), nbestSize, alpha));
    }

    public Sentencepiece.NBestSentencePieceText nbestEncodeAsSerializedProto(String input, int nbestSize) throws InvalidProtocolBufferException {
        return Sentencepiece.NBestSentencePieceText.parseFrom(
                SentencePieceJNI.sppNBestEncodeAsSerializedProto(rawPtr, input.getBytes(StandardCharsets.UTF_8), nbestSize));
    }

    public Sentencepiece.SentencePieceText decodePiecesAsSerializedProto(List<String> pieces) throws InvalidProtocolBufferException {
        byte[][] bytes = pieces.stream()
                .map(str -> str.getBytes(StandardCharsets.UTF_8))
                .toArray(byte[][]::new);
        return Sentencepiece.SentencePieceText.parseFrom(
                SentencePieceJNI.sppDecodePiecesAsSerializedProto(rawPtr, bytes));
    }

    public Sentencepiece.SentencePieceText decodeIdsAsSerializedProto(int... ids) throws InvalidProtocolBufferException {
        return Sentencepiece.SentencePieceText.parseFrom(
                SentencePieceJNI.sppDecodeIdsAsSerializedProto(rawPtr, ids));
    }

    public int getPieceSize() {
        return SentencePieceJNI.sppGetPieceSize(rawPtr);
    }

    public int pieceToId(String piece) {
        return SentencePieceJNI.sppPieceToId(rawPtr, piece.getBytes(StandardCharsets.UTF_8));
    }

    public String idToPiece(int id) {
        return new String(SentencePieceJNI.sppIdToPiece(rawPtr, id), StandardCharsets.UTF_8);
    }

    public float getScore(int id) {
        return SentencePieceJNI.sppGetScore(rawPtr, id);
    }

    public boolean isUnknown(int id) {
        return SentencePieceJNI.sppIsUnknown(rawPtr, id);
    }

    public boolean isControl(int id) {
        return SentencePieceJNI.sppIsControl(rawPtr, id);
    }

    public boolean isUnused(int id) {
        return SentencePieceJNI.sppIsUnused(rawPtr, id);
    }

    public int unkId() {
        return SentencePieceJNI.sppUnkId(rawPtr);
    }

    public int bosId() {
        return SentencePieceJNI.sppBosId(rawPtr);
    }

    public int eosId() {
        return SentencePieceJNI.sppEosId(rawPtr);
    }

    public int padId() {
        return SentencePieceJNI.sppPadId(rawPtr);
    }
}
