package com.github.google.sentencepiece;

import com.google.protobuf.InvalidProtocolBufferException;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import sentencepiece.Sentencepiece;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class SentencePieceProcessorTest {

    private SentencePieceProcessor sp;
    private SentencePieceProcessor jasp;

    @Before
    public void setUp() throws IOException {
        sp = new SentencePieceProcessor();
        jasp = new SentencePieceProcessor();
        sp.load("../python/test/test_model.model");
        jasp.load("../python/test//test_ja_model.model");
        sp.loadFromSerializedProto(Files.readAllBytes(Paths.get("../python/test/test_model.model")));
        jasp.loadFromSerializedProto(Files.readAllBytes(Paths.get("../python/test//test_ja_model.model")));
    }

    @After
    public void tearDown() {
        sp.close();
        jasp.close();
    }

    @Test
    public void testLoad() {
        Assert.assertEquals(1000, sp.getPieceSize());
        Assert.assertEquals(0, sp.pieceToId("<unk>"));
        Assert.assertEquals(1, sp.pieceToId("<s>"));
        Assert.assertEquals(2, sp.pieceToId("</s>"));
        Assert.assertEquals("<unk>", sp.idToPiece(0));
        Assert.assertEquals("<s>", sp.idToPiece(1));
        Assert.assertEquals("</s>", sp.idToPiece(2));
        Assert.assertEquals(0, sp.unkId());
        Assert.assertEquals(1, sp.bosId());
        Assert.assertEquals(2, sp.eosId());
        Assert.assertEquals(-1, sp.padId());
        for (int i = 0; i < sp.getPieceSize(); i++) {
            String piece = sp.idToPiece(i);
            Assert.assertEquals(i, sp.pieceToId(piece));
        }
    }

    @Test
    public void testRoundTrip() {
        String text = "I saw a girl with a telescope.";
        int[] ids = sp.encodeAsIds(text);
        List<String> pieces1 = sp.encodeAsPieces(text);
        List<String> pieces2 = sp.nbestEncodeAsPieces(text, 10).get(0);
        Assert.assertEquals(pieces1, pieces2);
        Assert.assertEquals(text, sp.decodePieces(pieces1));
        Assert.assertEquals(text, sp.decodeIds(ids));
        for (int i = 0; i < 100; i++) {
            Assert.assertEquals(text, sp.decodePieces(sp.sampleEncodeAsPieces(text, 64, 0.5f)));
            Assert.assertEquals(text, sp.decodePieces(sp.sampleEncodeAsPieces(text, -1, 0.5f)));
            Assert.assertEquals(text, sp.decodeIds(sp.sampleEncodeAsIds(text, 64, 0.5f)));
            Assert.assertEquals(text, sp.decodeIds(sp.sampleEncodeAsIds(text, -1, 0.5f)));
        }
    }

    @Test
    public void testJaLoad() {
        Assert.assertEquals(8000, jasp.getPieceSize());
        Assert.assertEquals(0, jasp.pieceToId("<unk>"));
        Assert.assertEquals(1, jasp.pieceToId("<s>"));
        Assert.assertEquals(2, jasp.pieceToId("</s>"));
        Assert.assertEquals("<unk>", jasp.idToPiece(0));
        Assert.assertEquals("<s>", jasp.idToPiece(1));
        Assert.assertEquals("</s>", jasp.idToPiece(2));
        for (int i = 0; i < jasp.getPieceSize(); i++) {
            String piece = jasp.idToPiece(i);
            Assert.assertEquals(i, jasp.pieceToId(piece));
        }
    }

    @Test
    public void testJaRoundTrip() {
        String text = "清水寺は京都にある。";
        int[] ids = jasp.encodeAsIds(text);
        List<String> pieces1 = jasp.encodeAsPieces(text);
        List<String> pieces2 = jasp.nbestEncodeAsPieces(text, 10).get(0);
        Assert.assertEquals(pieces1, pieces2);
        Assert.assertEquals(text, jasp.decodePieces(pieces1));
        Assert.assertEquals(text, jasp.decodeIds(ids));
        for (int i = 0; i < 100; i++) {
            Assert.assertEquals(text, jasp.decodePieces(jasp.sampleEncodeAsPieces(text, 64, 0.5f)));
            Assert.assertEquals(text, jasp.decodePieces(jasp.sampleEncodeAsPieces(text, -1, 0.5f)));
        }
    }

    @Test
    public void testSerializedProto() throws InvalidProtocolBufferException {
        Sentencepiece.SentencePieceText none = Sentencepiece.SentencePieceText.getDefaultInstance();
        Sentencepiece.NBestSentencePieceText noneNB = Sentencepiece.NBestSentencePieceText.getDefaultInstance();

        String text = "I saw a girl with a telescope.";
        Assert.assertNotEquals(none, sp.encodeAsSerializedProto(text));
        Assert.assertNotEquals(none, sp.sampleEncodeAsSerializedProto(text, 10, 0.2f));
        Assert.assertNotEquals(noneNB, sp.nbestEncodeAsSerializedProto(text, 10));
        Assert.assertNotEquals(none, sp.decodePiecesAsSerializedProto(Arrays.asList("foo", "bar")));
        Assert.assertNotEquals(none, sp.decodeIdsAsSerializedProto(20, 30));
    }
}
