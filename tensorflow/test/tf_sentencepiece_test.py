#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools as it
import os
import unittest
import tensorflow as tf
import sentencepiece as spm
import tf_sentencepiece as tfspm

class SentencePieceProcssorOpTest(unittest.TestCase):

  def _getSentencePieceModelFile(self):
    return '../python/test/test_ja_model.model'

  def _getExpected(self, processor, reverse=False, add_bos=False,
                   add_eos=False, padding=''):
    options = []
    if reverse:
      options.append('reverse')
    if add_bos:
      options.append('bos')
    if add_eos:
      options.append('eos')

    processor.SetEncodeExtraOptions(':'.join(options))
    processor.SetDecodeExtraOptions(':'.join(options))

    sentences = ['Hello world.', 'I have a pen.',
                 'I saw a girl with a telescope.']
    pieces = []
    ids = []
    seq_len = []

    for s in sentences:
      x = processor.EncodeAsPieces(s)
      y = processor.EncodeAsIds(s)
      pieces.append(x)
      ids.append(y)
      seq_len.append(len(x))
      self.assertEqual(len(x), len(y))

    # padding
    max_len = max(seq_len)
    pieces = [x + [padding] * (max_len - len(x)) for x in pieces]
    ids = [x + [0] * (max_len - len(x)) for x in ids]

    return sentences, pieces, ids, seq_len

  def testGetPieceSize(self):
    sentencepiece_model_file = self._getSentencePieceModelFile()
    processor = spm.SentencePieceProcessor()
    processor.Load(sentencepiece_model_file)

    with tf.Session():
      s = tfspm.piece_size(
          model_file=sentencepiece_model_file)
      self.assertEqual(s.eval(), processor.GetPieceSize())

  def testConvertPiece(self):
    sentencepiece_model_file = self._getSentencePieceModelFile()
    processor = spm.SentencePieceProcessor()
    processor.Load(sentencepiece_model_file)
    (sentences, expected_pieces,
     expected_ids, expected_seq_len) = self._getExpected(processor,
                                                         padding='<unk>')

    with tf.Session():
      ids_matrix = tfspm.piece_to_id(
          tf.constant(expected_pieces),
          model_file=sentencepiece_model_file)
      ids_vec = tfspm.piece_to_id(
          tf.constant(expected_pieces[0]),
          model_file=sentencepiece_model_file)
      ids_scalar = tfspm.piece_to_id(
          tf.constant(expected_pieces[0][0]),
          model_file=sentencepiece_model_file)

      self.assertEqual(ids_matrix.eval().tolist(), expected_ids)
      self.assertEqual(ids_vec.eval().tolist(), expected_ids[0])
      self.assertEqual(ids_scalar.eval(), expected_ids[0][0])

      pieces_matrix = tfspm.id_to_piece(
          tf.constant(expected_ids),
          model_file=sentencepiece_model_file)
      pieces_vec = tfspm.id_to_piece(
          tf.constant(expected_ids[0]),
          model_file=sentencepiece_model_file)
      pieces_scalar = tfspm.id_to_piece(
          tf.constant(expected_ids[0][0]),
          model_file=sentencepiece_model_file)

      self.assertEqual(pieces_matrix.eval().tolist(), expected_pieces)
      self.assertEqual(pieces_vec.eval().tolist(), expected_pieces[0])
      self.assertEqual(pieces_scalar.eval(), expected_pieces[0][0])


  def testEncodeAndDecode(self):
    sentencepiece_model_file = self._getSentencePieceModelFile()
    processor = spm.SentencePieceProcessor()
    processor.Load(sentencepiece_model_file)

    with tf.Session():
      for reverse, add_bos, add_eos in list(it.product(
          (True, False), repeat=3)):
        (sentences, expected_pieces,
         expected_ids, expected_seq_len) = self._getExpected(
             processor, reverse, add_bos, add_eos)

        # Encode sentences into pieces/ids.
        s = tf.constant(sentences)
        pieces, seq_len1 = tfspm.encode(
            s, model_file=sentencepiece_model_file,
            reverse=reverse, add_bos=add_bos, add_eos=add_eos,
            out_type=tf.string)
        ids, seq_len2 = tfspm.encode(
            s, model_file=sentencepiece_model_file,
            reverse=reverse, add_bos=add_bos, add_eos=add_eos)

        self.assertEqual(pieces.eval().tolist(), expected_pieces)
        self.assertEqual(ids.eval().tolist(), expected_ids)
        self.assertEqual(seq_len1.eval().tolist(), expected_seq_len)
        self.assertEqual(seq_len2.eval().tolist(), expected_seq_len)

        # Decode pieces into sentences/ids.
        pieces = tf.constant(expected_pieces)
        ids = tf.constant(expected_ids)
        seq_len = tf.constant(expected_seq_len, dtype=tf.int32)
        decoded_sentences1 = tfspm.decode(
            pieces, seq_len, model_file=sentencepiece_model_file,
            reverse=reverse)
        decoded_sentences2 = tfspm.decode(
            ids, seq_len, model_file=sentencepiece_model_file,
            reverse=reverse)

        self.assertEqual(decoded_sentences1.eval().tolist(), sentences)
        self.assertEqual(decoded_sentences2.eval().tolist(), sentences)

  def testSampleEncodeAndDecode(self):
    sentencepiece_model_file = self._getSentencePieceModelFile()
    processor = spm.SentencePieceProcessor()
    processor.Load(sentencepiece_model_file)
    sentences, _, _, _ = self._getExpected(processor)

    with tf.Session():
      for n, a in [(-1, 0.1), (64, 0.1), (0, 0.0)]:
        # Round trip test.
        nbest_size = tf.constant(n)
        alpha = tf.constant(a)
        s = tf.constant(sentences)

        pieces, seq_len1 = tfspm.encode(
            s, nbest_size=nbest_size, alpha=alpha,
            model_file=sentencepiece_model_file, out_type=tf.string)
        ids, seq_len2 = tfspm.encode(
            s, nbest_size=nbest_size, alpha=alpha,
            model_file=sentencepiece_model_file)
        decoded_sentences1 = tfspm.decode(
            pieces, seq_len1, model_file=sentencepiece_model_file)
        decoded_sentences2 = tfspm.decode(
            ids, seq_len2, model_file=sentencepiece_model_file)

        self.assertEqual(decoded_sentences1.eval().tolist(), sentences)
        self.assertEqual(decoded_sentences2.eval().tolist(), sentences)

  def testEncodeAndDecodeSparse(self):
    sentencepiece_model_file = self._getSentencePieceModelFile()
    processor = spm.SentencePieceProcessor()
    processor.Load(sentencepiece_model_file)

    with tf.Session():
      for reverse, add_bos, add_eos in list(it.product(
          (True, False), repeat=3)):
        (sentences, expected_pieces, expected_ids,
         _) = self._getExpected(processor, reverse, add_bos, add_eos)

        # Encode sentences into sparse pieces/ids.
        s = tf.constant(sentences)
        pieces = tfspm.encode_sparse(
            s, model_file=sentencepiece_model_file,
            reverse=reverse, add_bos=add_bos, add_eos=add_eos,
            out_type=tf.string)
        ids = tfspm.encode_sparse(
            s, model_file=sentencepiece_model_file,
            reverse=reverse, add_bos=add_bos, add_eos=add_eos)
        pieces = tf.sparse_tensor_to_dense(pieces, default_value='')
        ids = tf.sparse_tensor_to_dense(ids, default_value=0)

        self.assertEqual(ids.eval().tolist(), expected_ids)
        self.assertEqual(pieces.eval().tolist(), expected_pieces)

  def testGetPieceType(self):
    sentencepiece_model_file = self._getSentencePieceModelFile()
    processor = spm.SentencePieceProcessor()
    processor.Load(sentencepiece_model_file)
    expected_is_unknown = []
    expected_is_control = []
    expected_is_unused = []
    ids = []

    for i in range(processor.GetPieceSize()):
      ids.append(i)
      expected_is_unknown.append(processor.IsUnknown(i))
      expected_is_control.append(processor.IsControl(i))
      expected_is_unused.append(processor.IsUnused(i))

    with tf.Session():
      s = tf.constant(ids)
      is_unknown = tfspm.is_unknown(s, model_file=sentencepiece_model_file)
      is_control = tfspm.is_control(s, model_file=sentencepiece_model_file)
      is_unused = tfspm.is_unused(s, model_file=sentencepiece_model_file)

      self.assertEqual(is_unknown.eval().tolist(), expected_is_unknown)
      self.assertEqual(is_control.eval().tolist(), expected_is_control)
      self.assertEqual(is_unused.eval().tolist(), expected_is_unused)


  def testLoadModelProto(self):
    # Makes a serialized model proto.
    model_proto = open(self._getSentencePieceModelFile(), 'rb').read()
    with tf.Session() as sess:
      sentences = ['Hello world.']
      a = tf.constant(sentences)
      sess.run(tfspm.encode(
          a, model_proto=model_proto,
          out_type=tf.string))

  def testInvalidModelPath(self):
    with tf.Session() as sess:
      with self.assertRaises(tf.errors.NotFoundError):
        sentences = ['Hello world.']
        a = tf.constant(sentences)
        sess.run(tfspm.encode(
            a, model_file='invalid path', out_type=tf.string))

  def testInvalidModelProto(self):
    with tf.Session() as sess:
      with self.assertRaises(tf.errors.InternalError):
        sentences = ['Hello world.']
        a = tf.constant(sentences)
        sess.run(tfspm.encode(
            a, model_proto='invalid proto', out_type=tf.string))

  def testInvalidInput(self):
    sentences = ['Hello world.', 'This is a test.']
    ids = [[0,1],[2,3]]
    model_file = self._getSentencePieceModelFile()
    with tf.Session() as sess:
      a = tf.constant(sentences)
      b = tf.constant(ids)

      alpha = tf.constant([1.0, 2.0])
      sess.run(tfspm.encode(
          a, model_file=model_file, alpha=alpha, name='foo'))

      nbest_size = tf.constant([1, 2], dtype=tf.int32)
      sess.run(tfspm.encode(
          a, model_file=model_file, nbest_size=nbest_size, name='foo'))

      alpha = tf.constant(1.0)
      sess.run(tfspm.encode(
          a, model_file=model_file, alpha=alpha, name='foo'))

      nbest_size = tf.constant(10, dtype=tf.int32)
      sess.run(tfspm.encode(
          a, model_file=model_file, nbest_size=nbest_size, name='foo'))

      sess.run(tfspm.decode(
          b, sequence_length=tf.constant([2, 2]), model_file=model_file))

      with self.assertRaises(ValueError):
        a = tf.constant(sentences)
        alpha = tf.constant([1.0, 2.0, 3.0])
        sess.run(tfspm.encode(
            a, model_file=model_file, alpha=alpha))
      with self.assertRaises(ValueError):
        a = tf.constant(sentences)
        nbest_size = tf.constant([1, 2, 3], dtype=tf.int32)
        sess.run(tfspm.encode(
            a, model_file=model_file, nbest_size=nbest_size))
      with self.assertRaises(ValueError):
        a = tf.constant(sentences)
        alpha = tf.constant([[1.0], [2.0]])
        sess.run(tfspm.encode(
            a, model_file=model_file, alpha=alpha))
      with self.assertRaises(ValueError):
        a = tf.constant(sentences)
        nbest_size = tf.constant([[1], [2]], dtype=tf.int32)
        sess.run(tfspm.encode(
            a, model_file=model_file, nbest_size=nbest_size))
      with self.assertRaises(ValueError):
        b = tf.constant(ids)
        sess.run(tfspm.decode(
            a, sequence_length=2, model_file=model_file))
      with self.assertRaises(ValueError):
        b = tf.constant(ids)
        sess.run(tfspm.decode(
            a, sequence_length=tf.constant([2, 2, 2]),
            model_file=model_file))


def suite():
  suite = unittest.TestSuite()
  suite.addTests(unittest.makeSuite(SentencePieceProcssorOpTest))
  return suite


if __name__ == '__main__':
  unittest.main()
