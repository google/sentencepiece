#!/usr/bin/python
# -*- coding: utf-8 -*-

import itertools as it
import os
import sys
import unittest
import tensorflow as tf
import tf_sentencepiece as tfspm

class SentencePieceProcssorOpTest(unittest.TestCase):

  def _getSentencePieceModelFile(self):
    return os.path.join('..', 'python', 'test', 'test_model.model')

  def _getPieceSize(self):
    return 1000

  def _getExpected(self, reverse=False, add_bos=False,
                   add_eos=False, padding=''):
    # TF uses str(bytes) as a string representation.
    padding = padding.encode('utf8')
    sentences = [b'Hello world.', b'I have a pen.',
                 b'I saw a girl with a telescope.']
    pieces = [[b'\xe2\x96\x81He', b'll', b'o', b'\xe2\x96\x81world', b'.'],
              [b'\xe2\x96\x81I', b'\xe2\x96\x81have', b'\xe2\x96\x81a',
               b'\xe2\x96\x81p', b'en', b'.'],
              [b'\xe2\x96\x81I', b'\xe2\x96\x81saw', b'\xe2\x96\x81a',
               b'\xe2\x96\x81girl', b'\xe2\x96\x81with',
               b'\xe2\x96\x81a', b'\xe2\x96\x81',
               b'te', b'le', b's', b'c', b'o', b'pe', b'.']]
    ids = [[151, 88, 21, 887, 6],
           [9, 76, 11, 68, 98, 6],
           [9, 459, 11, 939, 44, 11, 4, 142, 82, 8, 28, 21, 132, 6]]
    seq_len = [5, 6, 14]

    if reverse:
      ids = [x[::-1] for x in ids]
      pieces = [x[::-1] for x in pieces]

    if add_bos:
      ids = [[1] + x for x in ids]
      pieces = [[b'<s>'] + x for x in pieces]
      seq_len = [x + 1 for x in seq_len]

    if add_eos:
      ids = [x + [2] for x in ids]
      pieces = [x + [b'</s>'] for x in pieces]
      seq_len = [x + 1 for x in seq_len]

    max_len = max(seq_len)
    pieces = [x + [padding] * (max_len - len(x)) for x in pieces]
    ids = [x + [0] * (max_len - len(x)) for x in ids]

    return sentences, pieces, ids, seq_len

  def testGetPieceSize(self):
    sentencepiece_model_file = self._getSentencePieceModelFile()

    with tf.Session():
      s = tfspm.piece_size(
          model_file=sentencepiece_model_file)
      self.assertEqual(s.eval(), self._getPieceSize())

  def testConvertPiece(self):
    sentencepiece_model_file = self._getSentencePieceModelFile()
    (sentences, expected_pieces,
     expected_ids, expected_seq_len) = self._getExpected(padding='<unk>')

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

    with tf.Session():
      for reverse, add_bos, add_eos in list(it.product(
          (True, False), repeat=3)):
        (sentences, expected_pieces,
         expected_ids, expected_seq_len) = self._getExpected(
             reverse=reverse, add_bos=add_bos, add_eos=add_eos)

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
    sentences, _, _, _ = self._getExpected()

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

    with tf.Session():
      for reverse, add_bos, add_eos in list(it.product(
          (True, False), repeat=3)):
        (sentences, expected_pieces, expected_ids,
         _) = self._getExpected(reverse, add_bos, add_eos)

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
    expected_is_unknown = []
    expected_is_control = []
    expected_is_unused = []
    ids = []

    for i in range(self._getPieceSize()):
      ids.append(i)
      expected_is_unknown.append(i == 0)
      expected_is_control.append(i == 1 or i == 2)
      expected_is_unused.append(False)

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
