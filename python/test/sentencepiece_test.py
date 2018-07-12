#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.!

import sentencepiece as spm
import unittest
import sys

class TestSentencepieceProcessor(unittest.TestCase):
  """Test case for SentencePieceProcessor"""

  def setUp(self):
    self.sp_ = spm.SentencePieceProcessor()
    self.assertTrue(self.sp_.Load('test/test_model.model'))
    self.jasp_ = spm.SentencePieceProcessor()
    self.assertTrue(self.jasp_.Load('test/test_ja_model.model'))
    self.assertTrue(self.sp_.LoadFromSerializedProto(
        open('test/test_model.model', 'rb').read()))
    self.jasp_ = spm.SentencePieceProcessor()
    self.assertTrue(self.jasp_.LoadFromSerializedProto(
        open('test/test_ja_model.model', 'rb').read()))


  def test_load(self):
    self.assertEqual(1000, self.sp_.GetPieceSize())
    self.assertEqual(0, self.sp_.PieceToId('<unk>'))
    self.assertEqual(1, self.sp_.PieceToId('<s>'))
    self.assertEqual(2, self.sp_.PieceToId('</s>'))
    self.assertEqual('<unk>', self.sp_.IdToPiece(0))
    self.assertEqual('<s>', self.sp_.IdToPiece(1))
    self.assertEqual('</s>', self.sp_.IdToPiece(2))
    self.assertEqual(0, self.sp_.unk_id())
    self.assertEqual(1, self.sp_.bos_id())
    self.assertEqual(2, self.sp_.eos_id())
    self.assertEqual(-1, self.sp_.pad_id())
    for i in range(self.sp_.GetPieceSize()):
      piece = self.sp_.IdToPiece(i)
      self.assertEqual(i, self.sp_.PieceToId(piece))

  def test_roundtrip(self):
    text = 'I saw a girl with a telescope.'
    ids = self.sp_.EncodeAsIds(text)
    pieces1 = self.sp_.EncodeAsPieces(text)
    pieces2 = self.sp_.NBestEncodeAsPieces(text, 10)[0]
    self.assertEqual(pieces1, pieces2)
    self.assertEqual(text, self.sp_.DecodePieces(pieces1))
    self.assertEqual(text, self.sp_.DecodeIds(ids))
    for n in range(100):
      self.assertEqual(text, self.sp_.DecodePieces(self.sp_.SampleEncodeAsPieces(text, 64, 0.5)))
      self.assertEqual(text, self.sp_.DecodePieces(self.sp_.SampleEncodeAsPieces(text, -1, 0.5)))
      self.assertEqual(text, self.sp_.DecodeIds(self.sp_.SampleEncodeAsIds(text, 64, 0.5)))
      self.assertEqual(text, self.sp_.DecodeIds(self.sp_.SampleEncodeAsIds(text, -1, 0.5)))

  def test_ja_load(self):
    self.assertEqual(8000, self.jasp_.GetPieceSize())
    self.assertEqual(0, self.jasp_.PieceToId('<unk>'))
    self.assertEqual(1, self.jasp_.PieceToId('<s>'))
    self.assertEqual(2, self.jasp_.PieceToId('</s>'))
    self.assertEqual('<unk>', self.jasp_.IdToPiece(0))
    self.assertEqual('<s>', self.jasp_.IdToPiece(1))
    self.assertEqual('</s>', self.jasp_.IdToPiece(2))
    for i in range(self.jasp_.GetPieceSize()):
      piece = self.jasp_.IdToPiece(i)
      self.assertEqual(i, self.jasp_.PieceToId(piece))

  def test_ja_roundtrip(self):
    text = '清水寺は京都にある。'
    ids = self.jasp_.EncodeAsIds(text)
    pieces1 = self.jasp_.EncodeAsPieces(text)
    pieces2 = self.jasp_.NBestEncodeAsPieces(text, 10)[0]
    self.assertEqual(pieces1, pieces2)
    self.assertEqual(text, self.jasp_.DecodePieces(pieces1))
    self.assertEqual(text, self.jasp_.DecodeIds(ids))
    for n in range(100):
      self.assertEqual(text, self.sp_.DecodePieces(self.sp_.SampleEncodeAsPieces(text, 64, 0.5)))
      self.assertEqual(text, self.sp_.DecodePieces(self.sp_.SampleEncodeAsPieces(text, -1, 0.5)))


  def test_unicode_roundtrip(self):
    text = u'I saw a girl with a telescope.'
    ids = self.sp_.EncodeAsIds(text)
    pieces = self.sp_.EncodeAsPieces(text)
    self.assertEqual(text, self.sp_.DecodePieces(pieces))
    self.assertEqual(text, self.sp_.DecodeIds(ids))
    # python2 returns `str`.
    if sys.version_info < (3,0,0):
      text = text.encode('utf-8')
      self.assertEqual(text, self.sp_.DecodeIds(ids))
      self.assertEqual(text, self.sp_.DecodePieces(pieces))

  def test_unicode_ja_roundtrip(self):
    text = u'清水寺は京都にある。'
    ids = self.jasp_.EncodeAsIds(text)
    pieces = self.jasp_.EncodeAsPieces(text)
    self.assertEqual(text, self.jasp_.DecodePieces(pieces))
    # python2 returns `str`.
    if sys.version_info < (3,0,0):
      text = text.encode('utf-8')
      self.assertEqual(text, self.jasp_.DecodeIds(ids))

  def test_train(self):
    spm.SentencePieceTrainer.Train(
        "--input=test/botchan.txt --model_prefix=m --vocab_size=1000")
    sp = spm.SentencePieceProcessor()
    sp.Load('m.model')
    with open("test/botchan.txt") as file:
      for line in file:
        sp.DecodePieces(sp.EncodeAsPieces(line))
        sp.DecodeIds(sp.EncodeAsIds(line))

  # snake case API.
  def test_load_snake(self):
    self.assertEqual(1000, self.sp_.get_piece_size())
    self.assertEqual(0, self.sp_.piece_to_id('<unk>'))
    self.assertEqual(1, self.sp_.piece_to_id('<s>'))
    self.assertEqual(2, self.sp_.piece_to_id('</s>'))
    self.assertEqual('<unk>', self.sp_.id_to_piece(0))
    self.assertEqual('<s>', self.sp_.id_to_piece(1))
    self.assertEqual('</s>', self.sp_.id_to_piece(2))
    for i in range(self.sp_.get_piece_size()):
      piece = self.sp_.id_to_piece(i)
      self.assertEqual(i, self.sp_.piece_to_id(piece))

  def test_roundtrip_snake(self):
    text = 'I saw a girl with a telescope.'
    ids = self.sp_.encode_as_ids(text)
    pieces1 = self.sp_.encode_as_pieces(text)
    pieces2 = self.sp_.nbest_encode_as_pieces(text, 10)[0]
    self.assertEqual(pieces1, pieces2)
    self.assertEqual(text, self.sp_.decode_pieces(pieces1))
    self.assertEqual(text, self.sp_.decode_ids(ids))
    for n in range(100):
      self.assertEqual(text, self.sp_.decode_pieces(self.sp_.sample_encode_as_pieces(text, 64, 0.5)))
      self.assertEqual(text, self.sp_.decode_pieces(self.sp_.sample_encode_as_pieces(text, -1, 0.5)))
      self.assertEqual(text, self.sp_.decode_ids(self.sp_.sample_encode_as_ids(text, 64, 0.5)))
      self.assertEqual(text, self.sp_.decode_ids(self.sp_.sample_encode_as_ids(text, -1, 0.5)))

  def test_ja_load_snake(self):
    self.assertEqual(8000, self.jasp_.get_piece_size())
    self.assertEqual(0, self.jasp_.piece_to_id('<unk>'))
    self.assertEqual(1, self.jasp_.piece_to_id('<s>'))
    self.assertEqual(2, self.jasp_.piece_to_id('</s>'))
    self.assertEqual('<unk>', self.jasp_.id_to_piece(0))
    self.assertEqual('<s>', self.jasp_.id_to_piece(1))
    self.assertEqual('</s>', self.jasp_.id_to_piece(2))
    for i in range(self.jasp_.get_piece_size()):
      piece = self.jasp_.id_to_piece(i)
      self.assertEqual(i, self.jasp_.piece_to_id(piece))

  def test_ja_roundtrip_snake(self):
    text = '清水寺は京都にある。'
    ids = self.jasp_.encode_as_ids(text)
    pieces1 = self.jasp_.encode_as_pieces(text)
    pieces2 = self.jasp_.nbest_encode_as_pieces(text, 10)[0]
    self.assertEqual(pieces1, pieces2)
    self.assertEqual(text, self.jasp_.decode_pieces(pieces1))
    self.assertEqual(text, self.jasp_.decode_ids(ids))
    for n in range(100):
      self.assertEqual(text, self.sp_.decode_pieces(self.sp_.sample_encode_as_pieces(text, 64, 0.5)))
      self.assertEqual(text, self.sp_.decode_pieces(self.sp_.sample_encode_as_pieces(text, -1, 0.5)))

  def test_unicode_roundtrip_snake(self):
    text = u'I saw a girl with a telescope.'
    ids = self.sp_.encode_as_ids(text)
    pieces = self.sp_.encode_as_pieces(text)
    self.assertEqual(text, self.sp_.decode_pieces(pieces))
    # python2 returns `str`.
    if sys.version_info < (3,0,0):
      text = text.encode('utf-8')
      self.assertEqual(text, self.sp_.decode_ids(ids))

  def test_unicode_ja_roundtrip_snake(self):
    text = u'清水寺は京都にある。'
    ids = self.jasp_.encode_as_ids(text)
    pieces = self.jasp_.encode_as_pieces(text)
    self.assertEqual(text, self.jasp_.decode_pieces(pieces))
    # python2 returns `str`.
    if sys.version_info < (3,0,0):
      text = text.encode('utf-8')
      self.assertEqual(text, self.jasp_.decode_ids(ids))

  def test_train_snake(self):
    spm.SentencePieceTrainer.train(
        "--input=test/botchan.txt --model_prefix=m --vocab_size=1000")
    sp = spm.SentencePieceProcessor()
    sp.load('m.model')
    with open("test/botchan.txt") as file:
      for line in file:
        sp.decode_pieces(sp.encode_as_pieces(line))
        sp.decode_ids(sp.encode_as_ids(line))


def suite():
  suite = unittest.TestSuite()
  suite.addTests(unittest.makeSuite(TestSentencepieceProcessor))
  return suite


if __name__ == '__main__':
  unittest.main()
