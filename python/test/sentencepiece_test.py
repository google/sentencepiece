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

import codecs
import io
import sentencepiece as spm
import unittest
import sys
import os
import pickle

from collections import defaultdict

data_dir = 'test'
if sys.platform == 'win32':
  data_dir = os.path.join('..', 'data')


class TestSentencepieceProcessor(unittest.TestCase):
  """Test case for SentencePieceProcessor"""

  def setUp(self):
    self.sp_ = spm.SentencePieceProcessor()
    self.jasp_ = spm.SentencePieceProcessor()
    self.assertTrue(self.sp_.Load(os.path.join('test', 'test_model.model')))
    self.assertTrue(
        self.jasp_.Load(os.path.join('test', 'test_ja_model.model')))
    with open(os.path.join('test', 'test_model.model'), 'rb') as f:
      self.assertTrue(self.sp_.LoadFromSerializedProto(f.read()))
    with open(os.path.join('test', 'test_ja_model.model'), 'rb') as f:
      self.assertTrue(self.jasp_.LoadFromSerializedProto(f.read()))

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
      self.assertEqual(
          text,
          self.sp_.DecodePieces(self.sp_.SampleEncodeAsPieces(text, 64, 0.5)))
      self.assertEqual(
          text,
          self.sp_.DecodePieces(self.sp_.SampleEncodeAsPieces(text, -1, 0.5)))
      self.assertEqual(
          text, self.sp_.DecodeIds(self.sp_.SampleEncodeAsIds(text, 64, 0.5)))
      self.assertEqual(
          text, self.sp_.DecodeIds(self.sp_.SampleEncodeAsIds(text, -1, 0.5)))

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
      self.assertEqual(
          text,
          self.jasp_.DecodePieces(
              self.jasp_.SampleEncodeAsPieces(text, 64, 0.5)))
      self.assertEqual(
          text,
          self.jasp_.DecodePieces(
              self.jasp_.SampleEncodeAsPieces(text, -1, 0.5)))

  def test_unicode_roundtrip(self):
    text = u'I saw a girl with a telescope.'
    ids = self.sp_.EncodeAsIds(text)
    pieces = self.sp_.EncodeAsPieces(text)
    self.assertEqual(text, self.sp_.DecodePieces(pieces))
    self.assertEqual(text, self.sp_.DecodeIds(ids))
    # python2 returns `str`.
    if sys.version_info < (3, 0, 0):
      text = text.encode('utf-8')
      self.assertEqual(text, self.sp_.DecodeIds(ids))
      self.assertEqual(text, self.sp_.DecodePieces(pieces))

  def test_unicode_ja_roundtrip(self):
    text = u'清水寺は京都にある。'
    ids = self.jasp_.EncodeAsIds(text)
    pieces = self.jasp_.EncodeAsPieces(text)
    self.assertEqual(text, self.jasp_.DecodePieces(pieces))
    # python2 returns `str`.
    if sys.version_info < (3, 0, 0):
      text = text.encode('utf-8')
      self.assertEqual(text, self.jasp_.DecodeIds(ids))

  def test_pickle(self):
    with open('sp.pickle', 'wb') as f:
      pickle.dump(self.sp_, f)

    id1 = self.sp_.encode('hello world.', out_type=int)

    with open('sp.pickle', 'rb') as f:
      sp = pickle.load(f)

    id2 = sp.encode('hello world.', out_type=int)

    self.assertEqual(id1, id2)

  def test_train(self):
    spm.SentencePieceTrainer.Train('--input=' +
                                   os.path.join(data_dir, 'botchan.txt') +
                                   ' --model_prefix=m --vocab_size=1000')
    sp = spm.SentencePieceProcessor()
    sp.Load('m.model')
    with codecs.open(
        os.path.join(data_dir, 'botchan.txt'), 'r', encoding='utf-8') as file:
      for line in file:
        sp.DecodePieces(sp.EncodeAsPieces(line))
        sp.DecodeIds(sp.EncodeAsIds(line))

  def test_train(self):
    spm.SentencePieceTrainer.Train('--input=' +
                                   os.path.join(data_dir, 'botchan.txt') +
                                   ' --model_prefix=m --vocab_size=1000')
    # Load as 'rb' for Python3.5/2.7.
    is1 = open(os.path.join(data_dir, 'botchan.txt'), 'rb')
    is2 = open(os.path.join(data_dir, 'botchan.txt'), 'rb')
    os1 = io.BytesIO()
    os2 = io.BytesIO()

    spm.SentencePieceTrainer.train(
        input=os.path.join(data_dir, 'botchan.txt'),
        model_prefix='m',
        vocab_size=1000)

    spm.SentencePieceTrainer.train(
        sentence_iterator=is1, model_prefix='m', vocab_size=1000)

    spm.SentencePieceTrainer.train(
        input=os.path.join(data_dir, 'botchan.txt'),
        model_writer=os1,
        vocab_size=1000)

    spm.SentencePieceTrainer.train(
        sentence_iterator=is2, model_writer=os2, vocab_size=1000)

    sp1 = spm.SentencePieceProcessor(model_proto=os1.getvalue())
    sp2 = spm.SentencePieceProcessor(model_proto=os2.getvalue())
    self.assertEqual([sp1.id_to_piece(i) for i in range(sp1.get_piece_size())],
                     [sp2.id_to_piece(i) for i in range(sp2.get_piece_size())])

  def test_train_kwargs(self):
    spm.SentencePieceTrainer.train(
        input=[os.path.join(data_dir, 'botchan.txt')],
        model_prefix='m',
        vocab_size=1002,
        user_defined_symbols=['foo', 'bar', ','])
    sp = spm.SentencePieceProcessor()
    sp.Load('m.model')
    with codecs.open(
        os.path.join(data_dir, 'botchan.txt'), 'r', encoding='utf-8') as file:
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
      self.assertEqual(
          text,
          self.sp_.decode_pieces(
              self.sp_.sample_encode_as_pieces(text, 64, 0.5)))
      self.assertEqual(
          text,
          self.sp_.decode_pieces(
              self.sp_.sample_encode_as_pieces(text, -1, 0.5)))
      self.assertEqual(
          text,
          self.sp_.decode_ids(self.sp_.sample_encode_as_ids(text, 64, 0.5)))
      self.assertEqual(
          text,
          self.sp_.decode_ids(self.sp_.sample_encode_as_ids(text, -1, 0.5)))

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
      self.assertEqual(
          text,
          self.jasp_.decode_pieces(
              self.jasp_.sample_encode_as_pieces(text, 64, 0.5)))
      self.assertEqual(
          text,
          self.jasp_.decode_pieces(
              self.jasp_.sample_encode_as_pieces(text, -1, 0.5)))

  def test_unicode_roundtrip_snake(self):
    text = u'I saw a girl with a telescope.'
    ids = self.sp_.encode_as_ids(text)
    pieces = self.sp_.encode_as_pieces(text)
    self.assertEqual(text, self.sp_.decode_pieces(pieces))
    # python2 returns `str`.
    if sys.version_info < (3, 0, 0):
      text = text.encode('utf-8')
      self.assertEqual(text, self.sp_.decode_ids(ids))

  def test_unicode_ja_roundtrip_snake(self):
    text = u'清水寺は京都にある。'
    ids = self.jasp_.encode_as_ids(text)
    pieces = self.jasp_.encode_as_pieces(text)
    self.assertEqual(text, self.jasp_.decode_pieces(pieces))
    # python2 returns `str`.
    if sys.version_info < (3, 0, 0):
      text = text.encode('utf-8')
      self.assertEqual(text, self.jasp_.decode_ids(ids))

  def test_train_snake(self):
    spm.SentencePieceTrainer.train('--input=' +
                                   os.path.join(data_dir, 'botchan.txt') +
                                   ' --model_prefix=m --vocab_size=1000')
    sp = spm.SentencePieceProcessor()
    sp.load('m.model')
    with codecs.open(
        os.path.join(data_dir, 'botchan.txt'), 'r', encoding='utf-8') as file:
      for line in file:
        sp.decode_pieces(sp.encode_as_pieces(line))
        sp.decode_ids(sp.encode_as_ids(line))

  def test_serialized_proto(self):
    text = u'I saw a girl with a telescope.'
    self.assertNotEqual('', self.sp_.EncodeAsSerializedProto(text))
    self.assertNotEqual('',
                        self.sp_.SampleEncodeAsSerializedProto(text, 10, 0.2))
    self.assertNotEqual('', self.sp_.NBestEncodeAsSerializedProto(text, 10))
    self.assertNotEqual('',
                        self.sp_.DecodePiecesAsSerializedProto(['foo', 'bar']))
    self.assertNotEqual('', self.sp_.DecodeIdsAsSerializedProto([20, 30]))
    self.assertNotEqual('', self.sp_.encode_as_serialized_proto(text))
    self.assertNotEqual(
        '', self.sp_.sample_encode_as_serialized_proto(text, 10, 0.2))
    self.assertNotEqual('', self.sp_.nbest_encode_as_serialized_proto(text, 10))
    self.assertNotEqual(
        '', self.sp_.decode_pieces_as_serialized_proto(['foo', 'bar']))
    self.assertNotEqual('', self.sp_.decode_ids_as_serialized_proto([20, 30]))

  def test_new_api(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join('test', 'test_model.model'))
    text = 'hello world'
    text2 = 'Tokyo'
    ids = self.sp_.EncodeAsIds(text)
    ids2 = self.sp_.EncodeAsIds(text2)
    pieces = self.sp_.EncodeAsPieces(text)
    pieces2 = self.sp_.EncodeAsPieces(text2)
    self.assertEqual(sp.encode(text), ids)
    self.assertEqual(sp.encode(text, out_type=str), pieces)
    detok_ids = self.sp_.DecodeIds(ids)
    detok_pieces = self.sp_.DecodePieces(pieces)
    self.assertEqual(sp.decode(ids), detok_ids)
    self.assertEqual(sp.decode(pieces), detok_pieces)

    # add_bos, add_eos, reverse
    self.assertEqual([sp.bos_id()] + ids, sp.encode(text, add_bos=True))
    self.assertEqual(ids + [sp.eos_id()], sp.encode(text, add_eos=True))
    rids = ids[:]
    rids.reverse()
    self.assertEqual(rids, sp.encode(text, reverse=True))

    # different shape.
    self.assertEqual([ids, ids2], sp.encode([text, text2]))
    self.assertEqual([pieces, pieces2], sp.encode([text, text2], out_type=str))
    self.assertEqual([text, text2], sp.decode([ids, ids2]))
    self.assertEqual([text, text2], sp.decode([pieces, pieces2]))

  def test_new_api_init(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join('test', 'test_model.model'),
        add_bos=True,
        add_eos=True,
        out_type=str)
    text = 'hello world'
    pieces = ['<s>'] + self.sp_.EncodeAsPieces(text) + ['</s>']
    self.assertEqual(pieces, sp.encode(text))

  def test_new_api_sampling(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join('test', 'test_model.model'),
        out_type=str,
        enable_sampling=True)
    ids = defaultdict(int)
    for n in range(100):
      ++ids[' '.join(sp.encode('hello world'))]
    self.assertGreater(len(ids), 1)

    ids2 = defaultdict(int)
    for n in range(100):
      ++ids2[' '.join(sp.encode('hello world', enable_sampling=False))]
    self.assertEqual(len(ids2), 1)

  def test_valid_range(self):
    size = self.sp_.piece_size()
    funcs = [
        'IdToPiece', 'GetScore', 'IsUnknown', 'IsControl', 'IsUnused', 'IsByte',
        'DecodeIds', 'DecodeIdsAsSerializedProto'
    ]
    for m in funcs:
      getattr(self.sp_, m)([10, 20, 30])

    for m in funcs:
      try:
        getattr(self.sp_, m)([size])
        self.assertTrue(False)
      except:
        self.assertTrue(True)


def suite():
  suite = unittest.TestSuite()
  suite.addTests(unittest.makeSuite(TestSentencepieceProcessor))
  return suite


if __name__ == '__main__':
  unittest.main()
