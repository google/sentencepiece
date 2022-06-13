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

import io
import sentencepiece as spm
import unittest
import sys
import os
import pickle

from collections import defaultdict

print('VERSION={}'.format(spm.__version__))

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

    ids2 = self.sp_.encode_as_ids(text)
    pieces3 = self.sp_.encode_as_pieces(text)
    pieces4 = self.sp_.nbest_encode_as_pieces(text, 10)[0]
    self.assertEqual(pieces3, pieces4)
    self.assertEqual(pieces1, pieces3)
    self.assertEqual(ids, ids2)
    self.assertEqual(text, self.sp_.decode_pieces(pieces3))
    self.assertEqual(text, self.sp_.decode_ids(ids2))
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

    self.assertEqual(
        self.sp_.calculate_entropy(text, 0.1),
        self.sp_.CalculateEntropy(text, 0.1))

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

    ids2 = self.jasp_.encode_as_ids(text)
    pieces3 = self.jasp_.encode_as_pieces(text)
    pieces4 = self.jasp_.nbest_encode_as_pieces(text, 10)[0]
    self.assertEqual(pieces3, pieces4)
    self.assertEqual(pieces1, pieces3)
    self.assertEqual(ids, ids2)
    self.assertEqual(text, self.jasp_.decode_pieces(pieces1))
    self.assertEqual(text, self.jasp_.decode_ids(ids2))
    for n in range(100):
      self.assertEqual(
          text,
          self.jasp_.decode_pieces(
              self.jasp_.sample_encode_as_pieces(text, 64, 0.5)))
      self.assertEqual(
          text,
          self.jasp_.decode_pieces(
              self.jasp_.sample_encode_as_pieces(text, -1, 0.5)))

      self.assertEqual(
          self.jasp_.calculate_entropy(text, 0.1),
          self.jasp_.CalculateEntropy(text, 0.1))

  def test_train(self):
    spm.SentencePieceTrainer.Train('--input=' +
                                   os.path.join(data_dir, 'botchan.txt') +
                                   ' --model_prefix=m --vocab_size=1000')
    sp = spm.SentencePieceProcessor()
    sp.Load('m.model')
    with open(os.path.join(data_dir, 'botchan.txt'), 'r') as file:
      for line in file:
        sp.DecodePieces(sp.EncodeAsPieces(line))
        sp.DecodeIds(sp.EncodeAsIds(line))

  def test_train_iterator(self):
    spm.SentencePieceTrainer.Train('--input=' +
                                   os.path.join(data_dir, 'botchan.txt') +
                                   ' --model_prefix=m --vocab_size=1000')
    # Load as 'rb' for Python3.5/2.7.
    os1 = io.BytesIO()
    os2 = io.BytesIO()

    # suppress logging (redirect to /dev/null)
    spm.SentencePieceTrainer.train(
        input=os.path.join(data_dir, 'botchan.txt'),
        model_prefix='m',
        vocab_size=1000,
        logstream=open(os.devnull, 'w'))

    with open(os.path.join(data_dir, 'botchan.txt'), 'rb') as is1:
      spm.SentencePieceTrainer.train(
          sentence_iterator=is1,
          model_prefix='m',
          vocab_size=1000,
          logstream=open(os.devnull, 'w'))

    spm.SentencePieceTrainer.train(
        input=os.path.join(data_dir, 'botchan.txt'),
        model_writer=os1,
        vocab_size=1000,
        logstream=open(os.devnull, 'w'))

    with open(os.path.join(data_dir, 'botchan.txt'), 'rb') as is2:
      spm.SentencePieceTrainer.train(
          sentence_iterator=is2,
          model_writer=os2,
          vocab_size=1000,
          logstream=open(os.devnull, 'w'))

    sp1 = spm.SentencePieceProcessor(model_proto=os1.getvalue())
    sp2 = spm.SentencePieceProcessor(model_proto=os2.getvalue())
    self.assertEqual([sp1.id_to_piece(i) for i in range(sp1.get_piece_size())],
                     [sp2.id_to_piece(i) for i in range(sp2.get_piece_size())])

  def test_train_kwargs(self):
    # suppress logging (redirect to /dev/null)
    spm.SentencePieceTrainer.train(
        input=[os.path.join(data_dir, 'botchan.txt')],
        model_prefix='m',
        vocab_size=1002,
        user_defined_symbols=['foo', 'bar', ',', ' ', '\t', '\b', '\n', '\r'],
        logstream=open(os.devnull, 'w'))
    sp = spm.SentencePieceProcessor()
    sp.Load('m.model')
    with open(os.path.join(data_dir, 'botchan.txt'), 'r') as file:
      for line in file:
        sp.DecodePieces(sp.EncodeAsPieces(line))
        sp.DecodeIds(sp.EncodeAsIds(line))

    s = 'hello\tworld\r\nthis\tis a \b pen'
    self.assertEqual(s, sp.decode(sp.encode(s)))

  def test_serialized_proto(self):
    text = 'I saw a girl with a telescope.'
    s1 = self.sp_.EncodeAsSerializedProto(text)
    s2 = self.sp_.SampleEncodeAsSerializedProto(text, 10, 0.2)
    s3 = self.sp_.NBestEncodeAsSerializedProto(text, 10)
    s4 = self.sp_.DecodePiecesAsSerializedProto(['foo', 'bar'])
    s5 = self.sp_.DecodeIdsAsSerializedProto([20, 30])

    t1 = self.sp_.encode_as_serialized_proto(text)
    t2 = self.sp_.sample_encode_as_serialized_proto(text, 10, 0.2)
    t3 = self.sp_.nbest_encode_as_serialized_proto(text, 10)
    t4 = self.sp_.decode_pieces_as_serialized_proto(['foo', 'bar'])
    t5 = self.sp_.decode_ids_as_serialized_proto([20, 30])

    self.assertEqual(type(s1), bytes)
    self.assertEqual(type(s2), bytes)
    self.assertEqual(type(t2), bytes)
    self.assertEqual(type(s3), bytes)
    self.assertEqual(type(s4), bytes)
    self.assertEqual(type(s5), bytes)

    self.assertEqual(s1, t1)
    self.assertEqual(s3, t3)
    self.assertEqual(s4, t4)
    self.assertEqual(s5, t5)

  def test_new_api(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join('test', 'test_model.model'))
    text = 'hello world'
    text2 = 'Tokyo'
    ids = self.sp_.EncodeAsIds(text)
    ids2 = self.sp_.EncodeAsIds(text2)
    pieces = self.sp_.EncodeAsPieces(text)
    pieces2 = self.sp_.EncodeAsPieces(text2)
    protos = self.sp_.EncodeAsSerializedProto(text)
    proto2 = self.sp_.EncodeAsSerializedProto(text2)

    self.assertEqual(sp.encode(text, out_type=int), ids)
    self.assertEqual(sp.encode(text, out_type=str), pieces)
    self.assertEqual(sp.encode(text, out_type='proto'), protos)

    self.assertEqual(sp.encode([text], out_type=int), [ids])
    self.assertEqual(sp.encode([text], out_type=str), [pieces])
    self.assertEqual(sp.encode([text], out_type='proto'), [protos])

    detok_ids = self.sp_.DecodeIds(ids)
    detok_pieces = self.sp_.DecodePieces(pieces)
    self.assertEqual(sp.decode(ids), detok_ids)
    self.assertEqual(sp.decode(pieces), detok_pieces)
    self.assertEqual(sp.decode([]), '')
    self.assertEqual(sp.decode([[]]), [''])

    # add_bos, add_eos, reverse
    self.assertEqual([sp.bos_id()] + ids, sp.encode(text, add_bos=True))
    self.assertEqual(ids + [sp.eos_id()], sp.encode(text, add_eos=True))
    self.assertEqual(ids + [sp.eos_id()], sp.EncodeAsIds(text, add_eos=True))
    rids = ids[:]
    rids.reverse()

    self.assertEqual(rids, sp.encode(text, reverse=True))
    self.assertEqual(rids, sp.EncodeAsIds(text, reverse=True))

    # different shape.
    self.assertEqual([ids, ids2], sp.encode([text, text2]))
    self.assertEqual([pieces, pieces2], sp.encode([text, text2], out_type=str))
    self.assertEqual([text, text2], sp.decode([ids, ids2]))
    self.assertEqual([text, text2], sp.decode([pieces, pieces2]))

    pieces = list(reversed(self.sp_.EncodeAsPieces(text)))
    self.assertEqual(pieces, sp.encode(text, reverse=True, out_type=str))

    # emit unk piece
    unk_char = '藤'
    pieces = self.sp_.EncodeAsIds(unk_char, emit_unk_piece=True)
    pieces2 = self.sp_.encode(unk_char, out_type=int, emit_unk_piece=True)
    self.assertEqual(pieces[1], sp.unk_id())
    self.assertEqual(pieces2[1], sp.unk_id())
    self.assertEqual(pieces, pieces2)

    pieces = self.sp_.EncodeAsPieces(unk_char, emit_unk_piece=True)
    pieces2 = self.sp_.encode(unk_char, out_type=str, emit_unk_piece=True)
    self.assertEqual(pieces[1], '<unk>')
    self.assertEqual(pieces2[1], '<unk>')
    self.assertEqual(pieces, pieces2)

    pieces = self.sp_.EncodeAsPieces(unk_char, emit_unk_piece=False)
    pieces2 = self.sp_.encode(unk_char, out_type=str, emit_unk_piece=False)
    self.assertEqual(pieces[1], unk_char)
    self.assertEqual(pieces2[1], unk_char)
    self.assertEqual(pieces, pieces2)

  def test_new_api_init(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join('test', 'test_model.model'),
        add_bos=True,
        add_eos=True,
        out_type=str)
    text = 'hello world'
    pieces = ['<s>'] + self.sp_.EncodeAsPieces(text) + ['</s>']
    self.assertEqual(pieces, sp.encode(text))

    pieces = self.sp_.EncodeAsPieces(text) + ['</s>']
    self.assertEqual(pieces, sp.encode(text, add_bos=False, add_eos=True))

  def test_sampling(self):
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

  def test_nbest(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join('test', 'test_model.model'))
    text = 'hello world'
    results = sp.nbest_encode(text, nbest_size=10, out_type=str)
    self.assertEqual(results, sp.NBestEncode(text, nbest_size=10, out_type=str))
    for n in results:
      self.assertEqual(sp.decode(n), text)
    decoded = sp.decode(results)
    for n in decoded:
      self.assertEqual(n, text)
    results = sp.nbest_encode(text, nbest_size=10, out_type=int)
    self.assertEqual(results, sp.NBestEncode(text, nbest_size=10, out_type=int))
    for n in results:
      self.assertEqual(sp.decode(n), text)
    decoded = sp.decode(results)
    for n in decoded:
      self.assertEqual(n, text)

  def test_sample_and_score(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join('test', 'test_model.model'))
    text = 'hello world'
    results = sp.sample_encode_and_score(text, wor=True, out_type=str)
    for n in results:
      self.assertEqual(sp.decode(n[0]), text)
    results = sp.sample_encode_and_score(text, wor=True, out_type=int)
    for n in results:
      self.assertEqual(sp.decode(n[0]), text)

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

  def test_batch(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join('test', 'test_model.model'))
    with open(os.path.join(data_dir, 'botchan.txt'), 'r') as file:
      texts = file.readlines()

    r1 = sp.encode(texts, out_type=str, num_threads=None)
    r2 = sp.encode(texts, out_type=str, num_threads=1)
    r3 = sp.encode(texts, out_type=str, num_threads=-1)
    r4 = sp.encode(texts, out_type=str, num_threads=8)
    r5 = [sp.encode(s, out_type=str) for s in texts]
    self.assertEqual(r1, r2)
    self.assertEqual(r1, r3)
    self.assertEqual(r1, r4)
    self.assertEqual(r1, r5)

    d1 = sp.decode(r1, num_threads=None)
    d2 = sp.decode(r2, num_threads=1)
    d3 = sp.decode(r3, num_threads=-1)
    d4 = sp.decode(r4, num_threads=8)
    d5 = [sp.decode(s) for s in r5]
    self.assertEqual(d1, d2)
    self.assertEqual(d1, d3)
    self.assertEqual(d1, d4)
    self.assertEqual(d1, d5)

    r1 = sp.encode(texts, out_type=int, num_threads=None)
    r2 = sp.encode(texts, out_type=int, num_threads=1)
    r3 = sp.encode(texts, out_type=int, num_threads=-1)
    r4 = sp.encode(texts, out_type=int, num_threads=8)
    r5 = [sp.encode(s, out_type=int) for s in texts]
    self.assertEqual(r1, r2)
    self.assertEqual(r1, r3)
    self.assertEqual(r1, r4)
    self.assertEqual(r1, r5)

    d1 = sp.decode(r1, num_threads=None)
    d2 = sp.decode(r2, num_threads=1)
    d3 = sp.decode(r3, num_threads=-1)
    d4 = sp.decode(r4, num_threads=8)
    d5 = [sp.decode(s) for s in r5]
    self.assertEqual(d1, d2)
    self.assertEqual(d1, d3)
    self.assertEqual(d1, d4)
    self.assertEqual(d1, d5)

    r1 = sp.encode(texts, out_type='proto', num_threads=None)
    r2 = sp.encode(texts, out_type='proto', num_threads=1)
    r3 = sp.encode(texts, out_type='proto', num_threads=-1)
    r4 = sp.encode(texts, out_type='proto', num_threads=8)
    r5 = [sp.encode(s, out_type='proto') for s in texts]
    self.assertEqual(r1, r2)
    self.assertEqual(r1, r3)
    self.assertEqual(r1, r4)
    self.assertEqual(r1, r5)

    e1 = sp.calculate_entropy(texts, theta=1.0, num_threads=10)
    e2 = sp.CalculateEntropy(texts, theta=1.0, num_threads=10)
    e3 = [sp.calculate_entropy(s, theta=1.0) for s in texts]
    self.assertEqual(e1, e2)
    self.assertEqual(e1, e3)

  def test_pickle(self):
    with open('sp.pickle', 'wb') as f:
      pickle.dump(self.sp_, f)

    id1 = self.sp_.encode('hello world.', out_type=int)

    with open('sp.pickle', 'rb') as f:
      sp = pickle.load(f)

    id2 = sp.encode('hello world.', out_type=int)

    self.assertEqual(id1, id2)


def suite():
  suite = unittest.TestSuite()
  suite.addTests(unittest.makeSuite(TestSentencepieceProcessor))
  return suite


if __name__ == '__main__':
  unittest.main()
