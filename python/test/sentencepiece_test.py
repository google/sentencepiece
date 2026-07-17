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

from collections import defaultdict
import glob
import io
import os
import pickle
import sys
import tempfile
import threading
import unittest
import pytest
import sentencepiece as spm
try:
  from sentencepiece import sentencepiece_pb2
  has_protobuf = True
except ImportError:
  has_protobuf = False

TESTED_RETURN_TYPES = [str, int, 'serialized_proto']
if has_protobuf:
  TESTED_RETURN_TYPES.append('proto')

HERE = os.path.dirname(os.path.abspath(__file__))
print(HERE)

print('VERSION={}'.format(spm.__version__))

data_dir = HERE


class TestSentencepieceProcessor(unittest.TestCase):
  """Test case for SentencePieceProcessor"""

  def setUp(self):
    self.sp_ = spm.SentencePieceProcessor()
    self.jasp_ = spm.SentencePieceProcessor()
    self.assertTrue(self.sp_.Load(os.path.join(HERE, 'test_model.model')))
    self.assertTrue(self.jasp_.Load(os.path.join(HERE, 'test_ja_model.model')))
    with open(os.path.join(HERE, 'test_model.model'), 'rb') as f:
      self.assertTrue(self.sp_.LoadFromSerializedProto(f.read()))
    with open(os.path.join(HERE, 'test_ja_model.model'), 'rb') as f:
      self.assertTrue(self.jasp_.LoadFromSerializedProto(f.read()))

  def tearDown(self):
    patterns = ['m_*.model', 'm_*.vocab', 'sp_*.pickle']
    for pattern in patterns:
      for file_path in glob.glob(pattern):
        if os.path.isfile(file_path):
          os.remove(file_path)

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

  def test_decode_invalid_ids(self):
    # kOutOfRange should map to IndexError
    with self.assertRaises(IndexError):
      self.sp_.decode([10000])
    with self.assertRaises(IndexError):
      self.sp_.decode([[0, 10000]])
    with self.assertRaises(IndexError):
      self.sp_.decode(10000)
    with self.assertRaises(IndexError):
      self.sp_.DecodeIds([10000])

  def test_decode_empty_input(self):
    # Empty input must return a value typed according to return_type,
    # matching the non-empty code path.
    for empty in ([], None):
      self.assertEqual(self.sp_.decode(empty, return_type=str), '')
      self.assertEqual(self.sp_.decode(empty, return_type=bytes), b'')
      serialized = self.sp_.decode(empty, return_type='serialized_proto')
      self.assertEqual(type(serialized), bytes)
      offsets = self.sp_.decode(empty, return_type='offset_mapping')
      self.assertIsInstance(offsets, dict)
      self.assertEqual(offsets['text'], '')
      self.assertEqual(offsets['ids'], [])
      self.assertEqual(offsets['pieces'], [])
      self.assertEqual(offsets['offsets'], [])
      if has_protobuf:
        proto = self.sp_.decode(empty, return_type='proto')
        self.assertIsInstance(proto, sentencepiece_pb2.SentencePieceText)
        self.assertEqual(proto.text, '')
        self.assertEqual(len(proto.pieces), 0)

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
          self.sp_.DecodePieces(self.sp_.SampleEncodeAsPieces(text, 64, 0.5)),
      )
      self.assertEqual(
          text,
          self.sp_.DecodePieces(self.sp_.SampleEncodeAsPieces(text, -1, 0.5)),
      )
      self.assertEqual(
          text, self.sp_.DecodeIds(self.sp_.SampleEncodeAsIds(text, 64, 0.5))
      )
      self.assertEqual(
          text, self.sp_.DecodeIds(self.sp_.SampleEncodeAsIds(text, -1, 0.5))
      )

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
              self.sp_.sample_encode_as_pieces(text, 64, 0.5)
          ),
      )
      self.assertEqual(
          text,
          self.sp_.decode_pieces(
              self.sp_.sample_encode_as_pieces(text, -1, 0.5)
          ),
      )
      self.assertEqual(
          text,
          self.sp_.decode_ids(self.sp_.sample_encode_as_ids(text, 64, 0.5)),
      )
      self.assertEqual(
          text,
          self.sp_.decode_ids(self.sp_.sample_encode_as_ids(text, -1, 0.5)),
      )



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
              self.jasp_.SampleEncodeAsPieces(text, 64, 0.5)
          ),
      )
      self.assertEqual(
          text,
          self.jasp_.DecodePieces(
              self.jasp_.SampleEncodeAsPieces(text, -1, 0.5)
          ),
      )

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
              self.jasp_.sample_encode_as_pieces(text, 64, 0.5)
          ),
      )
      self.assertEqual(
          text,
          self.jasp_.decode_pieces(
              self.jasp_.sample_encode_as_pieces(text, -1, 0.5)
          ),
      )



  def test_train(self):
    tid = threading.get_native_id()
    spm.SentencePieceTrainer.Train(
        '--input='
        + os.path.join(data_dir, 'botchan.txt')
        + f' --model_prefix=m_{tid} --vocab_size=1000'
    )
    sp = spm.SentencePieceProcessor()
    sp.Load(f'm_{tid}.model')
    with open(os.path.join(data_dir, 'botchan.txt'), 'r') as file:
      for line in file:
        sp.DecodePieces(sp.EncodeAsPieces(line))
        sp.DecodeIds(sp.EncodeAsIds(line))

  def test_special_tokens_combinations(self):
    tid = threading.get_native_id()

    # 1. CONTROL (default)
    spm.SentencePieceTrainer.train(
        input=os.path.join(data_dir, 'botchan.txt'),
        model_prefix=f'm_control_{tid}',
        vocab_size=1000,
    )
    sp = spm.SentencePieceProcessor()
    self.assertTrue(sp.Load(f'm_control_{tid}.model'))
    self.assertNotEqual(-1, sp.bos_id())
    self.assertEqual([sp.bos_id()] + sp.encode('a'), sp.encode('a', add_bos=True))
    self.assertEqual(sp.encode('a') + [sp.eos_id()], sp.encode('a', add_eos=True))
    self.assertEqual([sp.bos_id()] + sp.encode('a') + [sp.eos_id()], sp.encode('a', add_bos=True, add_eos=True))
    
    self.assertEqual([sp.IdToPiece(sp.bos_id())] + sp.encode('a', return_type=str), sp.encode('a', add_bos=True, return_type=str))

    # 2. USER_DEFINED
    spm.SentencePieceTrainer.train(
        input=os.path.join(data_dir, 'botchan.txt'),
        model_prefix=f'm_user_{tid}',
        vocab_size=1000,
        user_defined_symbols=['<s>', '</s>'],
        bos_piece='<s>',
        eos_piece='</s>',
    )
    sp = spm.SentencePieceProcessor()
    self.assertTrue(sp.Load(f'm_user_{tid}.model'))
    self.assertEqual(-1, sp.bos_id())
    with self.assertRaises(ValueError):
      sp.encode('a', add_bos=True)
    with self.assertRaises(ValueError):
      sp.encode('a', add_eos=True)
    with self.assertRaises(ValueError):
      sp.encode('a', add_bos=True, return_type=str)
    with self.assertRaises(ValueError):
      sp.encode('a', add_eos=True, return_type=str)

    # 3. Missing (disabled)
    spm.SentencePieceTrainer.train(
        input=os.path.join(data_dir, 'botchan.txt'),
        model_prefix=f'm_missing_{tid}',
        vocab_size=1000,
        bos_id=-1,
        eos_id=-1,
    )
    sp = spm.SentencePieceProcessor()
    self.assertTrue(sp.Load(f'm_missing_{tid}.model'))
    self.assertEqual(-1, sp.bos_id())
    with self.assertRaises(ValueError):
      sp.encode('a', add_bos=True)
    with self.assertRaises(ValueError):
      sp.encode('a', add_eos=True)
    with self.assertRaises(ValueError):
      sp.encode('a', add_bos=True, return_type=str)
    with self.assertRaises(ValueError):
      sp.encode('a', add_eos=True, return_type=str)

  def test_train_iterator(self):
    tid = threading.get_native_id()
    spm.SentencePieceTrainer.Train(
        '--input='
        + os.path.join(data_dir, 'botchan.txt')
        + f' --model_prefix=m_{tid} --vocab_size=1000'
    )
    # Load as 'rb' for Python3.5/2.7.
    os1 = io.BytesIO()
    os2 = io.BytesIO()

    # suppress logging (redirect to /dev/null)
    spm.SentencePieceTrainer.train(
        input=os.path.join(data_dir, 'botchan.txt'),
        model_prefix=f'm_{tid}',
        vocab_size=1000,
        # logstream=open(os.devnull, 'w'),
    )

    with open(os.path.join(data_dir, 'botchan.txt'), 'rb') as is1:
      spm.SentencePieceTrainer.train(
          sentence_iterator=is1,
          model_prefix=f'm_{tid}',
          vocab_size=1000,
          # logstream=open(os.devnull, 'w'),
      )

    spm.SentencePieceTrainer.train(
        input=os.path.join(data_dir, 'botchan.txt'),
        model_writer=os1,
        vocab_size=1000,
        # logstream=open(os.devnull, 'w'),
    )

    with open(os.path.join(data_dir, 'botchan.txt'), 'rb') as is2:
      spm.SentencePieceTrainer.train(
          sentence_iterator=is2,
          model_writer=os2,
          vocab_size=1000,
          # logstream=open(os.devnull, 'w'),
      )

    sp1 = spm.SentencePieceProcessor(model_proto=os1.getvalue())
    sp2 = spm.SentencePieceProcessor(model_proto=os2.getvalue())
    self.assertEqual(
        [sp1.id_to_piece(i) for i in range(sp1.get_piece_size())],
        [sp2.id_to_piece(i) for i in range(sp2.get_piece_size())],
    )

  def test_train_kwargs(self):
    tid = threading.get_native_id()
    # suppress logging (redirect to /dev/null)
    spm.SentencePieceTrainer.train(
        input=[os.path.join(data_dir, 'botchan.txt')],
        model_prefix=f'm_{tid}',
        vocab_size=1002,
        user_defined_symbols=['foo', 'bar', ',', ' ', '\t', '\b', '\n', '\r'],
        # logstream=open(os.devnull, 'w'),
    )
    sp = spm.SentencePieceProcessor()
    sp.Load(f'm_{tid}.model')
    with open(os.path.join(data_dir, 'botchan.txt'), 'r') as file:
      for line in file:
        sp.DecodePieces(sp.EncodeAsPieces(line))
        sp.DecodeIds(sp.EncodeAsIds(line))

    s = 'hello\tworld\r\nthis\tis a \b pen'
    self.assertEqual(s, sp.decode(sp.encode(s)))

  def test_serialized_proto(self):
    text = 'I saw a girl with a telescope.'
    y1 = self.sp_.encode(text, return_type='serialized_proto')
    y2 = self.sp_.encode(
        text, enable_sampling=True, return_type='serialized_proto'
    )
    y3 = self.sp_.nbest_encode(text, return_type='serialized_proto', nbest_size=10)
    y4 = self.sp_.decode(['foo', 'bar'], return_type='serialized_proto')
    y5 = self.sp_.decode([20, 30], return_type='serialized_proto')

    self.assertEqual(type(y1), bytes)
    self.assertEqual(type(y2), bytes)
    self.assertEqual(type(y3), bytes)
    self.assertEqual(type(y4), bytes)
    self.assertEqual(type(y5), bytes)

    ids = self.jasp_.EncodeAsIds(text)
    pieces = self.jasp_.EncodeAsPieces(text)
    s1 = self.jasp_.encode(text, return_type='serialized_proto')
    s2 = self.jasp_.decode(ids, return_type='serialized_proto')
    s3 = self.jasp_.decode(pieces, return_type='serialized_proto')
    self.assertEqual(s2, s1)
    self.assertEqual(s3, s1)

  def test_decode_bytes(self):
    texts = ['Hello world', '清水寺は京都にある。']
    ids = self.jasp_.encode(texts, return_type=int)
    s1 = self.jasp_.decode(ids, return_type=bytes)
    s2 = self.jasp_.decode(ids, return_type=str)
    self.assertEqual(len(s1), 2)
    self.assertEqual(type(s1[0]), bytes)
    self.assertEqual(type(s1[1]), bytes)
    self.assertEqual(len(s2), 2)
    self.assertEqual(type(s2[0]), str)
    self.assertEqual(type(s2[1]), str)
    self.assertEqual(s1[0].decode(encoding='utf-8'), s2[0])
    self.assertEqual(s1[1].decode(encoding='utf-8'), s2[1])

    text = 'Hello world'
    ids = self.jasp_.encode(text, return_type=int)
    s1 = self.jasp_.decode(ids, return_type=bytes)
    s2 = self.jasp_.decode(ids, return_type=str)
    self.assertEqual(type(s1), bytes)
    self.assertEqual(type(s2), str)
    self.assertEqual(s1.decode(encoding='utf-8'), s2)

    if has_protobuf:
      x = self.jasp_.encode(text, return_type='proto')
      self.assertIsInstance(x, sentencepiece_pb2.SentencePieceText)
      self.assertEqual(x.text, text)

      x = self.jasp_.decode(ids, return_type='proto')
      self.assertIsInstance(x, sentencepiece_pb2.SentencePieceText)
      self.assertEqual(x.text, text)

  @unittest.skipUnless(has_protobuf, 'protobuf is not installed')
  def test_proto(self):
    text = 'I saw a girl with a telescope.'
    s1 = self.sp_.EncodeAsProto(text)
    s2 = self.sp_.SampleEncodeAsProto(text, 10, 0.2)
    s3 = self.sp_.NBestEncodeAsProto(text, 10)
    s4 = self.sp_.DecodePiecesAsProto(['foo', 'bar'])
    s5 = self.sp_.DecodeIdsAsProto([20, 30])
    s7 = self.sp_.ParallelEncodeAsProto(text, chunk_len=5, num_threads=2)

    t1 = self.sp_.encode_as_proto(text)
    t2 = self.sp_.sample_encode_as_proto(text, 10, 0.2)
    t3 = self.sp_.nbest_encode_as_proto(text, 10)
    t4 = self.sp_.decode_pieces_as_proto(['foo', 'bar'])
    t5 = self.sp_.decode_ids_as_proto([20, 30])
    t7 = self.sp_.parallel_encode_as_proto(text, chunk_len=5, num_threads=2)

    y1 = self.sp_.encode(text, return_type='proto')
    y2 = self.sp_.encode(text, enable_sampling=True, return_type='proto')
    y3 = self.sp_.nbest_encode(text, return_type='proto', nbest_size=10)
    y4 = self.sp_.decode(['foo', 'bar'], return_type='proto')
    y5 = self.sp_.decode([20, 30], return_type='proto')
    y7 = self.sp_.parallel_encode(text, chunk_len=5, num_threads=2, return_type='proto')

    self.assertIsInstance(s1, sentencepiece_pb2.SentencePieceText)
    self.assertIsInstance(s2, sentencepiece_pb2.SentencePieceText)
    self.assertIsInstance(s3, sentencepiece_pb2.NBestSentencePieceText)
    self.assertIsInstance(s4, sentencepiece_pb2.SentencePieceText)
    self.assertIsInstance(s5, sentencepiece_pb2.SentencePieceText)
    self.assertIsInstance(s7, sentencepiece_pb2.SentencePieceText)

    self.assertEqual(s1, t1)
    self.assertEqual(s3, t3)
    self.assertEqual(s4, t4)
    self.assertEqual(s5, t5)
    self.assertEqual(s7, t7)
    self.assertEqual(s1, y1)
    self.assertEqual(s3, y3)
    self.assertEqual(s4, y4)
    self.assertEqual(s5, y5)
    self.assertEqual(s7, y7)

    x1 = self.sp_.encode(text, return_type='serialized_proto')
    x2 = self.sp_.encode(text, enable_sampling=True, return_type='serialized_proto')
    x3 = self.sp_.nbest_encode(text, return_type='serialized_proto', nbest_size=10)
    x4 = self.sp_.decode(['foo', 'bar'], return_type='serialized_proto')
    x5 = self.sp_.decode([20, 30], return_type='serialized_proto')
    x7 = self.sp_.parallel_encode(text, chunk_len=5, num_threads=2, return_type='serialized_proto')

    self.assertEqual(x1, t1.SerializeToString())
    self.assertEqual(x3, t3.SerializeToString())
    self.assertEqual(x4, t4.SerializeToString())
    self.assertEqual(x5, t5.SerializeToString())
    self.assertEqual(x7, s7.SerializeToString())

    v1 = self.sp_.EncodeAsIds(text)
    v2 = self.sp_.EncodeAsPieces(text)
    self.assertEqual([x.id for x in s1.pieces], v1)
    self.assertEqual([x.piece for x in s1.pieces], v2)
    self.assertEqual(text, s1.text)

    text_bytes = s1.text.encode('utf-8')
    surfaces1 = [text_bytes[x.begin : x.end].decode('utf-8') for x in s1.pieces]
    surfaces2 = [x.surface for x in s1.pieces]
    self.assertEqual(surfaces1, surfaces2)

    # slice
    self.assertEqual(s1.pieces[::-1], list(reversed(s1.pieces)))
    self.assertEqual(s3.nbests[::-1], list(reversed(s3.nbests)))

    # Japanese offset
    s1_ja = self.jasp_.EncodeAsProto(
        '吾輩は猫である。Hello world. ABC 123'
    )
    text_bytes_ja = s1_ja.text.encode('utf-8')
    surfaces1_ja = [text_bytes_ja[x.begin : x.end].decode('utf-8') for x in s1_ja.pieces]
    surfaces2_ja = [x.surface for x in s1_ja.pieces]
    self.assertEqual(surfaces1_ja, surfaces2_ja)

    ids_ja = [x.id for x in s1_ja.pieces]
    s2_ja = self.jasp_.DecodeIdsAsProto(ids_ja)
    self.assertEqual(s2_ja, s1_ja)

    pieces_ja = [x.piece for x in s1_ja.pieces]
    s2_ja = self.jasp_.DecodePiecesAsProto(pieces_ja)
    self.assertEqual(s2_ja, s1_ja)

    # Verify immutable_proto raises ValueError
    with self.assertRaises(ValueError):
      self.sp_.encode(text, return_type='immutable_proto')
    with self.assertRaises(ValueError):
      self.sp_.decode([20, 30], return_type='immutable_proto')

  def test_new_api(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(HERE, 'test_model.model')
    )
    text = 'hello world'
    text2 = 'Tokyo'
    ids = self.sp_.EncodeAsIds(text)
    ids2 = self.sp_.EncodeAsIds(text2)
    pieces = self.sp_.EncodeAsPieces(text)
    pieces2 = self.sp_.EncodeAsPieces(text2)
    sprotos = self.sp_.encode(text, return_type='serialized_proto')
    sproto2 = self.sp_.encode(text2, return_type='serialized_proto')


    self.assertEqual(sp.encode(text, return_type=int), ids)
    self.assertEqual(sp.encode(text, return_type=str), pieces)
    self.assertEqual(sp.encode(text, return_type='serialized_proto'), sprotos)


    self.assertEqual(sp.encode([text], return_type=int), [ids])
    self.assertEqual(sp.encode([text], return_type=str), [pieces])
    self.assertEqual(sp.encode([text], return_type='serialized_proto'), [sprotos])




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
    self.assertEqual([pieces, pieces2], sp.encode([text, text2], return_type=str))
    self.assertEqual([text, text2], sp.decode([ids, ids2]))
    self.assertEqual([text, text2], sp.decode([pieces, pieces2]))

    pieces = list(reversed(self.sp_.EncodeAsPieces(text)))
    self.assertEqual(pieces, sp.encode(text, reverse=True, return_type=str))

    # emit unk piece
    unk_char = '藤'
    pieces = self.sp_.EncodeAsIds(unk_char, emit_unk_piece=True)
    pieces2 = self.sp_.encode(unk_char, return_type=int, emit_unk_piece=True)
    self.assertEqual(pieces[1], sp.unk_id())
    self.assertEqual(pieces2[1], sp.unk_id())
    self.assertEqual(pieces, pieces2)

    pieces = self.sp_.EncodeAsPieces(unk_char, emit_unk_piece=True)
    pieces2 = self.sp_.encode(unk_char, return_type=str, emit_unk_piece=True)
    self.assertEqual(pieces[1], '<unk>')
    self.assertEqual(pieces2[1], '<unk>')
    self.assertEqual(pieces, pieces2)

    pieces = self.sp_.EncodeAsPieces(unk_char, emit_unk_piece=False)
    pieces2 = self.sp_.encode(unk_char, return_type=str, emit_unk_piece=False)
    self.assertEqual(pieces[1], unk_char)
    self.assertEqual(pieces2[1], unk_char)
    self.assertEqual(pieces, pieces2)

  @unittest.skipUnless(has_protobuf, 'protobuf is not installed')
  def test_new_api_proto(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(HERE, 'test_model.model')
    )
    text = 'hello world'
    text2 = 'Tokyo'
    ids = self.sp_.EncodeAsIds(text)
    ids2 = self.sp_.EncodeAsIds(text2)
    pieces = self.sp_.EncodeAsPieces(text)
    pieces2 = self.sp_.EncodeAsPieces(text2)
    protos = self.sp_.EncodeAsProto(text)
    protos2 = self.sp_.EncodeAsProto(text2)

    self.assertEqual(sp.encode(text, return_type='proto'), protos)
    self.assertEqual(sp.encode([text], return_type='proto'), [protos])

    self.assertEqual(len(protos.pieces), len(pieces))
    self.assertEqual(len(protos.pieces), len(ids))
    self.assertEqual(protos.text, text)

    self.assertEqual(len(protos2.pieces), len(pieces2))
    self.assertEqual(len(protos2.pieces), len(ids2))
    self.assertEqual(protos2.text, text2)

    for i in range(len(protos.pieces)):
      self.assertEqual(ids[i], protos.pieces[i].id)
      self.assertEqual(pieces[i], protos.pieces[i].piece)

    for i, piece in enumerate(protos.pieces):
      self.assertEqual(ids[i], piece.id)
      self.assertEqual(pieces[i], piece.piece)

    for i in range(len(protos2.pieces)):
      self.assertEqual(ids2[i], protos2.pieces[i].id)
      self.assertEqual(pieces2[i], protos2.pieces[i].piece)

    for i, piece in enumerate(protos2.pieces):
      self.assertEqual(ids2[i], piece.id)
      self.assertEqual(pieces2[i], piece.piece)

  def test_new_api_init(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(HERE, 'test_model.model'),
        add_bos=True,
        add_eos=True,
        return_type=str,
    )
    text = 'hello world'
    pieces = ['<s>'] + self.sp_.EncodeAsPieces(text) + ['</s>']
    self.assertEqual(pieces, sp.encode(text))

    pieces = self.sp_.EncodeAsPieces(text) + ['</s>']
    self.assertEqual(pieces, sp.encode(text, add_bos=False, add_eos=True))

  def test_classmethod_factories(self):
    model_path = os.path.join(HERE, 'test_model.model')

    # 1. Test loading via from_file
    sp_file = spm.SentencePieceProcessor.from_file(
        model_path,
        add_bos=True,
        add_eos=True,
        return_type=str
    )
    text = 'hello world'
    expected_pieces = ['<s>'] + self.sp_.EncodeAsPieces(text) + ['</s>']
    self.assertEqual(expected_pieces, sp_file.encode(text))

    # 2. Test loading via from_proto
    with open(model_path, 'rb') as f:
      model_proto = f.read()

    sp_proto = spm.SentencePieceProcessor.from_proto(
        model_proto,
        add_bos=True,
        add_eos=True,
        return_type=str
    )
    self.assertEqual(expected_pieces, sp_proto.encode(text))

    # 3. Verify parameter exclusions (should raise ValueError)
    with self.assertRaises(ValueError):
      spm.SentencePieceProcessor.from_file(model_path, model_proto=model_proto)

    with self.assertRaises(ValueError):
      spm.SentencePieceProcessor.from_proto(model_proto, model_file=model_path)

  def test_sampling(self):
    sp = self.sp_

    for return_type in TESTED_RETURN_TYPES:
      ids = defaultdict(int)
      for n in range(100):
        out = sp.encode('hello world', return_type=return_type, enable_sampling=True)
        if has_protobuf and isinstance(out, sentencepiece_pb2.SentencePieceText):
          out = out.SerializeToString()
        if type(out) is list:
          out = tuple(out)
        ids[out] += 1  # Fixed the C++ style ++ bug to actually increment
      self.assertGreater(len(ids), 1)

      ids2 = defaultdict(int)
      for n in range(100):
        out = sp.encode('hello world', return_type=return_type, enable_sampling=False)
        if has_protobuf and isinstance(out, sentencepiece_pb2.SentencePieceText):
          out = out.SerializeToString()
        if type(out) is list:
          out = tuple(out)
        ids2[out] += 1  # Fixed the C++ style ++ bug to actually increment
      self.assertEqual(len(ids2), 1)

      out = sp.encode(
          ['hello world', 'this is a test'],
          return_type=return_type,
          enable_sampling=True,
      )
      self.assertEqual(len(out), 2)
      out = sp.encode(
          ['hello world', 'this is a test'],
          return_type=return_type,
          enable_sampling=False,
      )
      self.assertEqual(len(out), 2)

  def test_word_model_user_defined_symbol(self):
    with tempfile.TemporaryDirectory() as work_dir:
      input_file = os.path.join(work_dir, 'input.txt')
      model_prefix = os.path.join(work_dir, 'word_model')
      with open(input_file, 'w', encoding='utf-8') as f:
        f.write('hello . world\n')
        f.write('hello . test\n')

      spm.SentencePieceTrainer.train(
          input=input_file,
          model_prefix=model_prefix,
          model_type='word',
          vocab_size=8,
          hard_vocab_limit=False,
          normalization_rule_name='identity',
          user_defined_symbols=['.'],
          bos_id=-1,
          eos_id=-1,
      )

      sp = spm.SentencePieceProcessor(model_file=model_prefix + '.model')
      pieces = sp.encode('hello . world', return_type=str)

      self.assertEqual(['▁hello', '▁.', '▁world'], pieces)

  def test_nbest(self):
    sp = self.sp_
    text = 'hello world'
    text2 = 'I have a pen.'

    for return_type in TESTED_RETURN_TYPES:
      results = sp.nbest_encode(text, nbest_size=10, return_type=return_type)
      self.assertEqual(
          results, sp.NBestEncode(text, nbest_size=10, return_type=return_type)
      )

      if return_type in [str, int]:
        for n in results:
          self.assertEqual(sp.decode(n), text)

        for n in sp.decode(results):
          self.assertEqual(n, text)

      # batch test
      results = sp.nbest_encode([text, text2], nbest_size=10, return_type=return_type)
      self.assertEqual(
          results,
          sp.NBestEncode([text, text2], nbest_size=10, return_type=return_type),
      )
      self.assertEqual(len(results), 2)

      if return_type in [str, int]:
        for n in results[0]:
          self.assertEqual(sp.decode(n), text)

        for n in results[1]:
          self.assertEqual(sp.decode(n), text2)

        decoded = sp.decode(results[0])
        self.assertEqual(len(decoded), 10)
        for n in decoded:
          self.assertEqual(n, text)
        decoded = sp.decode(results[1])
        self.assertEqual(len(decoded), 10)
        for n in decoded:
          self.assertEqual(n, text2)

    self.assertEqual(
        sp.nbest_encode(text, nbest_size=10, return_type=str),
        sp.nbest_encode_as_pieces(text, nbest_size=10),
    )
    self.assertEqual(
        sp.nbest_encode(text, nbest_size=10, return_type=int),
        sp.nbest_encode_as_ids(text, nbest_size=10),
    )

    if has_protobuf:
      self.assertEqual(
          sp.nbest_encode(text, nbest_size=10, return_type='proto'),
          sp.nbest_encode_as_proto(text, nbest_size=10),
      )
  # SetNBestTimeout/set_nbest_timeout modify a global atomic variable in C++.
  # This makes this test thread-unsafe when run in parallel with other tests
  # that perform nbest encoding.
  @pytest.mark.thread_unsafe
  def test_nbest_timeout(self):
    model_prefix = 'm_timeout'
    spm.SentencePieceTrainer.train(
        input=os.path.join(HERE, 'botchan.txt'),
        model_prefix=model_prefix,
        vocab_size=1000,
        model_type='unigram',
    )
    sp = spm.SentencePieceProcessor(model_file=model_prefix + '.model')
    long_input = 'the' * 1000
    results = sp.nbest_encode(long_input, nbest_size=10, return_type=str)
    self.assertEqual(len(results), 10)

    spm.SetNBestTimeout(1)
    results_timeout = sp.nbest_encode(long_input, nbest_size=10, return_type=str)
    self.assertEqual(len(results_timeout), 1)

    spm.SetNBestTimeout(0)
    results_no_timeout = sp.nbest_encode(long_input, nbest_size=10, return_type=str)
    self.assertEqual(len(results_no_timeout), 10)

    spm.set_nbest_timeout(1)
    results_timeout2 = sp.nbest_encode(long_input, nbest_size=10, return_type=str)
    self.assertEqual(len(results_timeout2), 1)
    spm.set_nbest_timeout(0)



  def test_valid_range(self):
    size = self.sp_.piece_size()
    funcs = [
        'IdToPiece',
        'GetScore',
        'IsUnknown',
        'IsControl',
        'IsUnused',
        'IsByte',
        'DecodeIds',
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
        model_file=os.path.join(HERE, 'test_model.model')
    )
    with open(os.path.join(data_dir, 'botchan.txt'), 'r') as file:
      texts = file.readlines()

    pool = spm.ThreadPool(8)
    self.assertEqual(pool.num_threads(), 8)

    for return_type in TESTED_RETURN_TYPES:
      r1 = sp.encode(texts, return_type=return_type, num_threads=None)
      r2 = sp.encode(texts, return_type=return_type, num_threads=1)
      r3 = sp.encode(texts, return_type=return_type, num_threads=-1)
      r4 = sp.encode(texts, return_type=return_type, num_threads=8)
      r5 = sp.encode(texts, return_type=return_type, thread_pool=pool)
      r6 = [sp.encode(s, return_type=return_type) for s in texts]
      self.assertEqual(r1, r2)
      self.assertEqual(r1, r3)
      self.assertEqual(r1, r4)
      self.assertEqual(r1, r5)
      self.assertEqual(r1, r6)

      if return_type in [str, int]:
        d1 = sp.decode(r1, num_threads=None)
        d2 = sp.decode(r2, num_threads=1)
        d3 = sp.decode(r3, num_threads=-1)
        d4 = sp.decode(r4, num_threads=8)
        d5 = sp.decode(r4, thread_pool=pool)
        d6 = [sp.decode(s) for s in r6]

        self.assertEqual(d1, d2)
        self.assertEqual(d1, d3)
        self.assertEqual(d1, d4)
        self.assertEqual(d1, d5)
        self.assertEqual(d1, d6)

  def test_parallel(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(HERE, 'test_bpe_model.model')
    )
    with open(os.path.join(data_dir, 'botchan.txt'), 'r') as file:
      texts = file.readlines()

    # make long input
    text = ''.join(texts)

    pool = spm.ThreadPool(8)
    self.assertEqual(pool.num_threads(), 8)

    for return_type in TESTED_RETURN_TYPES:
      r1 = sp.encode(text, return_type=return_type)
      for chunk_len in [100, 1000, 10000]:
        r2 = sp.parallel_encode(
            text, return_type=return_type, chunk_len=chunk_len, num_threads=8
        )
        r3 = sp.parallel_encode(
            text, return_type=return_type, chunk_len=chunk_len, thread_pool=pool
        )
        self.assertEqual(r1, r2)
        self.assertEqual(r1, r3)

  def test_pickle(self):
    tid = threading.get_native_id()
    with open(f'sp_{tid}.pickle', 'wb') as f:
      pickle.dump(self.sp_, f)

    id1 = self.sp_.encode('hello world.', return_type=int)

    with open(f'sp_{tid}.pickle', 'rb') as f:
      sp = pickle.load(f)

    id2 = sp.encode('hello world.', return_type=int)

    self.assertEqual(id1, id2)

  def test_global_params(self):
    spm.SetRandomGeneratorSeed(0)
    spm.SetMinLogLevel(2)
    spm.set_random_generator_seed(1)
    spm.set_min_log_level(3)

  def test_normalize(self):
    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(HERE, 'test_model.model')
    )

    self.assertEqual('▁KADOKAWAABC', sp.normalize('ＫＡＤＯＫＡＷＡABC'))
    self.assertEqual('▁KADOKAWAABC', sp.Normalize('ＫＡＤＯＫＡＷＡABC'))

    x = sp.Normalize('ＫＡＤＯＫＡＷＡABC', with_offsets=True)
    self.assertEqual('▁KADOKAWAABC', x[0])
    self.assertEqual([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], x[1])

    x = sp.Normalize('ＫＡＤＯＫＡＷＡABC'.encode('utf8'), with_offsets=True)
    self.assertEqual('▁KADOKAWAABC'.encode('utf8'), x[0])
    self.assertEqual(
        [0, 0, 0, 0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 26, 27], x[1]
    )

    self.assertEqual(
        ['▁KADOKAWAABC', '▁平成'], sp.normalize(['ＫＡＤＯＫＡＷＡABC', '㍻'])
    )
    self.assertEqual(
        ['▁KADOKAWAABC', '▁平成'], sp.Normalize(['ＫＡＤＯＫＡＷＡABC', '㍻'])
    )

    x = sp.Normalize(
        ['ＫＡＤＯＫＡＷＡABC'.encode('utf8'), '㍻'.encode('utf8')],
        with_offsets=True,
    )
    self.assertEqual(len(x), 2)
    self.assertEqual('▁KADOKAWAABC'.encode('utf8'), x[0][0])
    self.assertEqual(
        [0, 0, 0, 0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 26, 27], x[0][1]
    )

    x = sp.Normalize(['ＫＡＤＯＫＡＷＡABC', '㍻'], with_offsets=True)
    self.assertEqual(len(x), 2)
    self.assertEqual('▁KADOKAWAABC', x[0][0])
    self.assertEqual([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], x[0][1])

    self.assertEqual('▁平成', x[1][0])
    self.assertEqual([0, 0, 0, 1], x[1][1])

  def test_normalizer(self):
    sp = spm.SentencePieceNormalizer(
        model_file=os.path.join(HERE, 'test_model.model')
    )

    self.assertEqual('KADOKAWAABC', sp.normalize('ＫＡＤＯＫＡＷＡABC'))
    self.assertEqual('KADOKAWAABC', sp.Normalize('ＫＡＤＯＫＡＷＡABC'))

    x = sp.Normalize('ＫＡＤＯＫＡＷＡABC'.encode('utf8'), with_offsets=True)
    self.assertEqual('KADOKAWAABC'.encode('utf8'), x[0])
    self.assertEqual([0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 26, 27], x[1])

    x = sp.Normalize('ＫＡＤＯＫＡＷＡABC', with_offsets=True)
    self.assertEqual('KADOKAWAABC', x[0])
    self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], x[1])

    self.assertEqual(
        ['KADOKAWAABC', '平成'], sp.normalize(['ＫＡＤＯＫＡＷＡABC', '㍻'])
    )
    self.assertEqual(
        ['KADOKAWAABC', '平成'], sp.Normalize(['ＫＡＤＯＫＡＷＡABC', '㍻'])
    )

    x = sp.Normalize(
        ['ＫＡＤＯＫＡＷＡABC'.encode('utf8'), '㍻'.encode('utf8')],
        with_offsets=True,
    )
    self.assertEqual(len(x), 2)
    self.assertEqual('KADOKAWAABC'.encode('utf8'), x[0][0])
    self.assertEqual([0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 26, 27], x[0][1])

    x = sp.Normalize(['ＫＡＤＯＫＡＷＡABC', '㍻'], with_offsets=True)
    self.assertEqual(len(x), 2)
    self.assertEqual('KADOKAWAABC', x[0][0])
    self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], x[0][1])
    self.assertEqual('平成', x[1][0])
    self.assertEqual([0, 0, 1], x[1][1])

    sp = spm.SentencePieceNormalizer(
        model_file=os.path.join(HERE, 'test_model.model'),
        add_dummy_prefix=True,
        escape_whitespaces=True,
        remove_extra_whitespaces=False,
    )
    self.assertEqual('▁hello▁▁world', sp.normalize('hello  world'))

    sp = spm.SentencePieceNormalizer(
        model_file=os.path.join(HERE, 'test_model.model'),
        add_dummy_prefix=True,
        escape_whitespaces=True,
        remove_extra_whitespaces=True,
    )
    self.assertEqual('▁hello▁world', sp.normalize('  hello  world  '))

    sp = spm.SentencePieceNormalizer(
        model_file=os.path.join(HERE, 'test_model.model'),
        add_dummy_prefix=False,
        escape_whitespaces=False,
        remove_extra_whitespaces=True,
    )
    self.assertEqual('hello world', sp.normalize('  hello  world  '))

  def test_normalizer_rule(self):
    sp = spm.SentencePieceNormalizer(rule_name='identity')
    self.assertEqual('ＡＢＣ', sp.Normalize('ＡＢＣ'))

    sp = spm.SentencePieceNormalizer(rule_name='nfkc_cf')
    self.assertEqual('abc', sp.Normalize('ＡＢＣ'))

  def test_normalizer_map(self):
    norm_map = [
        ('foo', 'bar'),
        ('apple', 'orange'),
    ]
    sp = spm.SentencePieceNormalizer(norm_map=norm_map)
    self.assertEqual('bar', sp.Normalize('foo'))
    self.assertEqual('orange', sp.Normalize('apple'))
    self.assertEqual('banana', sp.Normalize('banana'))
    self.assertEqual('bar orange', sp.Normalize('foo apple'))

    decompiled = sp.Decompile()
    self.assertEqual(2, len(decompiled))
    self.assertEqual('apple', decompiled[0][0])
    self.assertEqual('orange', decompiled[0][1])
    self.assertEqual('foo', decompiled[1][0])
    self.assertEqual('bar', decompiled[1][1])

    # Test invalid UTF-8, empty source, identity conversion, and duplicate keys.
    self.assertRaises(ValueError, spm.SentencePieceNormalizer, norm_map=[(b'\xFF', b'bar')])
    self.assertRaises(ValueError, spm.SentencePieceNormalizer, norm_map=[(b'foo', b'\xFF')])
    self.assertRaises(ValueError, spm.SentencePieceNormalizer, norm_map=[('', 'bar')])
    self.assertRaises(ValueError, spm.SentencePieceNormalizer, norm_map=[(b'', b'bar')])
    self.assertRaises(ValueError, spm.SentencePieceNormalizer, norm_map=[('foo', 'foo')])
    self.assertRaises(ValueError, spm.SentencePieceNormalizer, norm_map=[(b'foo', b'foo')])
    self.assertRaises(ValueError, spm.SentencePieceNormalizer, norm_map=[('foo', 'bar'), ('foo', 'baz')])
    self.assertRaises(ValueError, spm.SentencePieceNormalizer, norm_map=[(b'foo', b'bar'), (b'foo', b'baz')])

  def test_trainer_with_normalizer(self):
    tid = threading.get_native_id()
    norm_map = [
        ('foo', 'bar'),
        ('apple', 'orange'),
    ]
    normalizer = spm.SentencePieceNormalizer(norm_map=norm_map, add_dummy_prefix=False, escape_whitespaces=True)

    spm.SentencePieceTrainer.Train(
        input=os.path.join(data_dir, 'botchan.txt'),
        model_prefix=f'm_{tid}',
        vocab_size=100,
        normalizer=normalizer
    )

    sp_norm = spm.SentencePieceNormalizer(model_file=f'm_{tid}.model')
    self.assertEqual('bar', sp_norm.Normalize('foo'))
    self.assertEqual('orange', sp_norm.Normalize('apple'))

    sp = spm.SentencePieceProcessor()
    self.assertTrue(sp.Load(f'm_{tid}.model'))
    pieces = sp.EncodeAsPieces('foo')
    self.assertTrue(len(pieces) > 0)
    self.assertNotEqual(pieces[0][0], '\u2581')

    # Test conflict error
    with self.assertRaises(ValueError):
      spm.SentencePieceTrainer.Train(
          input=os.path.join(data_dir, 'botchan.txt'),
          model_prefix=f'm_{tid}_override',
          vocab_size=100,
          normalizer=normalizer,
          add_dummy_prefix=True
      )


  def test_offset_mapping(self):
    sp = self.sp_
    
    # helper to compute expected offsets from proto in python
    def get_expected_offsets(text, proto):
      expected = []
      text_bytes = text.encode('utf-8')
      for p in proto.pieces:
        start_char = len(text_bytes[:p.begin].decode('utf-8'))
        end_char = len(text_bytes[:p.end].decode('utf-8'))
        expected.append((start_char, end_char))
      return expected

    # 1. Test single text with ASCII characters
    text = "hello world"
    res = sp.encode(text, return_type='offset_mapping')
    self.assertIsInstance(res, dict)
    self.assertIn('ids', res)
    self.assertIn('pieces', res)
    self.assertIn('offsets', res)
    
    self.assertEqual(len(res['ids']), len(res['pieces']))
    self.assertEqual(len(res['ids']), len(res['offsets']))
    
    proto = sp.encode(text, return_type='proto')
    expected = get_expected_offsets(text, proto)
    self.assertEqual(res['offsets'], expected)
    
    for (start, end), p in zip(res['offsets'], proto.pieces):
      self.assertEqual(text[start:end], p.surface)

    # 2. Test multi-byte Unicode characters (East Asian + Emojis)
    unicode_text = "😊吾輩は猫である。😊"
    res_unicode = sp.encode(unicode_text, return_type='offset_mapping')
    proto_unicode = sp.encode(unicode_text, return_type='proto')
    expected_unicode = get_expected_offsets(unicode_text, proto_unicode)
    self.assertEqual(res_unicode['offsets'], expected_unicode)
    
    for (start, end), p in zip(res_unicode['offsets'], proto_unicode.pieces):
      self.assertEqual(unicode_text[start:end], p.surface)

    # 3. Test backward compatibility: return_type='offset_mapping'
    res_compat = sp.encode(unicode_text, return_type='offset_mapping')
    self.assertEqual(res_unicode, res_compat)

    # 4. Test helper method: EncodeAsOffsetMapping
    res_helper = sp.EncodeAsOffsetMapping(unicode_text)
    self.assertEqual(res_unicode, res_helper)

    # 5. Test batch encoding
    texts = ["hello world", "😊吾輩は猫である。😊"]
    res_batch = sp.encode(texts, return_type='offset_mapping')
    self.assertIsInstance(res_batch, list)
    self.assertEqual(len(res_batch), 2)
    self.assertEqual(res_batch[0], sp.encode(texts[0], return_type='offset_mapping'))
    self.assertEqual(res_batch[1], sp.encode(texts[1], return_type='offset_mapping'))
    # 6. Test bytes input (should return raw byte offsets directly)
    byte_text = "😊吾輩は猫である。😊".encode('utf-8')
    res_bytes = sp.encode(byte_text, return_type='offset_mapping')
    proto_bytes = sp.encode(byte_text, return_type='proto')

    expected_byte_offsets = [(p.begin, p.end) for p in proto_bytes.pieces]
    self.assertEqual(res_bytes['offsets'], expected_byte_offsets)

    # Verify pieces are also bytes
    for p in res_bytes['pieces']:
      self.assertIsInstance(p, bytes)

    # Verify we can slice raw bytes using offsets and match proto piece values
    for (start, end), p in zip(res_bytes['offsets'], proto_bytes.pieces):
      self.assertEqual(byte_text[start:end], p.surface.encode('utf-8'))

  def test_decode_offset_mapping(self):
    sp = self.sp_
    
    # helper to compute expected offsets from proto in python
    def get_expected_offsets(text, proto):
      expected = []
      text_bytes = text.encode('utf-8')
      for p in proto.pieces:
        start_char = len(text_bytes[:p.begin].decode('utf-8'))
        end_char = len(text_bytes[:p.end].decode('utf-8'))
        expected.append((start_char, end_char))
      return expected

    # We start with some IDs
    text = "hello world"
    ids = sp.encode(text)
    
    # 1. Test decode IDs to offset mapping (default: return_bytes=False -> Unicode offsets)
    res = sp.decode(ids, return_type='offset_mapping')
    self.assertIsInstance(res, dict)
    self.assertIn('text', res)
    self.assertIn('ids', res)
    self.assertIn('pieces', res)
    self.assertIn('offsets', res)
    self.assertEqual(res['ids'], ids)
    
    decoded_text = sp.decode(ids)
    self.assertEqual(res['text'], decoded_text)
    
    # We can compare against proto
    proto = sp.decode(ids, return_type='proto')
    decoded_text = sp.decode(ids)
    expected = get_expected_offsets(decoded_text, proto)
    self.assertEqual(res['offsets'], expected)
    
    for (start, end), p in zip(res['offsets'], proto.pieces):
      self.assertEqual(decoded_text[start:end], p.surface)

    # 2. Test decode IDs to offset mapping with return_bytes=True (byte offsets)
    res_bytes = sp.decode(ids, return_type='offset_mapping', return_bytes=True)
    self.assertIn('text', res_bytes)
    self.assertIsInstance(res_bytes['text'], bytes)
    decoded_bytes = decoded_text.encode('utf-8')
    self.assertEqual(res_bytes['text'], decoded_bytes)
    expected_bytes = [(p.begin, p.end) for p in proto.pieces]
    self.assertEqual(res_bytes['offsets'], expected_bytes)
    for p in res_bytes['pieces']:
      self.assertIsInstance(p, bytes)
    for (start, end), p in zip(res_bytes['offsets'], proto.pieces):
      self.assertEqual(decoded_bytes[start:end], p.surface.encode('utf-8'))

    # 3. Test decode pieces to offset mapping
    pieces_str = sp.encode(text, return_type=str)
    res_pieces = sp.decode(pieces_str, return_type='offset_mapping')
    self.assertEqual(res['offsets'], res_pieces['offsets'])
    self.assertEqual(res['pieces'], res_pieces['pieces'])
    
    pieces_bytes = [p.encode('utf-8') for p in pieces_str]
    res_pieces_bytes = sp.decode(pieces_bytes, return_type='offset_mapping')
    # Because input was bytes, it should automatically return bytes offsets/pieces
    self.assertEqual(res_bytes['offsets'], res_pieces_bytes['offsets'])
    self.assertEqual(res_bytes['pieces'], res_pieces_bytes['pieces'])

    # 4. Test batch decode
    ids_batch = [sp.encode("hello world"), sp.encode("吾輩は猫である")]
    res_batch = sp.decode(ids_batch, return_type='offset_mapping')
    self.assertIsInstance(res_batch, list)
    self.assertEqual(len(res_batch), 2)
    self.assertEqual(res_batch[0], sp.decode(ids_batch[0], return_type='offset_mapping'))
    self.assertEqual(res_batch[1], sp.decode(ids_batch[1], return_type='offset_mapping'))

    # 5. Test batch decode with return_bytes=True
    res_batch_bytes = sp.decode(ids_batch, return_type='offset_mapping', return_bytes=True)
    self.assertEqual(res_batch_bytes[0], sp.decode(ids_batch[0], return_type='offset_mapping', return_bytes=True))
    self.assertEqual(res_batch_bytes[1], sp.decode(ids_batch[1], return_type='offset_mapping', return_bytes=True))

  def test_decode_return_type_bytes(self):
    sp = self.sp_
    text = "hello world"
    ids = sp.encode(text)
    pieces_str = sp.encode(text, return_type=str)
    pieces_bytes = [p.encode('utf-8') for p in pieces_str]

    # 1. Single ID input
    self.assertEqual(sp.decode(ids, return_type=bytes), text.encode('utf-8'))
    
    # 2. Pieces input (str pieces) -> forces bytes output
    self.assertEqual(sp.decode(pieces_str, return_type=bytes), text.encode('utf-8'))

    # 3. Pieces input (bytes pieces) -> forces bytes output
    self.assertEqual(sp.decode(pieces_bytes, return_type=bytes), text.encode('utf-8'))

    # 4. Batch ID input
    ids2 = sp.encode("吾輩は猫である")
    text2 = sp.decode(ids2)
    ids_batch = [ids, ids2]
    expected_batch = [text.encode('utf-8'), text2.encode('utf-8')]
    self.assertEqual(sp.decode(ids_batch, return_type=bytes), expected_batch)

    # 5. Batch pieces input (str pieces)
    pieces_str2 = sp.encode("吾輩は猫である", return_type=str)
    pieces_str_batch = [pieces_str, pieces_str2]
    expected_pieces_batch = [text.encode('utf-8'), sp.decode(pieces_str2).encode('utf-8')]
    self.assertEqual(sp.decode(pieces_str_batch, return_type=bytes), expected_pieces_batch)

    # 6. Invalid return_bytes combinations in Decode
    with self.assertRaises(ValueError):
      sp.decode(ids, return_type=str, return_bytes=True)
    with self.assertRaises(ValueError):
      sp.decode(ids, return_type=bytes, return_bytes=False)
    with self.assertRaises(ValueError):
      sp.decode(ids, return_type=int, return_bytes=True)

  def test_legacy_out_type_compat(self):
    sp = self.sp_
    text = "hello world"
    ids = sp.encode(text)
    
    # out_type works as alias for return_type
    self.assertEqual(sp.encode(text, out_type=int), ids)
    self.assertEqual(sp.decode(ids, out_type=str), text)
    
    # Cannot specify both
    with self.assertRaises(ValueError):
      sp.encode(text, return_type=int, out_type=int)
    with self.assertRaises(ValueError):
      sp.decode(ids, return_type=str, out_type=str)

  def test_normalizer_rule_tsv(self):
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsv') as f:
      # Map 'A' (0041) to 'B' (0042)
      f.write("0041\t0042\n")
      tsv_path = f.name
      
    try:
      sp = spm.SentencePieceNormalizer(rule_tsv=tsv_path)
      self.assertEqual('BBB', sp.normalize('AAA'))
      self.assertEqual('BBB', sp.normalize('ABA'))
    finally:
      os.unlink(tsv_path)

  def test_normalizer_model_proto(self):
    model_path = os.path.join(HERE, 'test_model.model')
    with open(model_path, 'rb') as f:
      model_proto = f.read()
      
    sp = spm.SentencePieceNormalizer(model_proto=model_proto)
    self.assertEqual('KADOKAWAABC', sp.normalize('ＫＡＤＯＫＡＷＡABC'))

  def test_encode_return_type_explicit(self):
    sp = self.sp_
    text_str = "hello world"
    text_bytes = b"hello world"
    
    # Expected pieces (as str)
    pieces_str = sp.encode(text_str, return_type=str)
    self.assertTrue(all(isinstance(p, str) for p in pieces_str))
    
    # Expected pieces (as bytes)
    pieces_bytes = [p.encode('utf-8') for p in pieces_str]
    
    # 1. return_type=str always returns str
    self.assertEqual(sp.encode(text_str, return_type=str), pieces_str)
    self.assertEqual(sp.encode(text_bytes, return_type=str), pieces_str)
    
    # 2. return_type=bytes always returns bytes
    self.assertEqual(sp.encode(text_str, return_type=bytes), pieces_bytes)
    self.assertEqual(sp.encode(text_bytes, return_type=bytes), pieces_bytes)
    
    # 3. Batch versions
    self.assertEqual(sp.encode([text_str, text_str], return_type=str), [pieces_str, pieces_str])
    self.assertEqual(sp.encode([text_bytes, text_bytes], return_type=str), [pieces_str, pieces_str])
    self.assertEqual(sp.encode([text_str, text_str], return_type=bytes), [pieces_bytes, pieces_bytes])
    self.assertEqual(sp.encode([text_bytes, text_bytes], return_type=bytes), [pieces_bytes, pieces_bytes])

    # 4. NBestEncode
    nbest_str = sp.nbest_encode(text_str, nbest_size=5, return_type=str)
    self.assertTrue(all(isinstance(p, str) for res in nbest_str for p in res))
    nbest_bytes = sp.nbest_encode(text_str, nbest_size=5, return_type=bytes)
    self.assertTrue(all(isinstance(p, bytes) for res in nbest_bytes for p in res))
    self.assertEqual([[p.decode('utf-8') for p in res] for res in nbest_bytes], nbest_str)

    # 5. ParallelEncode
    parallel_str = sp.parallel_encode([text_str], chunk_len=5, num_threads=2, return_type=str)[0]
    self.assertTrue(all(isinstance(p, str) for p in parallel_str))
    parallel_bytes = sp.parallel_encode([text_str], chunk_len=5, num_threads=2, return_type=bytes)[0]
    self.assertTrue(all(isinstance(p, bytes) for p in parallel_bytes))
    self.assertEqual([p.decode('utf-8') for p in parallel_bytes], parallel_str)
    # 6. Offset Mapping with explicit return_bytes
    # Default behavior (None) matches input type
    om_default_str = sp.encode(text_str, return_type='offset_mapping')
    self.assertTrue(all(isinstance(p, str) for p in om_default_str['pieces']))
    self.assertTrue(all(isinstance(o[0], int) and isinstance(o[1], int) for o in om_default_str['offsets']))

    om_default_bytes = sp.encode(text_bytes, return_type='offset_mapping')
    self.assertTrue(all(isinstance(p, bytes) for p in om_default_bytes['pieces']))

    # Force bytes on str input
    om_force_bytes = sp.encode(text_str, return_type='offset_mapping', return_bytes=True)
    self.assertTrue(all(isinstance(p, bytes) for p in om_force_bytes['pieces']))
    self.assertEqual(om_force_bytes['pieces'], om_default_bytes['pieces'])
    self.assertEqual(om_force_bytes['offsets'], om_default_bytes['offsets'])

    # Force str on bytes input
    om_force_str = sp.encode(text_bytes, return_type='offset_mapping', return_bytes=False)
    self.assertTrue(all(isinstance(p, str) for p in om_force_str['pieces']))
    self.assertEqual(om_force_str['pieces'], om_default_str['pieces'])
    self.assertEqual(om_force_str['offsets'], om_default_str['offsets'])

    # Batch version of force bytes/str
    om_batch_force_bytes = sp.encode([text_str], return_type='offset_mapping', return_bytes=True)[0]
    self.assertTrue(all(isinstance(p, bytes) for p in om_batch_force_bytes['pieces']))

    # 7. Invalid return_bytes combinations
    with self.assertRaises(ValueError):
      sp.encode(text_str, return_type=str, return_bytes=True)
    with self.assertRaises(ValueError):
      sp.encode(text_str, return_type=bytes, return_bytes=False)
    with self.assertRaises(ValueError):
      sp.encode(text_str, return_type=int, return_bytes=True)

  def test_native_batch_piece_to_id(self):
    sp = self.sp_
    valid_ids = [3, 4, 5]
    valid_pieces = [sp.IdToPiece(i) for i in valid_ids]

    # Single
    self.assertEqual(sp.PieceToId(valid_pieces[0]), valid_ids[0])
    self.assertEqual(sp.PieceToId("unknown_piece_xyz"), 0)

    # Batch list
    pieces_list = valid_pieces + ["unknown_piece_xyz"]
    ids = sp.PieceToId(pieces_list)
    self.assertIsInstance(ids, list)
    self.assertEqual(ids[:-1], valid_ids)
    self.assertEqual(ids[-1], 0)

    # Batch tuple
    pieces_tuple = tuple(valid_pieces)
    ids_t = sp.PieceToId(pieces_tuple)
    self.assertIsInstance(ids_t, list)
    self.assertEqual(ids_t, valid_ids)

    # Type error
    with self.assertRaises(TypeError):
      sp.PieceToId(123)
    with self.assertRaises(TypeError):
      sp.PieceToId([123])
    with self.assertRaises(TypeError):
      sp.PieceToId(["a", 123])

  def test_native_batch_id_to_piece(self):
    sp = self.sp_
    vocab_size = sp.vocab_size()

    # Single
    piece = sp.IdToPiece(3)
    self.assertIsInstance(piece, str)
    with self.assertRaises(IndexError):
      sp.IdToPiece(-1)
    with self.assertRaises(IndexError):
      sp.IdToPiece(vocab_size)

    # Batch list
    ids = [0, 1, 2, 3]
    pieces = sp.IdToPiece(ids)
    self.assertIsInstance(pieces, list)
    for i, p in zip(ids, pieces):
      self.assertEqual(p, sp.IdToPiece(i))
    with self.assertRaises(IndexError):
      sp.IdToPiece([0, -1, 2])
    with self.assertRaises(IndexError):
      sp.IdToPiece([0, vocab_size])

    # Batch tuple
    ids_t = (0, 1, 2)
    pieces_t = sp.IdToPiece(ids_t)
    self.assertIsInstance(pieces_t, list)

    # Type error
    with self.assertRaises(TypeError):
      sp.IdToPiece("a")
    with self.assertRaises(TypeError):
      sp.IdToPiece(["a"])

  def test_native_batch_other_id_methods(self):
    sp = self.sp_
    vocab_size = sp.vocab_size()
    methods = [
        sp.GetScore,
        sp.IsUnknown,
        sp.IsControl,
        sp.IsUnused,
        sp.IsByte
    ]
    for method in methods:
      # Single
      res = method(0)
      with self.assertRaises(IndexError):
        method(-1)
      with self.assertRaises(IndexError):
        method(vocab_size)
      
      # Batch list
      res_batch = method([0, 1, 2])
      self.assertIsInstance(res_batch, list)
      self.assertEqual(len(res_batch), 3)
      with self.assertRaises(IndexError):
        method([0, -1])
          
      # Batch tuple
      res_tuple = method((0, 1))
      self.assertIsInstance(res_tuple, list)
      self.assertEqual(len(res_tuple), 2)

def suite():
  suite = unittest.TestSuite()
  suite.addTests(unittest.makeSuite(TestSentencepieceProcessor))
  return suite


if __name__ == '__main__':
  unittest.main()
