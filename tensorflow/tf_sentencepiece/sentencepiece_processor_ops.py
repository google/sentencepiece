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

r"""Ops for SentencePiece Encoding/Decoding."""

# TODO(taku):  Implements n-best output

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

_gen_sentencepiece_processor_op = tf.load_op_library(
    os.path.join(os.path.dirname(__file__), '_sentencepiece_processor_ops.so'))


def piece_size(model_file=None, model_proto=None, name=None):
  """Returns the piece size (vocabulary size).

  Args:
    model_file: The sentencepiece model file path.
    model_proto: The sentencepiece model serialized proto.
                 Either `model_file` or `model_proto` must be set.
    name: The name argument that is passed to the op function.
  Returns:
    A scalar representing the vocabulary size.
  """

  return _gen_sentencepiece_processor_op.sentencepiece_get_piece_size(
      model_file=model_file, model_proto=model_proto, name=name)


def piece_to_id(input, model_file=None, model_proto=None, name=None):
  """Converts piece into vocabulary id.

  Args:
    input: An arbitrary tensor of string.
    model_file: The sentencepiece model file path.
    model_proto: The sentencepiece model serialized proto.
                 Either `model_file` or `model_proto` must be set.
    name: The name argument that is passed to the op function.
  Returns:
    A tensor of int32 with the same shape as input.
  """

  return _gen_sentencepiece_processor_op.sentencepiece_piece_to_id(
      input, model_file=model_file, model_proto=model_proto, name=name)


def id_to_piece(input, model_file=None, model_proto=None, name=None):
  """Converts vocabulary id into piece.

  Args:
    input: An arbitrary tensor of int32.
    model_file: The sentencepiece model file path.
    model_proto: The sentencepiece model serialized proto.
                 Either `model_file` or `model_proto` must be set.
    name: The name argument that is passed to the op function.
  Returns:
    A tensor of string with the same shape as input.
  """

  return _gen_sentencepiece_processor_op.sentencepiece_id_to_piece(
      input, model_file=model_file, model_proto=model_proto, name=name)


def is_unknown(input, model_file=None, model_proto=None, name=None):
  """Returns true if input id is unknown piece.

  Args:
    input: An arbitrary tensor of int32.
    model_file: The sentencepiece model file path.
    model_proto: The sentencepiece model serialized proto.
                 Either `model_file` or `model_proto` must be set.
    name: The name argument that is passed to the op function.
  Returns:
    A tensor of bool with the same shape as input.
  """

  return _gen_sentencepiece_processor_op.sentencepiece_get_piece_type(
      input, model_file=model_file, model_proto=model_proto, name=name,
      piece_type=0)


def is_control(input, model_file=None, model_proto=None, name=None):
  """Returns true if input id is control piece.

  Args:
    input: An arbitrary tensor of int32.
    model_file: The sentencepiece model file path.
    model_proto: The sentencepiece model serialized proto.
                 Either `model_file` or `model_proto` must be set.
    name: The name argument that is passed to the op function.
  Returns:
    A tensor of bool with the same shape as input.
  """

  return _gen_sentencepiece_processor_op.sentencepiece_get_piece_type(
      input, model_file=model_file, model_proto=model_proto, name=name,
      piece_type=1)


def is_unused(input, model_file=None, model_proto=None, name=None):
  """Returns true if input id is unused piece.

  Args:
    input: An arbitrary tensor of int32.
    model_file: The sentencepiece model file path.
    model_proto: The sentencepiece model serialized proto.
                 Either `model_file` or `model_proto` must be set.
    name: The name argument that is passed to the op function.
  Returns:
    A tensor of bool with the same shape as input.
  """

  return _gen_sentencepiece_processor_op.sentencepiece_get_piece_type(
      input, model_file=model_file, model_proto=model_proto, name=name,
      piece_type=2)


def encode_dense(input_sentences, nbest_size=0, alpha=1.0,
                 model_file=None, model_proto=None,
                 reverse=False, add_bos=False, add_eos=False,
                 out_type=tf.int32, name=None):
  """Encodes sentences into pieces in dense tensor format.

  Args:
    input_sentences: A 1D string tensor of arbitrary size holding the raw
                     text of input sentences.
    nbest_size: A scalar or 1D tensor for sampling.
                nbest_size = {0,1}: No sampling is performed.
                nbest_size > 1: samples from the nbest_size results.
                nbest_size < 0: assuming that nbest_size is infinite
                and samples from the all hypothesis (lattice) using
                forward-filtering-and-backward-sampling algorithm.
    alpha: A scalar or 1D tensor for a moothing parameter.
           Inverse temparature for probablity rescaling.
    model_file: The sentencepiece model file path.
    model_proto: The sentencepiece model serialized proto.
                 Either `model_file` or `model_proto` must be set.
    reverse: Reverses the tokenized sequence (Default = false)
    add_bos: Add <s> to the result (Default = false)
    add_eos: Add </s> to the result (Default = false)
             <s>/</s> is added after reversing (if enabled).
    out_type: output type. tf.int32 or tf.string (Default = tf.int32)
              Setting tf.int32 directly encodes the string into an id sequence.
    name: The name argument that is passed to the op function.
  Returns:
    pieces: A dense 2D tensor representing the tokenized sentences.
    sequence_length: A 1D tensor representing the length of pieces.
  """

  return _gen_sentencepiece_processor_op.sentencepiece_encode_dense(
      input_sentences, nbest_size=nbest_size, alpha=alpha,
      model_file=model_file, model_proto=model_proto,
      reverse=reverse, add_bos=add_bos, add_eos=add_eos,
      out_type=out_type, name=name)


def encode_sparse(input_sentences, nbest_size=0, alpha=1.0,
                  model_file=None, model_proto=None,
                  reverse=False, add_bos=False, add_eos=False,
                  out_type=tf.int32, name=None):
  """Encodes sentences into pieces in sparse tensor format.

  Args:
    input_sentences: A 1D string tensor of arbitrary size holding the raw
                     text of input sentences.
    nbest_size: A scalar or 1D tensor for sampling.
                nbest_size = {0,1}: No sampling is performed.
                nbest_size > 1: samples from the nbest_size results.
                nbest_size < 0: assuming that nbest_size is infinite
                and samples from the all hypothesis (lattice) using
                forward-filtering-and-backward-sampling algorithm.
    alpha: A scalar or 1D tensor for a moothing parameter.
           Inverse temparature for probablity rescaling.
    model_file: The sentencepiece model file path.
    model_proto: The sentencepiece model serialized proto.
                 Either `model_file` or `model_proto` must be set.
    reverse: Reverses the tokenized sequence (Default = false)
    add_bos: Add <s> to the result (Default = false)
    add_eos: Add </s> to the result (Default = false)
             <s>/</s> is added after reversing (if enabled).
    out_type: output type. tf.int32 or tf.string (Default = tf.int32)
              Setting tf.int32 directly encodes the string into an id sequence.
    name: The name argument that is passed to the op function.

  Returns:
    pieces: A sparse 2D tensor representing the tokenized sentences.
  """

  indices, values, dense_shape = (
      _gen_sentencepiece_processor_op.sentencepiece_encode_sparse(
          input_sentences, nbest_size=nbest_size, alpha=alpha,
          model_file=model_file, model_proto=model_proto,
          reverse=reverse, add_bos=add_bos, add_eos=add_eos,
          out_type=out_type, name=name))
  return tf.SparseTensor(indices, values, dense_shape)


def decode(pieces, sequence_length, model_file=None, model_proto=None,
           reverse=False, name=None):
  """Decode pieces into postproecssed text.

  Args:
    pieces: A 2D int32 or string tensor [batch_size x max_length] of
            encoded sequences.
    sequence_length: A 1D int32 tensor [batch_size] representing the
                   length of pieces.
    model_file: The sentencepiece model file path.
    model_proto: The sentencepiece model serialized proto.
                 Either `model_file` or `model_proto` must be set.
    reverse: Reverses the tokenized sequence (Default = false)
    name: The name argument that is passed to the op function.

  Returns:
    text: A 1D string tensor of decoded string.
  """

  return _gen_sentencepiece_processor_op.sentencepiece_decode(
      pieces, sequence_length, model_file=model_file,
      model_proto=model_proto, reverse=reverse, name=name)

# Adds an alias for encode_dense. Accepts the `encode` function.
encode = encode_dense
sparse_encode = encode_sparse
dense_encode = encode_dense


tf.NotDifferentiable('SentencepieceGetPieceSize')
tf.NotDifferentiable('SentencepieceIdToPiece')
tf.NotDifferentiable('SentencepiecePieceToId')
tf.NotDifferentiable('SentencepieceGetPieceType')
tf.NotDifferentiable('SentencepieceEncodeDense')
tf.NotDifferentiable('SentencepieceEncodeSparse')
tf.NotDifferentiable('SentencepieceDecode')
