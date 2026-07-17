import os
import sys
import csv
import io
import importlib.resources
from . import _sentencepiece

def _is_sequence(x):
    return hasattr(x, '__len__') and not isinstance(x, (str, bytes))

_protobuf_loaded = False
_sentencepiece_pb2 = None

def _load_protobuf():
    global _protobuf_loaded, _sentencepiece_pb2
    if not _protobuf_loaded:
        try:
            from sentencepiece import sentencepiece_pb2
            _sentencepiece_pb2 = sentencepiece_pb2
        except ImportError as e:
            raise ImportError(
                "protobuf is required when using out_type='proto'. "
                "Please install protobuf: pip install protobuf"
            ) from e
        _protobuf_loaded = True

_numpy_loaded = False
_numpy = None

def _load_numpy():
    global _numpy_loaded, _numpy
    if not _numpy_loaded:
        try:
            import numpy as np
            _numpy = np
        except ImportError as e:
            raise ImportError(
                "numpy is required when return_type='numpy'. "
                "Please install numpy: pip install numpy"
            ) from e
        _numpy_loaded = True
    return _numpy

def _to_spt(serialized):
    _load_protobuf()
    spt = _sentencepiece_pb2.SentencePieceText()
    spt.ParseFromString(serialized)
    return spt

def _to_nbest_spt(serialized):
    _load_protobuf()
    nbest_spt = _sentencepiece_pb2.NBestSentencePieceText()
    nbest_spt.ParseFromString(serialized)
    return nbest_spt

# Dispatch table for encoding.
# Map of (is_batch, return_type) -> C++ method name.
# - is_batch: True if input is a list of strings, False if input is a single string.
# - return_type: The requested output format (int, str, bytes, proto, etc.).
# Note: 'proto' return type is mapped to '_EncodeAsSerializedProto' because the C++
# wrapper returns a serialized string which is then parsed into a ModelProto in Python.
# 'numpy' return type is mapped to '_EncodeAsBuffer' which returns raw binary buffers.
_ENCODE_DISPATCH = {
    # Single string input
    False: {
        int: '_EncodeAsIds',
        str: '_EncodeAsPieces',
        bytes: '_EncodeAsPieces',
        'serialized_proto': '_EncodeAsSerializedProto',
        'proto': '_EncodeAsSerializedProto',
        'offset_mapping': '_EncodeAsOffsetMapping',
        'numpy': '_EncodeAsBuffer',
    },
    # Batch of strings input
    True: {
        int: '_EncodeAsIdsBatch',
        str: '_EncodeAsPiecesBatch',
        bytes: '_EncodeAsPiecesBatch',
        'serialized_proto': '_EncodeAsSerializedProtoBatch',
        'proto': '_EncodeAsSerializedProtoBatch',
        'offset_mapping': '_EncodeAsOffsetMappingBatch',
        'numpy': '_EncodeAsBufferBatch',
    }
}

# Dispatch table for decoding.
# Map of (input_is_pieces, is_batch) -> dict of {return_type: C++ method name}.
# - input_is_pieces: True if input contains pieces (str/bytes), False if it contains IDs (int).
# - is_batch: True if input is a batch (list of sequences), False if it is a single sequence.
# - return_type: The requested output format (str, bytes, proto, etc.).
# Note: 'proto' return type is mapped to '_Decode*AsSerializedProto' because C++
# returns a serialized string which we then parse into a SentencePieceText proto in Python.
_DECODE_DISPATCH = {
    # IDs input, single sequence (or single ID, which gets wrapped)
    (False, False): {
        str: '_DecodeIds',
        bytes: '_DecodeIdsAsBytes',
        'serialized_proto': '_DecodeIdsAsSerializedProto',
        'offset_mapping': '_DecodeAsOffsetMapping',
    },
    # IDs input, batch of sequences
    (False, True): {
        str: '_DecodeIdsBatch',
        bytes: '_DecodeIdsAsBytesBatch',
        'serialized_proto': '_DecodeIdsAsSerializedProtoBatch',
        'offset_mapping': '_DecodeAsOffsetMappingBatch',
    },
    # Pieces input, single sequence (or single piece, which gets wrapped)
    (True, False): {
        str: '_DecodePieces',
        bytes: '_DecodePiecesAsBytes',
        'serialized_proto': '_DecodePiecesAsSerializedProto',
        'offset_mapping': '_DecodeAsOffsetMapping',
    },
    # Pieces input, batch of sequences
    (True, True): {
        str: '_DecodePiecesBatch',
        bytes: '_DecodePiecesAsBytesBatch',
        'serialized_proto': '_DecodePiecesAsSerializedProtoBatch',
        'offset_mapping': '_DecodeAsOffsetMappingBatch',
    }
}

# Expose data classes and global functions directly
ThreadPool = _sentencepiece.ThreadPool

SetRandomGeneratorSeed = _sentencepiece.SetRandomGeneratorSeed
SetMinLogLevel = _sentencepiece.SetMinLogLevel
SetNBestTimeout = _sentencepiece.SetNBestTimeout
SetDataDir = _sentencepiece.SetDataDir

set_random_generator_seed = SetRandomGeneratorSeed
set_min_log_level = SetMinLogLevel
set_nbest_timeout = SetNBestTimeout

class SentencePieceProcessor:
    def __init__(self,
                 model_file=None,
                 model_proto=None,
                 out_type=None,
                 add_bos=False,
                 add_eos=False,
                 reverse=False,
                 emit_unk_piece=False,
                 enable_sampling=False,
                 nbest_size=-1,
                 alpha=0.1,
                 num_threads=-1,
                 return_type=None):
        self._processor = _sentencepiece.SentencePieceProcessor()
        
        # Resolve return_type / out_type
        self._return_type = self._resolve_return_type(return_type, out_type, default=int)
        self._add_bos = add_bos
        self._add_eos = add_eos
        self._reverse = reverse
        self._emit_unk_piece = emit_unk_piece
        self._enable_sampling = enable_sampling
        self._nbest_size = nbest_size
        self._alpha = alpha
        self._num_threads = num_threads
        if model_file or model_proto:
            self.Load(model_file=model_file, model_proto=model_proto)

    def _resolve_and_validate_return_type(self, return_type):
        if return_type is None:
            return_type = self._return_type
        if return_type == 'immutable_proto':
            raise ValueError("immutable_proto is deprecated and no longer supported. Use return_type='proto' instead.")
        
        valid_types = (int, str, bytes, 'proto', 'serialized_proto', 'numpy', 'offset_mapping')
        if return_type not in valid_types:
            raise ValueError("Invalid return_type: {}. Must be one of {}".format(return_type, valid_types))
        return return_type

    def _resolve_return_type(self, return_type, out_type, default=None):
        if out_type is not None:
            if return_type is not None:
                raise ValueError("Cannot specify both out_type and return_type. Use return_type instead.")
            return_type = out_type
        if return_type is None:
            return_type = default if default is not None else self._return_type
        return self._resolve_and_validate_return_type(return_type)

    @classmethod
    def from_file(cls, model_file, **kwargs):
        """Creates a SentencePieceProcessor by loading a model file from disk."""
        if 'model_file' in kwargs or 'model_proto' in kwargs:
            raise ValueError("model_file and model_proto cannot be passed via kwargs when using from_file classmethod.")
        processor = cls(**kwargs)
        processor.LoadFromFile(model_file)
        return processor

    @classmethod
    def from_proto(cls, model_proto, **kwargs):
        """Creates a SentencePieceProcessor by loading a serialized model proto from memory."""
        if 'model_file' in kwargs or 'model_proto' in kwargs:
            raise ValueError("model_file and model_proto cannot be passed via kwargs when using from_proto classmethod.")
        processor = cls(**kwargs)
        processor.LoadFromSerializedProto(model_proto)
        return processor

    def Load(self, model_file=None, model_proto=None):
        if model_file and model_proto:
            raise ValueError('model_file and model_proto must be exclusive.')
        if model_proto:
            return self.LoadFromSerializedProto(model_proto)
        if model_file:
            return self.LoadFromFile(model_file)
        raise ValueError('Either model_file or model_proto must be specified.')

    def LoadFromFile(self, filename):
        return self._processor.LoadFromFile(filename)

    def LoadFromSerializedProto(self, serialized):
        if hasattr(serialized, 'SerializeToString'):
            serialized = serialized.SerializeToString()
        return self._processor.LoadFromSerializedProto(serialized)

    def Encode(self,
               input,
               add_bos=None,
               add_eos=None,
               reverse=None,
               emit_unk_piece=None,
               enable_sampling=None,
               nbest_size=None,
               alpha=None,
               num_threads=None,
               thread_pool=None,
               return_type=None,
               out_type=None,
               return_bytes=None):
        return_type = self._resolve_return_type(return_type, out_type)
        if return_bytes is not None and return_type != 'offset_mapping':
            raise ValueError("return_bytes is only supported when return_type='offset_mapping'")
        if add_bos is None:
            add_bos = self._add_bos
        if add_eos is None:
            add_eos = self._add_eos
        if reverse is None:
            reverse = self._reverse
        if emit_unk_piece is None:
            emit_unk_piece = self._emit_unk_piece
        if enable_sampling is None:
            enable_sampling = self._enable_sampling
        if nbest_size is None:
            nbest_size = self._nbest_size
        if alpha is None:
            alpha = self._alpha
        if num_threads is None:
            num_threads = self._num_threads

        if enable_sampling is True and (nbest_size is None or nbest_size == 0 or
                                         nbest_size == 1 or alpha is None):
            raise ValueError(
                'When enable_sampling is True, We must specify "nbest_size > 1" or "nbest_size = -1", '
                'and "alpha". "nbest_size" is enabled only on unigram mode ignored in BPE-dropout. '
                'when "nbest_size = -1" , this method samples from all candidates on the lattice '
                'instead of nbest segmentations.'
            )

        if num_threads is None or not isinstance(num_threads, int):
            raise TypeError('num_threads must be int')

        is_batch = isinstance(input, list)

        # Dispatch to C++ method
        methods = _ENCODE_DISPATCH.get(is_batch)
        if not methods or return_type not in methods:
            raise ValueError('unknown return_type={}'.format(return_type))

        method_name = methods[return_type]
        method = getattr(self._processor, method_name)

        # Build arguments dynamically.
        # CRITICAL: The order of elements in `args` must exactly match the positional
        # argument order expected by the pybind11 C++ methods in sentencepiece_pybind.cc.
        is_pieces_method = return_type in (str, bytes)
        has_unk = is_pieces_method or return_type in ('serialized_proto', 'proto', 'offset_mapping')
        if is_batch:
            args = [input, num_threads, thread_pool, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse]
        else:
            args = [input, enable_sampling, nbest_size, alpha, add_bos, add_eos, reverse]
        if has_unk:
            args.append(emit_unk_piece)
        if is_pieces_method:
            return_bytes_val = (return_type is bytes)
            args.append(return_bytes_val)
        elif return_type == 'offset_mapping':
            if return_bytes is None:
                # Default to matching input type
                return_bytes_val = isinstance(input, bytes) or (isinstance(input, list) and input and isinstance(input[0], bytes))
            else:
                return_bytes_val = return_bytes
            args.append(return_bytes_val)

        raw_res = method(*args)

        # Post-process results:
        # C++ returns raw types (like buffers or serialized strings) to keep
        # the binary extension independent of NumPy and Protobuf libraries.
        # We wrap/deserialize them here in Python to return high-level types.
        if return_type == 'numpy':
            # Convert raw C++ buffers to numpy arrays
            np = _load_numpy()
            if is_batch:
                return [np.frombuffer(buf, dtype=np.int32) for buf in raw_res]
            return np.frombuffer(raw_res, dtype=np.int32)
        elif return_type == 'proto':
            # Deserialize C++ serialized protobuf bytes into Python Protobuf objects
            if is_batch:
                return [_to_spt(s) for s in raw_res]
            return _to_spt(raw_res)

        return raw_res

    def EncodeAsPieces(self, input, **kwargs):
        return self.Encode(input=input, return_type=str, **kwargs)

    def EncodeAsIds(self, input, **kwargs):
        return self.Encode(input=input, return_type=int, **kwargs)

    def EncodeAsNumpy(self, input, **kwargs):
        return self.Encode(input=input, return_type='numpy', **kwargs)



    def EncodeAsProto(self, input, **kwargs):
        return self.Encode(input=input, return_type='proto', **kwargs)

    def EncodeAsOffsetMapping(self, input, **kwargs):
        return self.Encode(input=input, return_type='offset_mapping', **kwargs)



    def SampleEncodeAsPieces(self, input, nbest_size=None, alpha=None, **kwargs):
        return self.Encode(input=input, nbest_size=nbest_size, alpha=alpha,
                           return_type=str, enable_sampling=True, **kwargs)

    def SampleEncodeAsIds(self, input, nbest_size=None, alpha=None, **kwargs):
        return self.Encode(input=input, nbest_size=nbest_size, alpha=alpha,
                           return_type=int, enable_sampling=True, **kwargs)

    def SampleEncodeAsNumpy(self, input, nbest_size=None, alpha=None, **kwargs):
        return self.Encode(input=input, nbest_size=nbest_size, alpha=alpha,
                           return_type='numpy', enable_sampling=True, **kwargs)



    def SampleEncodeAsProto(self, input, nbest_size=None, alpha=None, **kwargs):
        return self.Encode(input=input, nbest_size=nbest_size, alpha=alpha,
                           return_type='proto', enable_sampling=True, **kwargs)



    def NBestEncode(self,
                    input,
                    add_bos=None,
                    add_eos=None,
                    reverse=None,
                    emit_unk_piece=None,
                    nbest_size=None,
                    return_type=None,
                    out_type=None):
        return_type = self._resolve_return_type(return_type, out_type)
        if add_bos is None:
            add_bos = self._add_bos
        if add_eos is None:
            add_eos = self._add_eos
        if reverse is None:
            reverse = self._reverse
        if emit_unk_piece is None:
            emit_unk_piece = self._emit_unk_piece
        if nbest_size is None:
            nbest_size = self._nbest_size

        if nbest_size <= 0:
            nbest_size = 1

        if return_type == 'numpy':
            np = _load_numpy()

            def _encode_np(text):
                raw_buffers = self._processor._NBestEncodeAsBuffer(
                    text, nbest_size, add_bos, add_eos, reverse)
                return [np.frombuffer(buf, dtype=np.int32) for buf in raw_buffers]

            if isinstance(input, list):
                return [_encode_np(n) for n in input]
            return _encode_np(input)

        def _encode(text):
            if return_type is int:
                return self._processor._NBestEncodeAsIds(
                    text, nbest_size, add_bos, add_eos, reverse)
            if return_type is str or return_type is bytes:
                return_bytes = (return_type is bytes)
                return self._processor._NBestEncodeAsPieces(
                    text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece, return_bytes)
            if return_type == 'serialized_proto':
                return self._processor._NBestEncodeAsSerializedProto(
                    text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece)
            if return_type == 'proto':
                serialized = self._processor._NBestEncodeAsSerializedProto(
                    text, nbest_size, add_bos, add_eos, reverse, emit_unk_piece)
                return _to_nbest_spt(serialized)
            raise ValueError('unknown return_type={}'.format(return_type))

        if isinstance(input, list):
            return [_encode(n) for n in input]
        return _encode(input)

    def NBestEncodeAsPieces(self, input, nbest_size=None, **kwargs):
        return self.NBestEncode(input=input, nbest_size=nbest_size, return_type=str, **kwargs)

    def NBestEncodeAsIds(self, input, nbest_size=None, **kwargs):
        return self.NBestEncode(input=input, nbest_size=nbest_size, return_type=int, **kwargs)

    def NBestEncodeAsNumpy(self, input, nbest_size=None, **kwargs):
        return self.NBestEncode(input=input, nbest_size=nbest_size, return_type='numpy', **kwargs)



    def NBestEncodeAsProto(self, input, nbest_size=None, **kwargs):
        return self.NBestEncode(input=input, nbest_size=nbest_size, return_type='proto', **kwargs)





    def ParallelEncode(self,
                       input,
                       add_bos=None,
                       add_eos=None,
                       reverse=None,
                       emit_unk_piece=None,
                       chunk_len=None,
                       num_threads=None,
                       thread_pool=None,
                       return_type=None,
                       out_type=None):
        return_type = self._resolve_return_type(return_type, out_type)
        if add_bos is None:
            add_bos = self._add_bos
        if add_eos is None:
            add_eos = self._add_eos
        if reverse is None:
            reverse = self._reverse
        if emit_unk_piece is None:
            emit_unk_piece = self._emit_unk_piece
        if num_threads is None:
            num_threads = self._num_threads

        if num_threads is None or not isinstance(num_threads, int):
            raise TypeError('num_threads must be int')
        if chunk_len is None or not isinstance(chunk_len, int) or chunk_len <= 0:
            raise ValueError('chunk_len must be positive.')

        if return_type == 'numpy':
            np = _load_numpy()

            def _encode_np(text):
                buf = self._processor._ParallelEncodeAsBuffer(
                    text, chunk_len, num_threads, thread_pool, add_bos, add_eos, reverse)
                return np.frombuffer(buf, dtype=np.int32)

            if isinstance(input, list):
                return [_encode_np(n) for n in input]
            return _encode_np(input)

        def _encode(text):
            if return_type is int:
                return self._processor._ParallelEncodeAsIds(
                    text, chunk_len, num_threads, thread_pool, add_bos, add_eos, reverse)
            if return_type is str or return_type is bytes:
                return_bytes = (return_type is bytes)
                return self._processor._ParallelEncodeAsPieces(
                    text, chunk_len, num_threads, thread_pool, add_bos, add_eos, reverse, emit_unk_piece, return_bytes)
            if return_type == 'serialized_proto':
                return self._processor._ParallelEncodeAsSerializedProto(
                    text, chunk_len, num_threads, thread_pool, add_bos, add_eos, reverse, emit_unk_piece)
            if return_type == 'proto':
                serialized = self._processor._ParallelEncodeAsSerializedProto(
                    text, chunk_len, num_threads, thread_pool, add_bos, add_eos, reverse, emit_unk_piece)
                return _to_spt(serialized)
            raise ValueError('unknown return_type={}'.format(return_type))

        if isinstance(input, list):
            return [_encode(n) for n in input]
        return _encode(input)

    def ParallelEncodeAsPieces(self, input, **kwargs):
        return self.ParallelEncode(input=input, return_type=str, **kwargs)

    def ParallelEncodeAsIds(self, input, **kwargs):
        return self.ParallelEncode(input=input, return_type=int, **kwargs)

    def ParallelEncodeAsNumpy(self, input, **kwargs):
        return self.ParallelEncode(input=input, return_type='numpy', **kwargs)



    def ParallelEncodeAsProto(self, input, **kwargs):
        return self.ParallelEncode(input=input, return_type='proto', **kwargs)



    def Decode(self, input, return_type=None, out_type=None, num_threads=None, thread_pool=None, return_bytes=None):
        return_type = self._resolve_return_type(return_type, out_type, default=str)
        if return_bytes is not None and return_type != 'offset_mapping':
            raise ValueError("return_bytes is only supported when return_type='offset_mapping'")
        if num_threads is None:
            num_threads = self._num_threads
        if num_threads is None or not isinstance(num_threads, int):
            raise TypeError('num_threads must be int')

        return_bytes_val = bool(return_bytes) if return_type == 'offset_mapping' else False
        cpp_return_type = 'serialized_proto' if return_type == 'proto' else return_type
        raw_val = self._decode_raw(input, cpp_return_type, num_threads, thread_pool, return_bytes=return_bytes_val)

        if return_type == 'proto':
            if isinstance(raw_val, list):
                return [_to_spt(s) for s in raw_val]
            return _to_spt(raw_val)
        return raw_val

    def _decode_raw(self, input, return_type, num_threads, thread_pool, return_bytes=False):
        # Normalize empty input to an empty id sequence and fall through to the
        # regular dispatch below, so that the result comes from the C++ method
        # and is typed according to return_type (e.g. bytes for
        # return_type=bytes or 'serialized_proto'), matching non-empty inputs.
        if input is None:
            input = []
        elif isinstance(input, (str, bytes)) and len(input) == 0:
            input = []

        # Helper to determine if a sequence contains pieces (str/bytes)
        def _is_pieces(seq):
            if len(seq) == 0:
                return False
            if hasattr(seq, 'dtype'):  # NumPy array is never pieces
                return False
            return isinstance(seq[0], (str, bytes))

        # Normalize single element to single-element list
        is_single = not _is_sequence(input)
        if is_single:
            input = [input]

        # Determine if it is a batch
        if hasattr(input, 'ndim'):
            is_batch = input.ndim > 1
        else:
            is_batch = len(input) > 0 and _is_sequence(input[0])

        # Determine if input contains pieces (str/bytes)
        if is_batch:
            input_is_pieces = _is_pieces(input[0])
        else:
            input_is_pieces = _is_pieces(input)

        # Dispatch to the correct C++ method
        methods = _DECODE_DISPATCH.get((input_is_pieces, is_batch))
        if not methods or return_type not in methods:
            raise ValueError('unknown return_type={} or input type'.format(return_type))

        method_name = methods[return_type]
        method = getattr(self._processor, method_name)

        if return_type == 'offset_mapping':
            if is_batch:
                return method(input, num_threads, thread_pool, return_bytes)
            else:
                return method(input, return_bytes)
        else:
            if is_batch:
                return method(input, num_threads, thread_pool)
            else:
                return method(input)

    def DecodePieces(self, input, return_type=str, **kwargs):
        return self.Decode(input=input, return_type=return_type, **kwargs)

    def DecodeIds(self, input, return_type=str, **kwargs):
        return self.Decode(input=input, return_type=return_type, **kwargs)



    def DecodePiecesAsProto(self, input, **kwargs):
        return self.Decode(input=input, return_type='proto', **kwargs)

    def DecodeIdsAsProto(self, input, **kwargs):
        return self.Decode(input=input, return_type='proto', **kwargs)





    def Normalize(self, input, with_offsets=None):
        if isinstance(input, list):
            if with_offsets:
                return [self._processor._NormalizeWithOffsets(x) for x in input]
            return [self._processor._Normalize(x) for x in input]
        if with_offsets:
            return self._processor._NormalizeWithOffsets(input)
        return self._processor._Normalize(input)



    def GetPieceSize(self):
        return self._processor.GetPieceSize()

    def PieceToId(self, piece):
        return self._processor.PieceToId(piece)

    def IdToPiece(self, id):
        return self._processor.IdToPiece(id)

    def GetScore(self, id):
        return self._processor.GetScore(id)

    def IsUnknown(self, id):
        return self._processor.IsUnknown(id)

    def IsControl(self, id):
        return self._processor.IsControl(id)

    def IsUnused(self, id):
        return self._processor.IsUnused(id)

    def IsByte(self, id):
        return self._processor.IsByte(id)

    def unk_id(self):
        return self._processor.unk_id()

    def bos_id(self):
        return self._processor.bos_id()

    def eos_id(self):
        return self._processor.eos_id()

    def pad_id(self):
        return self._processor.pad_id()

    def serialized_model_proto(self):
        return self._processor.serialized_model_proto()

    piece_size = GetPieceSize
    vocab_size = GetPieceSize

    def __getstate__(self):
        return self.serialized_model_proto()

    def __getstate_for_pickle__(self):
        return self.serialized_model_proto()

    def __setstate__(self, serialized_model_proto):
        self.__init__()
        self.LoadFromSerializedProto(serialized_model_proto)

    def __len__(self):
        return self.GetPieceSize()

    def __getitem__(self, piece):
        return self.PieceToId(piece)


class _LogStream:
    def __init__(self, ostream=None):
        self.ostream = ostream
        if self.ostream is not None:
            self.orig_stream_fileno = sys.stderr.fileno()

    def __enter__(self):
        if self.ostream is not None:
            self.orig_stream_dup = os.dup(self.orig_stream_fileno)
            os.dup2(self.ostream.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        if self.ostream is not None:
            os.close(self.orig_stream_fileno)
            os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
            os.close(self.orig_stream_dup)
            self.ostream.close()


class SentencePieceTrainer:
    @staticmethod
    def _Train(arg=None, **kwargs):
        if arg is not None and isinstance(arg, str):
            return _sentencepiece.SentencePieceTrainer._TrainFromString(arg)

        def _encode(value):
            if isinstance(value, list):
                f = io.StringIO()
                writer = csv.writer(f, lineterminator='')
                writer.writerow([str(v) for v in value])
                return f.getvalue()
            else:
                return str(value)

        sentence_iterator = None
        model_writer = None
        normalizer = None
        new_kwargs = {}
        for key, value in kwargs.items():
            if key in ['sentence_iterator', 'sentence_reader']:
                sentence_iterator = value
            elif key in ['model_writer']:
                model_writer = value
            elif key in ['normalizer']:
                normalizer = value
            else:
                new_kwargs[key] = _encode(value)

        if normalizer:
            new_kwargs['_serialized_normalizer_spec'] = normalizer.serialized_normalizer_spec()

        if model_writer:
            if sentence_iterator:
                model_proto = _sentencepiece.SentencePieceTrainer._TrainFromMap4(new_kwargs, sentence_iterator)
            else:
                model_proto = _sentencepiece.SentencePieceTrainer._TrainFromMap3(new_kwargs)
            model_writer.write(model_proto)
        else:
            if sentence_iterator:
                return _sentencepiece.SentencePieceTrainer._TrainFromMap2(new_kwargs, sentence_iterator)
            else:
                return _sentencepiece.SentencePieceTrainer._TrainFromMap(new_kwargs)
        return None

    @staticmethod
    def Train(arg=None, logstream=None, **kwargs):
        with _LogStream(ostream=logstream):
            return SentencePieceTrainer._Train(arg=arg, **kwargs)


class SentencePieceNormalizer:
    def __init__(self,
                 model_file=None,
                 model_proto=None,
                 normalizer_spec=None,
                 rule_tsv=None,
                 rule_name=None,
                 norm_map=None,
                 add_dummy_prefix=False,
                 escape_whitespaces=False,
                 remove_extra_whitespaces=False):
        self._normalizer = _sentencepiece.SentencePieceNormalizer()

        if model_file:
            self._normalizer.LoadFromFile(model_file)
        elif model_proto:
            if hasattr(model_proto, 'SerializeToString'):
                model_proto = model_proto.SerializeToString()
            self._normalizer.LoadFromSerializedProto(model_proto)
        elif normalizer_spec:
            if hasattr(normalizer_spec, 'SerializeToString'):
                normalizer_spec = normalizer_spec.SerializeToString()
            self._normalizer.LoadFromSerializedNormalizerSpec(normalizer_spec)
        elif rule_tsv:
            self._normalizer.LoadFromRuleTSV(rule_tsv)
        elif rule_name:
            self._normalizer.LoadFromRuleName(rule_name)
        elif norm_map:
            self._normalizer.LoadFromMap(norm_map)
        else:
            raise ValueError('no model is specified')

        self._normalizer._SetProtoField('add_dummy_prefix', add_dummy_prefix)
        self._normalizer._SetProtoField('escape_whitespaces', escape_whitespaces)
        self._normalizer._SetProtoField('remove_extra_whitespaces', remove_extra_whitespaces)

    def Decompile(self):
        return self._normalizer.Decompile()

    def serialized_model_proto(self):
        return self._normalizer.serialized_model_proto()

    def serialized_normalizer_spec(self):
        return self._normalizer.serialized_normalizer_spec()

    def Normalize(self, input, with_offsets=None):
        if isinstance(input, list):
            if with_offsets:
                return [self._normalizer._NormalizeWithOffsets(x) for x in input]
            return [self._normalizer._Normalize(x) for x in input]
        if with_offsets:
            return self._normalizer._NormalizeWithOffsets(input)
        return self._normalizer._Normalize(input)

    def __getstate__(self):
        return self._normalizer.serialized_model_proto()

    def __setstate__(self, serialized_model_proto):
        self._normalizer = _sentencepiece.SentencePieceNormalizer()
        self._normalizer.LoadFromSerializedProto(serialized_model_proto)


# Helpers for batchnize and snake_case (replicated from SWIG skeleton)
def _add_snake_case(classname):
    import re
    snake_map = {}
    for k, v in classname.__dict__.items():
        if re.match(r'^[A-Z]+', k):
            snake = re.sub(r'(?<!^)(?=[A-Z])', '_', k).lower().replace('n_best', 'nbest')
            snake_map[snake] = v
    for k, v in snake_map.items():
        setattr(classname, k, v)





# Run batchnize and snake_case on classes
SentencePieceProcessor.Tokenize = SentencePieceProcessor.Encode
SentencePieceProcessor.Detokenize = SentencePieceProcessor.Decode



_add_snake_case(SentencePieceProcessor)
_add_snake_case(SentencePieceTrainer)
_add_snake_case(SentencePieceNormalizer)

# Global variables and initialization
from ._version import __version__

try:
    SetDataDir(os.path.join(str(importlib.resources.files('sentencepiece')), 'package_data'))
except Exception:
    pass
