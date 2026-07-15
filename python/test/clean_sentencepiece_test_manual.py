import os
import unittest
import pickle
import io
import tempfile
import numpy as np
import sentencepiece as spm

try:
  from sentencepiece import sentencepiece_pb2
  has_protobuf = True
except ImportError:
  has_protobuf = False

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, 'test_model.model')
JA_MODEL_PATH = os.path.join(HERE, 'test_ja_model.model')

import threading
import time
import sys

class HeartbeatCounter:
  def __init__(self):
    self.count = 0
    self._lock = threading.Lock()

  def increment(self):
    with self._lock:
      self.count += 1

  def get_count(self):
    with self._lock:
      return self.count

def background_heartbeat(stop_event, counter):
  while not stop_event.is_set():
    counter.increment()
    time.sleep(0.01)

class TestSentencePieceProcessorClean(unittest.TestCase):

  def setUp(self):
    self.model_bytes = None
    with open(MODEL_PATH, 'rb') as f:
      self.model_bytes = f.read()

  def test_init_and_load(self):
    # Test default init
    sp = spm.SentencePieceProcessor()
    self.assertIsNotNone(sp)
    
    # Test Load from file
    self.assertTrue(sp.Load(MODEL_PATH))
    self.assertEqual(sp.vocab_size(), 1000)

    # Test Load from proto bytes
    sp2 = spm.SentencePieceProcessor()
    self.assertTrue(sp2.Load(model_proto=self.model_bytes))
    self.assertEqual(sp2.vocab_size(), 1000)

    # Test LoadFromFile
    sp3 = spm.SentencePieceProcessor()
    self.assertTrue(sp3.LoadFromFile(MODEL_PATH))

    # Test LoadFromSerializedProto
    sp4 = spm.SentencePieceProcessor()
    self.assertTrue(sp4.LoadFromSerializedProto(self.model_bytes))

    # Test classmethod factories
    sp5 = spm.SentencePieceProcessor.from_file(MODEL_PATH)
    self.assertEqual(sp5.vocab_size(), 1000)

    sp6 = spm.SentencePieceProcessor.from_proto(self.model_bytes)
    self.assertEqual(sp6.vocab_size(), 1000)

    # Error cases for init/load
    with self.assertRaises(ValueError):
      spm.SentencePieceProcessor(model_file=MODEL_PATH, model_proto=self.model_bytes)
    
    # Trigger ValueError in from_file by passing model_proto in kwargs
    with self.assertRaises(ValueError):
      spm.SentencePieceProcessor.from_file(MODEL_PATH, model_proto=self.model_bytes)

    # Trigger ValueError in from_proto by passing model_file in kwargs
    with self.assertRaises(ValueError):
      spm.SentencePieceProcessor.from_proto(self.model_bytes, model_file=MODEL_PATH)

    with self.assertRaises(ValueError):
      sp = spm.SentencePieceProcessor()
      sp.Load() # neither

    with self.assertRaises(ValueError):
      sp = spm.SentencePieceProcessor()
      sp.Load(model_file=MODEL_PATH, model_proto=self.model_bytes) # both

    with self.assertRaises(ValueError):
      spm.SentencePieceProcessor(return_type='invalid')

    # Deprecated out_type
    sp_out_type = spm.SentencePieceProcessor(out_type=str)
    self.assertEqual(sp_out_type._return_type, str)

    with self.assertRaises(ValueError):
      spm.SentencePieceProcessor(return_type=str, out_type=str)

    # Deprecated immutable_proto
    with self.assertRaises(ValueError):
      spm.SentencePieceProcessor(return_type='immutable_proto')

    # Internal helper direct call for coverage
    sp.Load(MODEL_PATH)
    self.assertEqual(sp._resolve_and_validate_return_type(None), int) # default is int

  def test_encode_decode_types_single(self):
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
    text = "This is a test."
    text_bytes = text.encode('utf-8')

    # return_type=int
    ids = sp.encode(text, return_type=int)
    self.assertIsInstance(ids, list)
    self.assertTrue(all(isinstance(x, int) for x in ids))
    self.assertEqual(ids, sp.encode_as_ids(text))

    # return_type=str
    pieces = sp.encode(text, return_type=str)
    self.assertIsInstance(pieces, list)
    self.assertTrue(all(isinstance(x, str) for x in pieces))
    self.assertEqual(pieces, sp.encode_as_pieces(text))

    # return_type=bytes
    pieces_bytes = sp.encode(text, return_type=bytes)
    self.assertIsInstance(pieces_bytes, list)
    self.assertTrue(all(isinstance(x, bytes) for x in pieces_bytes))

    # return_type=numpy
    ids_np = sp.encode(text, return_type='numpy')
    self.assertIsInstance(ids_np, np.ndarray)
    self.assertEqual(ids_np.dtype, np.int32)
    np.testing.assert_array_equal(ids_np, np.array(ids, dtype=np.int32))
    np.testing.assert_array_equal(ids_np, sp.encode_as_numpy(text))

    # return_type=serialized_proto
    proto_serialized = sp.encode(text, return_type='serialized_proto')
    self.assertIsInstance(proto_serialized, bytes)


    # return_type=proto
    if has_protobuf:
      proto = sp.encode(text, return_type='proto')
      self.assertIsInstance(proto, sentencepiece_pb2.SentencePieceText)
      self.assertEqual(proto.text, text)
      self.assertEqual(proto, sp.encode_as_proto(text))
    else:
      with self.assertRaises(ImportError):
        sp.encode(text, return_type='proto')

    # return_type=offset_mapping
    offset_map = sp.encode(text, return_type='offset_mapping')
    self.assertIsInstance(offset_map, dict)
    self.assertIn('ids', offset_map)
    self.assertIn('pieces', offset_map)
    self.assertIn('offsets', offset_map)
    self.assertEqual(offset_map['ids'], ids)
    self.assertEqual(offset_map['pieces'], pieces)
    self.assertEqual(len(offset_map['offsets']), len(ids))
    self.assertEqual(offset_map, sp.encode_as_offset_mapping(text))

    # return_bytes with offset_mapping
    offset_map_bytes = sp.encode(text, return_type='offset_mapping', return_bytes=True)
    self.assertTrue(all(isinstance(x, bytes) for x in offset_map_bytes['pieces']))
    self.assertEqual(offset_map_bytes['pieces'], pieces_bytes)

    # input as bytes
    ids_from_bytes = sp.encode(text_bytes, return_type=int)
    self.assertEqual(ids_from_bytes, ids)

    # Decode
    self.assertEqual(sp.decode(ids), text)
    self.assertEqual(sp.decode(pieces), text)
    self.assertEqual(sp.decode(pieces_bytes), text_bytes) # decoding bytes pieces returns bytes
    self.assertEqual(sp.decode(ids_np), text)

    # Decode to bytes
    self.assertEqual(sp.decode(ids, return_type=bytes), text_bytes)
    self.assertEqual(sp.decode(pieces, return_type=bytes), text_bytes)

    # Decode offset_mapping
    dec_offset_map = sp.decode(ids, return_type='offset_mapping')
    self.assertEqual(dec_offset_map['text'], text)
    self.assertEqual(dec_offset_map['ids'], ids)
    self.assertEqual(dec_offset_map['pieces'], pieces)

    dec_offset_map_bytes = sp.decode(ids, return_type='offset_mapping', return_bytes=True)
    self.assertEqual(dec_offset_map_bytes['text'], text_bytes)
    self.assertEqual(dec_offset_map_bytes['pieces'], pieces_bytes)

    if has_protobuf:
      dec_proto = sp.decode(ids, return_type='proto')
      self.assertIsInstance(dec_proto, sentencepiece_pb2.SentencePieceText)
      self.assertEqual(dec_proto.text, text)

    # Helper Decode calls
    self.assertEqual(sp.DecodePieces(pieces), text)
    self.assertEqual(sp.DecodeIds(ids), text)
    self.assertIsInstance(sp.decode(pieces, return_type='serialized_proto'), bytes)
    self.assertIsInstance(sp.decode(ids, return_type='serialized_proto'), bytes)
    if has_protobuf:
      self.assertIsInstance(sp.DecodePiecesAsProto(pieces), sentencepiece_pb2.SentencePieceText)
      self.assertIsInstance(sp.DecodeIdsAsProto(ids), sentencepiece_pb2.SentencePieceText)

    # Encode error cases
    with self.assertRaises(ValueError):
      sp.encode(text, return_type=int, return_bytes=True) # return_bytes only for offset_mapping
    with self.assertRaises(TypeError):
      sp.encode(text, num_threads='invalid')

    # Decode error cases
    with self.assertRaises(ValueError):
      sp.decode(ids, return_type='numpy')
    with self.assertRaises(ValueError):
      sp.decode(ids, return_type=str, return_bytes=True)
    with self.assertRaises(TypeError):
      sp.decode(ids, num_threads='invalid')



  def test_encode_decode_batch(self):
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
    texts = ["This is a test.", "Hello world!"]
    texts_bytes = [t.encode('utf-8') for t in texts]

    # Batch Encode
    ids_batch = sp.encode(texts, return_type=int)
    self.assertIsInstance(ids_batch, list)
    self.assertEqual(len(ids_batch), 2)
    self.assertTrue(all(isinstance(x, list) for x in ids_batch))

    pieces_batch = sp.encode(texts, return_type=str)
    pieces_bytes_batch = sp.encode(texts, return_type=bytes)
    
    ids_np_batch = sp.encode(texts, return_type='numpy')
    self.assertTrue(all(isinstance(x, np.ndarray) for x in ids_np_batch))

    proto_serialized_batch = sp.encode(texts, return_type='serialized_proto')
    self.assertTrue(all(isinstance(x, bytes) for x in proto_serialized_batch))

    if has_protobuf:
      proto_batch = sp.encode(texts, return_type='proto')
      self.assertTrue(all(isinstance(x, sentencepiece_pb2.SentencePieceText) for x in proto_batch))

    offset_map_batch = sp.encode(texts, return_type='offset_mapping')
    self.assertEqual(len(offset_map_batch), 2)

    # Batch Decode
    self.assertEqual(sp.decode(ids_batch), texts)
    self.assertEqual(sp.decode(pieces_batch), texts)
    self.assertEqual(sp.decode(pieces_bytes_batch), texts_bytes) # decoding bytes pieces returns bytes
    self.assertEqual(sp.decode(ids_np_batch), texts)

    self.assertEqual(sp.decode(ids_batch, return_type=bytes), texts_bytes)

    dec_offset_map_batch = sp.decode(ids_batch, return_type='offset_mapping')
    self.assertEqual(len(dec_offset_map_batch), 2)
    self.assertEqual(dec_offset_map_batch[0]['text'], texts[0])

    if has_protobuf:
      dec_proto_batch = sp.decode(ids_batch, return_type='proto')
      self.assertEqual(len(dec_proto_batch), 2)
      self.assertEqual(dec_proto_batch[0].text, texts[0])

  def test_decode_edge_cases(self):
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
    self.assertEqual(sp.decode(None), '')
    self.assertEqual(sp.decode([]), '')
    self.assertEqual(sp.decode(""), '')
    self.assertEqual(sp.decode(b""), '')
    # batch of empty
    self.assertEqual(sp.decode([[]]), [''])
    # ID 1 (<s>) is control, should decode to empty string if stripped
    self.assertEqual(sp.decode(1), '')
    
    # Find a normal piece to test decoding of single ID
    normal_id = -1
    for i in range(sp.vocab_size()):
      if not sp.is_control(i) and not sp.is_unknown(i) and not sp.is_unused(i):
        normal_id = i
        break
    self.assertNotEqual(normal_id, -1)
    decoded = sp.decode(normal_id)
    self.assertTrue(len(decoded) > 0)

  def test_encode_extra_options(self):
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
    self.assertEqual(sp.bos_id(), 1)
    self.assertEqual(sp.eos_id(), 2)
    
    text = "test"
    ids_normal = sp.encode(text, return_type=int)
    
    ids_bos = sp.encode(text, return_type=int, add_bos=True)
    self.assertEqual(ids_bos, [sp.bos_id()] + ids_normal)

    ids_eos = sp.encode(text, return_type=int, add_eos=True)
    self.assertEqual(ids_eos, ids_normal + [sp.eos_id()])

    ids_both = sp.encode(text, return_type=int, add_bos=True, add_eos=True)
    self.assertEqual(ids_both, [sp.bos_id()] + ids_normal + [sp.eos_id()])

    ids_reverse = sp.encode(text, return_type=int, reverse=True)
    self.assertEqual(ids_reverse, list(reversed(ids_normal)))

    # Test emit_unk_piece
    text_unk = "test ❤️"
    pieces_normal = sp.encode(text_unk, return_type=str)
    pieces_unk_emitted = sp.encode(text_unk, return_type=str, emit_unk_piece=True)

  def test_sampling_encode(self):
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
    text = "This is a test."
    
    ids1 = sp.encode(text, return_type=int, enable_sampling=True, nbest_size=-1, alpha=0.1)
    ids2 = sp.encode(text, return_type=int, enable_sampling=True, nbest_size=-1, alpha=0.1)

    # Error cases for sampling
    with self.assertRaises(ValueError):
      sp.encode(text, enable_sampling=True, nbest_size=1)
    with self.assertRaises(ValueError):
      sp.encode(text, enable_sampling=True, nbest_size=0)

    # Helper methods
    sp.sample_encode_as_ids(text, nbest_size=-1, alpha=0.1)
    sp.sample_encode_as_pieces(text, nbest_size=-1, alpha=0.1)
    sp.sample_encode_as_numpy(text, nbest_size=-1, alpha=0.1)

    if has_protobuf:
      sp.sample_encode_as_proto(text, nbest_size=-1, alpha=0.1)



  def test_nbest_encode(self):
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
    text = "This is a test."

    # Default nbest_size (should default to 1 because self._nbest_size is -1 <= 0)
    nbest_default = sp.nbest_encode(text, return_type=int)
    self.assertEqual(len(nbest_default), 1)

    nbest_ids = sp.nbest_encode(text, nbest_size=5, return_type=int)
    self.assertIsInstance(nbest_ids, list)
    self.assertEqual(len(nbest_ids), 5)

    nbest_pieces = sp.nbest_encode(text, nbest_size=5, return_type=str)
    self.assertEqual(len(nbest_pieces), 5)

    nbest_bytes = sp.nbest_encode(text, nbest_size=5, return_type=bytes)
    
    nbest_np = sp.nbest_encode(text, nbest_size=5, return_type='numpy')
    self.assertEqual(len(nbest_np), 5)

    nbest_serialized = sp.nbest_encode(text, nbest_size=5, return_type='serialized_proto')
    
    if has_protobuf:
      nbest_proto = sp.nbest_encode(text, nbest_size=5, return_type='proto')
      self.assertIsInstance(nbest_proto, sentencepiece_pb2.NBestSentencePieceText)

    # Batch NBest
    batch_nbest_ids = sp.nbest_encode([text, "another"], nbest_size=3, return_type=int)
    self.assertEqual(len(batch_nbest_ids), 2)
    self.assertEqual(len(batch_nbest_ids[0]), 3)

    batch_nbest_np = sp.nbest_encode([text, "another"], nbest_size=3, return_type='numpy')
    self.assertEqual(len(batch_nbest_np), 2)
    self.assertEqual(len(batch_nbest_np[0]), 3)
    self.assertIsInstance(batch_nbest_np[0][0], np.ndarray)

    # Unsupported return type
    with self.assertRaises(ValueError):
      sp.nbest_encode(text, nbest_size=3, return_type='offset_mapping')

    # Helpers
    sp.nbest_encode_as_ids(text, nbest_size=3)
    sp.nbest_encode_as_pieces(text, nbest_size=3)
    sp.nbest_encode_as_numpy(text, nbest_size=3)

    if has_protobuf:
      sp.nbest_encode_as_proto(text, nbest_size=3)



  

  def test_parallel_encode(self):
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
    text = "This is a test." * 100
    pool = spm.ThreadPool(4)

    ids = sp.parallel_encode(text, chunk_len=50, thread_pool=pool, return_type=int)
    self.assertIsInstance(ids, list)

    pieces = sp.parallel_encode(text, chunk_len=50, thread_pool=pool, return_type=str)
    pieces_bytes = sp.parallel_encode(text, chunk_len=50, thread_pool=pool, return_type=bytes)
    ids_np = sp.parallel_encode(text, chunk_len=50, thread_pool=pool, return_type='numpy')
    serialized = sp.parallel_encode(text, chunk_len=50, thread_pool=pool, return_type='serialized_proto')
    
    if has_protobuf:
      proto = sp.parallel_encode(text, chunk_len=50, thread_pool=pool, return_type='proto')
      self.assertIsInstance(proto, sentencepiece_pb2.SentencePieceText)

    # Batch Parallel Encode
    texts = [text, text]
    ids_batch = sp.parallel_encode(texts, chunk_len=50, thread_pool=pool, return_type=int)
    self.assertEqual(len(ids_batch), 2)

    # Batch Parallel Encode Numpy
    ids_np_batch = sp.parallel_encode(texts, chunk_len=50, thread_pool=pool, return_type='numpy')
    self.assertEqual(len(ids_np_batch), 2)
    self.assertIsInstance(ids_np_batch[0], np.ndarray)

    # Error cases
    with self.assertRaises(TypeError):
      sp.parallel_encode(text, chunk_len=50, num_threads='invalid')
    with self.assertRaises(ValueError):
      sp.parallel_encode(text, chunk_len=-1, thread_pool=pool)
    with self.assertRaises(ValueError):
      sp.parallel_encode(text, chunk_len=50, thread_pool=pool, return_type='offset_mapping')

    # Helpers
    sp.parallel_encode_as_ids(text, chunk_len=50, thread_pool=pool)
    sp.parallel_encode_as_pieces(text, chunk_len=50, thread_pool=pool)
    sp.parallel_encode_as_numpy(text, chunk_len=50, thread_pool=pool)

    if has_protobuf:
      sp.parallel_encode_as_proto(text, chunk_len=50, thread_pool=pool)



  def test_normalization_and_entropy(self):
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
    text = "Hello  World."
    
    # Normalize
    norm = sp.normalize(text)
    self.assertEqual(norm, sp.Normalize(text))
    
    norm_with_offsets = sp.normalize(text, with_offsets=True)
    self.assertIsInstance(norm_with_offsets, tuple)
    self.assertEqual(len(norm_with_offsets), 2)

    # Batch Normalize
    norms = sp.normalize([text, text])
    self.assertEqual(len(norms), 2)

    norms_with_offsets = sp.normalize([text, text], with_offsets=True)
    self.assertEqual(len(norms_with_offsets), 2)





  def test_vocab_metadata(self):
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
    
    self.assertEqual(sp.get_piece_size(), 1000)
    self.assertEqual(len(sp), 1000)

    # PieceToId / IdToPiece
    self.assertEqual(sp.piece_to_id('<unk>'), 0)
    self.assertEqual(sp.id_to_piece(0), '<unk>')

    # Batch PieceToId / IdToPiece
    ids = sp.piece_to_id(['<unk>', '<s>'])
    self.assertEqual(ids, [0, 1])
    pieces = sp.id_to_piece([0, 1])
    self.assertEqual(pieces, ['<unk>', '<s>'])

    # Dictionary access
    self.assertEqual(sp['<unk>'], 0)
    with self.assertRaises(TypeError):
      sp[0]

    # GetScore
    score = sp.get_score(0)
    self.assertIsInstance(score, float)
    scores = sp.get_score([0, 1])
    self.assertEqual(len(scores), 2)

    # IsUnknown, IsControl, IsUnused, IsByte
    self.assertTrue(sp.is_unknown(0))
    self.assertFalse(sp.is_unknown(1))
    self.assertTrue(all(isinstance(x, bool) for x in sp.is_unknown([0, 1])))

    self.assertTrue(sp.is_control(1))
    self.assertFalse(sp.is_control(0))

    sp.is_unused(0)
    sp.is_unused([0, 1])
    sp.is_byte(0)
    sp.is_byte([0, 1])

    # Special IDs
    self.assertEqual(sp.unk_id(), 0)
    self.assertEqual(sp.bos_id(), 1)
    self.assertEqual(sp.eos_id(), 2)
    self.assertEqual(sp.pad_id(), -1)

    # Out of range check for batchnized methods (IndexError)
    with self.assertRaises(IndexError):
      sp.id_to_piece(10000)
    with self.assertRaises(IndexError):
      sp.id_to_piece([0, 10000])

  def test_pickle(self):
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
    serialized = pickle.dumps(sp)
    sp2 = pickle.loads(serialized)
    self.assertEqual(sp.vocab_size(), sp2.vocab_size())
    self.assertEqual(sp.encode("test"), sp2.encode("test"))
    
    # Check __getstate_for_pickle__ directly
    self.assertEqual(sp.__getstate_for_pickle__(), sp.serialized_model_proto())

  def test_normalizer(self):
    normalizer = spm.SentencePieceNormalizer(model_file=MODEL_PATH)
    self.assertIsNotNone(normalizer)

    normalizer2 = spm.SentencePieceNormalizer(model_proto=self.model_bytes)
    self.assertIsNotNone(normalizer2)

    normalizer3 = spm.SentencePieceNormalizer(rule_name="nfkc")
    self.assertIsNotNone(normalizer3)

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsv') as f:
      # Map 'A' (0041) to 'a' (0061), and 'B' (0042) to 'b' (0062)
      f.write("0041\t0061\n")
      f.write("0042\t0062\n")
      tsv_path = f.name
    try:
      normalizer4 = spm.SentencePieceNormalizer(rule_tsv=tsv_path)
      self.assertIsNotNone(normalizer4)
      self.assertEqual(normalizer4.normalize("A B"), "a b")
    finally:
      os.remove(tsv_path)

    with self.assertRaises(ValueError):
      spm.SentencePieceNormalizer()

    text = "Hello  World."
    norm = normalizer.normalize(text)
    self.assertEqual(norm, normalizer.Normalize(text))

    norm_with_offsets = normalizer.normalize(text, with_offsets=True)
    self.assertIsInstance(norm_with_offsets, tuple)
    self.assertEqual(len(norm_with_offsets), 2)

    norms = normalizer.normalize([text, text])
    self.assertEqual(len(norms), 2)

    norms_with_offsets = normalizer.normalize([text, text], with_offsets=True)
    self.assertEqual(len(norms_with_offsets), 2)

    serialized = pickle.dumps(normalizer)
    normalizer_loaded = pickle.loads(serialized)
    self.assertEqual(normalizer.normalize(text), normalizer_loaded.normalize(text))

  def test_gil_release(self):
    heavy_text = " ".join(["Hello, world! Testing SentencePiece GIL release behavior."] * 50000)
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)

    def run_heavy_op_with_gil_check(op_func, name):
      counter = HeartbeatCounter()
      stop_event = threading.Event()
      bg_thread = threading.Thread(
          target=background_heartbeat, args=(stop_event, counter)
      )
      bg_thread.daemon = True
      bg_thread.start()

      time.sleep(0.05)

      start_time = time.time()
      result = op_func()
      end_time = time.time()
      elapsed_time = end_time - start_time

      stop_event.set()
      bg_thread.join()

      heartbeat_count = counter.get_count()

      is_gil_disabled = False
      if hasattr(sys, "_is_gil_enabled"):
        is_gil_disabled = not sys._is_gil_enabled()

      if is_gil_disabled:
        min_expected_heartbeats = 1
      else:
        sleep_interval = 0.01
        theoretical_max = elapsed_time / sleep_interval
        min_expected_heartbeats = int(theoretical_max * 0.3)

      if min_expected_heartbeats < 2:
        min_expected_heartbeats = 2

      self.assertGreaterEqual(
          heartbeat_count, min_expected_heartbeats,
          f"GIL Release Failure in {name}! The background thread was blocked. "
          f"Expected at least {min_expected_heartbeats} heartbeats, but only got {heartbeat_count}."
      )
      return result

    # 1. Test Encode
    ids = run_heavy_op_with_gil_check(lambda: sp.encode(heavy_text, return_type=int), "Encode")
    self.assertTrue(len(ids) > 0)

    # 2. Test Decode
    decoded = run_heavy_op_with_gil_check(lambda: sp.decode(ids), "Decode")
    self.assertEqual(decoded, heavy_text)

    # 3. Test ParallelEncode
    heavy_texts = [heavy_text] * 4
    pool = spm.ThreadPool(4)
    parallel_tokens = run_heavy_op_with_gil_check(
        lambda: sp.parallel_encode(heavy_texts, chunk_len=10000, thread_pool=pool, return_type=int),
        "ParallelEncode"
    )
    self.assertEqual(len(parallel_tokens), 4)

  def test_gil_release_encode_with_list_modification(self):
    # This test verifies that modifying the input list in a background thread
    # while C++ is encoding doesn't cause a crash (due to dangling pointers).
    single_heavy = "Hello, world! Testing SentencePiece GIL release behavior. " * 10000
    texts = [(single_heavy + str(i)) for i in range(50)]
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
    
    def modify_list():
      time.sleep(0.01)
      texts.clear()
      import gc
      gc.collect()

    bg_thread = threading.Thread(target=modify_list)
    bg_thread.start()
    
    ids = sp.encode(texts, return_type=int, num_threads=2)
    bg_thread.join()
    
    self.assertEqual(len(ids), 50)

  def test_gil_release_decode_pieces_with_list_modification(self):
    # This test verifies that modifying the input list of pieces in a background thread
    # while C++ is decoding doesn't cause a crash.
    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
    pieces = [sp.encode("Hello world", return_type=str) for _ in range(50)]
    
    def modify_list():
      time.sleep(0.01)
      pieces.clear()
      import gc
      gc.collect()

    bg_thread = threading.Thread(target=modify_list)
    bg_thread.start()
    
    decoded = sp.decode(pieces)
    bg_thread.join()
    
    self.assertEqual(len(decoded), 50)

  def test_encode_bytes_is_zero_copy(self):
    # This test verifies that Encode uses the Python bytes buffer directly (zero-copy).
    import ctypes
    
    # Verify offset first
    b_test = b"TestOffset"
    offset = 32
    try:
      current = ctypes.string_at(id(b_test) + offset, len(b_test))
      if current != b_test:
        self.skipTest("CPython PyBytesObject offset is different on this platform.")
    except Exception as e:
      self.skipTest(f"Cannot verify CPython offset: {e}")

    sp = spm.SentencePieceProcessor(model_file=MODEL_PATH)
    size = 100000
    original_char = b'a'
    modified_char = b'b'
    b = original_char * size
    
    ids_pure_a = sp.encode(original_char * size, return_type=int)
    
    def modify_buffer():
      time.sleep(0.001)
      addr = id(b) + offset
      new_data = modified_char * size
      ctypes.memmove(addr, new_data, size)

    bg_thread = threading.Thread(target=modify_buffer)
    bg_thread.start()
    
    ids = sp.encode(b, return_type=int)
    bg_thread.join()
    
    self.assertNotEqual(ids, ids_pure_a, "Content was copied, expected zero-copy.")

  def test_encode_str_is_zero_copy(self):
    # This test verifies that Encode uses the Python str UTF-8 buffer directly (zero-copy).
    # It dynamically trains a small model to ensure the test characters (Japanese)
    # are in the vocabulary and tokenize to different IDs.
    import ctypes
    
    # 1. Train a small model containing our test characters
    corpus_data = ("日本語\n甲乙丙\n" * 100).encode('utf-8')
    with tempfile.NamedTemporaryFile(delete=False) as f:
      f.write(corpus_data)
      corpus_path = f.name
      
    model_prefix = tempfile.mktemp()
    model_path = model_prefix + ".model"
    
    try:
      spm.SentencePieceTrainer.train(
          input=corpus_path,
          model_prefix=model_prefix,
          vocab_size=20,
          character_coverage=1.0,
          model_type="char"
      )
    finally:
      if os.path.exists(corpus_path):
        os.unlink(corpus_path)

    try:
      sp = spm.SentencePieceProcessor(model_file=model_path)
      original_word = "日本語"
      modified_word = "甲乙丙"
      
      # Verify they tokenize to different IDs
      ids_orig = sp.encode(original_word, return_type=int)
      ids_mod = sp.encode(modified_word, return_type=int)
      if ids_orig == ids_mod:
        self.skipTest("Dynamically trained model failed to separate test vocab.")

      size = 100000
      s = original_word * size
      
      # Trigger caching to populate the UTF-8 pointer
      ids_pure_orig = sp.encode(s, return_type=int)
      
      # Verify offset first (compact Unicode object UTF-8 pointer in CPython 3.12+ is 48, len is 40)
      addr = id(s)
      offset_len = 40
      offset_ptr = 48
      try:
        utf8_ptr = ctypes.c_void_p.from_address(addr + offset_ptr).value
        utf8_len = ctypes.c_void_p.from_address(addr + offset_len).value
        current = ctypes.string_at(utf8_ptr, utf8_len)
        if current != s.encode('utf-8'):
          self.skipTest("CPython PyCompactUnicodeObject layout is different on this platform.")
      except Exception as e:
        self.skipTest(f"Cannot verify CPython offset: {e}")

      def modify_buffer():
        time.sleep(0.001)
        new_data = (modified_word * size).encode('utf-8')
        ctypes.memmove(utf8_ptr, new_data, utf8_len)

      bg_thread = threading.Thread(target=modify_buffer)
      bg_thread.start()
      
      ids = sp.encode(s, return_type=int)
      bg_thread.join()
      
      self.assertNotEqual(ids, ids_pure_orig, "Content was copied, expected zero-copy.")
      
    finally:
      if os.path.exists(model_path):
        os.unlink(model_path)
      vocab_path = model_prefix + ".vocab"
      if os.path.exists(vocab_path):
        os.unlink(vocab_path)

  def test_global_functions(self):
    spm.set_random_generator_seed(42)
    spm.set_min_log_level(1)
    spm.set_nbest_timeout(100) # milliseconds
    spm.set_min_log_level(0) # Reset to default

  def test_async_methods(self):
    import asyncio
    import functools
    from concurrent.futures import ThreadPoolExecutor
    
    sp = spm.SentencePieceProcessor()
    sp.load(MODEL_PATH)
    
    executor = ThreadPoolExecutor(max_workers=2)
    
    async def async_encode(text, **kwargs):
      loop = asyncio.get_running_loop()
      func = functools.partial(sp.encode, text, **kwargs)
      return await loop.run_in_executor(executor, func)

    async def async_decode(ids, **kwargs):
      loop = asyncio.get_running_loop()
      func = functools.partial(sp.decode, ids, **kwargs)
      return await loop.run_in_executor(executor, func)
    
    async def run_test():
      # Test single encode/decode
      ids = await async_encode("hello world", return_type=int)
      self.assertIsInstance(ids, list)
      self.assertTrue(all(isinstance(x, int) for x in ids))
      
      text = await async_decode(ids)
      self.assertIsInstance(text, str)
      self.assertEqual(text.strip(), "hello world") 
      
      # Test batch encode/decode
      batch_ids = await async_encode(["hello world", "foo bar"], return_type=int)
      self.assertIsInstance(batch_ids, list)
      self.assertEqual(len(batch_ids), 2)
      
      batch_texts = await async_decode(batch_ids)
      self.assertIsInstance(batch_texts, list)
      self.assertEqual(len(batch_texts), 2)
      self.assertEqual(batch_texts[0].strip(), "hello world")
      self.assertEqual(batch_texts[1].strip(), "foo bar")
      
    try:
      asyncio.run(run_test())
    finally:
      executor.shutdown()


class TestSentencePieceTrainerClean(unittest.TestCase):

  def setUp(self):
    self.input_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    for _ in range(100):
      self.input_file.write("This is a test sentence for training.\n")
      self.input_file.write("SentencePiece is a useful tool.\n")
    self.input_file.close()

  def tearDown(self):
    os.remove(self.input_file.name)
    for ext in ['.model', '.vocab']:
      path = 'm_test' + ext
      if os.path.exists(path):
        os.remove(path)

  def test_train_from_string(self):
    args = f"--input={self.input_file.name} --model_prefix=m_test --vocab_size=30"
    self.assertTrue(spm.SentencePieceTrainer.train(args))
    self.assertTrue(os.path.exists("m_test.model"))
    self.assertTrue(os.path.exists("m_test.vocab"))

  def test_train_from_kwargs(self):
    spm.SentencePieceTrainer.train(
        input=self.input_file.name,
        model_prefix="m_test",
        vocab_size=30,
        model_type="unigram",
        user_defined_symbols=['foo', 'bar'] # Test list encoding to CSV
    )
    self.assertTrue(os.path.exists("m_test.model"))

  def test_train_with_iterator(self):
    sentences = ["This is a test sentence for training.", "SentencePiece is a useful tool."] * 50
    
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        model_prefix="m_test",
        vocab_size=30
    )
    self.assertTrue(os.path.exists("m_test.model"))

    model_io = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        model_writer=model_io,
        vocab_size=30
    )
    self.assertTrue(len(model_io.getvalue()) > 0)
    
    sp = spm.SentencePieceProcessor(model_proto=model_io.getvalue())
    self.assertEqual(sp.vocab_size(), 30)

  def test_train_with_writer_only(self):
    model_io = io.BytesIO()
    spm.SentencePieceTrainer.train(
        input=self.input_file.name,
        model_writer=model_io,
        vocab_size=30
    )
    self.assertTrue(len(model_io.getvalue()) > 0)
    sp = spm.SentencePieceProcessor(model_proto=model_io.getvalue())
    self.assertEqual(sp.vocab_size(), 30)

  def test_train_with_logstream(self):
    # Test logstream redirection during training
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as log_file:
      log_path = log_file.name
    try:
      with open(log_path, 'w') as logstream:
        spm.SentencePieceTrainer.train(
            input=self.input_file.name,
            model_prefix="m_test",
            vocab_size=30,
            logstream=logstream
        )
      # Check if log file is not empty
      self.assertTrue(os.path.exists(log_path))
      self.assertTrue(os.path.getsize(log_path) > 0)
    finally:
      os.remove(log_path)

if __name__ == '__main__':
  unittest.main()
