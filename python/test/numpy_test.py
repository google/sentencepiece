import os
import unittest
import sentencepiece as spm

try:
    import numpy as np
    has_numpy = True
except ImportError:
    has_numpy = False

@unittest.skipIf(not has_numpy, "numpy is not installed")
class TestNumpyIntegration(unittest.TestCase):
    def setUp(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'test_model.model')
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

    def test_encode_numpy_single(self):
        text = "This is a test sentence."
        
        # Standard list output
        ids_list = self.sp.encode(text, return_type=int)
        self.assertIsInstance(ids_list, list)
        self.assertTrue(all(isinstance(x, int) for x in ids_list))
        
        # NumPy output
        ids_np = self.sp.encode(text, return_type='numpy')
        self.assertIsInstance(ids_np, np.ndarray)
        self.assertEqual(ids_np.dtype, np.int32)
        self.assertEqual(ids_np.ndim, 1)
        
        # Check values match
        self.assertEqual(ids_np.tolist(), ids_list)
        
        # Check read-only
        with self.assertRaises(ValueError):
            ids_np[0] = 999

    def test_encode_numpy_batch(self):
        texts = ["Hello world", "This is another test."]
        
        # Standard list output
        ids_list = self.sp.encode(texts, return_type=int)
        
        # NumPy output
        ids_np_list = self.sp.encode(texts, return_type='numpy')
        self.assertIsInstance(ids_np_list, list)
        self.assertEqual(len(ids_np_list), len(texts))
        
        for ids_np, ids_l in zip(ids_np_list, ids_list):
            self.assertIsInstance(ids_np, np.ndarray)
            self.assertEqual(ids_np.dtype, np.int32)
            self.assertEqual(ids_np.tolist(), ids_l)
            # Check read-only
            with self.assertRaises(ValueError):
                ids_np[0] = 999

    def test_return_type_validation_and_compat(self):
        text = "This is a test sentence."
        
        # Verify return_type='numpy' works
        ids_np = self.sp.encode(text, return_type='numpy')
        self.assertIsInstance(ids_np, np.ndarray)
        
        # Verify out_type='numpy' works as alias
        ids_np_alias = self.sp.encode(text, out_type='numpy')
        self.assertIsInstance(ids_np_alias, np.ndarray)
        self.assertEqual(ids_np.tolist(), ids_np_alias.tolist())

        # Verify passing both raises ValueError
        with self.assertRaises(ValueError):
            self.sp.encode(text, return_type='numpy', out_type='numpy')
            
        with self.assertRaises(ValueError):
            self.sp.encode(text, return_type='numpy', out_type=int)
            
        # Verify invalid return_type raises ValueError
        with self.assertRaises(ValueError):
            self.sp.encode(text, return_type='invalid_type')

    def test_decode_numpy_single_32(self):
        text = "This is a test sentence."
        ids_np = self.sp.encode(text, return_type='numpy')
        
        # Decode from np.int32 array
        self.assertEqual(ids_np.dtype, np.int32)
        decoded = self.sp.decode(ids_np)
        self.assertEqual(decoded, text)

    def test_decode_numpy_single_64(self):
        text = "This is a test sentence."
        ids_list = self.sp.encode(text, return_type=int)
        
        # Create np.int64 array (default on many platforms for np.array)
        ids_np64 = np.array(ids_list, dtype=np.int64)
        self.assertEqual(ids_np64.dtype, np.int64)
        
        # Decode from np.int64 array (should trigger safe downcast in C++)
        decoded = self.sp.decode(ids_np64)
        self.assertEqual(decoded, text)

    def test_decode_numpy_batch_list(self):
        texts = ["Hello world", "This is another test."]
        ids_np_list = self.sp.encode(texts, return_type='numpy')
        
        # Decode from list of numpy arrays
        decoded = self.sp.decode(ids_np_list)
        self.assertEqual(decoded, texts)

    def test_decode_numpy_batch_2d(self):
        # To test 2D array, we need sequences of same length (or pad them).
        # We can just encode same text twice to get same length.
        text = "Hello world"
        ids = self.sp.encode(text, return_type=int)
        
        # Create 2D array
        ids_2d = np.array([ids, ids], dtype=np.int32)
        self.assertEqual(ids_2d.ndim, 2)
        
        decoded = self.sp.decode(ids_2d)
        self.assertEqual(decoded, [text, text])

        # Test 2D array with int64
        ids_2d_64 = np.array([ids, ids], dtype=np.int64)
        decoded_64 = self.sp.decode(ids_2d_64)
        self.assertEqual(decoded_64, [text, text])

    def test_decode_numpy_non_contiguous(self):
        text = "This is a test sentence."
        ids = self.sp.encode(text, return_type=int)

        for dtype in (np.int32, np.int64):
            arr = np.array(ids, dtype=dtype)

            # Reversed view: negative stride, data pointer at the last element.
            reversed_view = arr[::-1]
            self.assertFalse(reversed_view.flags['C_CONTIGUOUS'])
            self.assertEqual(self.sp.decode(reversed_view),
                             self.sp.decode(list(ids[::-1])))

            # Strided slice: stride larger than itemsize.
            strided = arr[::2]
            self.assertFalse(strided.flags['C_CONTIGUOUS'])
            self.assertEqual(self.sp.decode(strided),
                             self.sp.decode(list(ids[::2])))

    def test_decode_numpy_invalid_types(self):
        # Float array should fail
        ids_float = np.array([1.0, 2.0], dtype=np.float32)
        with self.assertRaises(TypeError):
            self.sp.decode(ids_float)

        # 3D array should fail
        ids_3d = np.array([[[1]]], dtype=np.int32)
        with self.assertRaises(TypeError):
            self.sp.decode(ids_3d)

    def test_parallel_encode_numpy(self):
        text = "This is a test sentence for parallel encode."
        
        # Single input
        ids_list = self.sp.ParallelEncode(text, return_type=int, chunk_len=5, num_threads=2)
        ids_np = self.sp.ParallelEncode(text, return_type='numpy', chunk_len=5, num_threads=2)
        self.assertIsInstance(ids_np, np.ndarray)
        self.assertEqual(ids_np.dtype, np.int32)
        self.assertEqual(ids_np.tolist(), ids_list)

        # Batch input
        texts = ["Hello world", "This is another test."]
        ids_list_batch = self.sp.ParallelEncode(texts, return_type=int, chunk_len=5, num_threads=2)
        ids_np_batch = self.sp.ParallelEncode(texts, return_type='numpy', chunk_len=5, num_threads=2)
        self.assertIsInstance(ids_np_batch, list)
        self.assertEqual(len(ids_np_batch), len(texts))
        for ids_n, ids_l in zip(ids_np_batch, ids_list_batch):
            self.assertIsInstance(ids_n, np.ndarray)
            self.assertEqual(ids_n.dtype, np.int32)
            self.assertEqual(ids_n.tolist(), ids_l)

    def test_nbest_encode_numpy(self):
        text = "This is a test sentence."
        
        # Single input (returns list of np.ndarray)
        nbest_list = self.sp.NBestEncode(text, return_type=int, nbest_size=5)
        nbest_np = self.sp.NBestEncode(text, return_type='numpy', nbest_size=5)
        self.assertIsInstance(nbest_np, list)
        self.assertEqual(len(nbest_np), 5)
        for ids_n, ids_l in zip(nbest_np, nbest_list):
            self.assertIsInstance(ids_n, np.ndarray)
            self.assertEqual(ids_n.dtype, np.int32)
            self.assertEqual(ids_n.tolist(), ids_l)

        # Batch input (returns list of list of np.ndarray)
        texts = ["Hello world", "This is another test."]
        nbest_list_batch = self.sp.NBestEncode(texts, return_type=int, nbest_size=5)
        nbest_np_batch = self.sp.NBestEncode(texts, return_type='numpy', nbest_size=5)
        self.assertIsInstance(nbest_np_batch, list)
        self.assertEqual(len(nbest_np_batch), len(texts))
        for nbest_n, nbest_l in zip(nbest_np_batch, nbest_list_batch):
            self.assertIsInstance(nbest_n, list)
            self.assertEqual(len(nbest_n), 5)
            for ids_n, ids_l in zip(nbest_n, nbest_l):
                self.assertIsInstance(ids_n, np.ndarray)
                self.assertEqual(ids_n.dtype, np.int32)
                self.assertEqual(ids_n.tolist(), ids_l)

    def test_numpy_memory_safety(self):
        # Create a processor local to this scope
        local_sp = spm.SentencePieceProcessor(model_file=self.model_path)
        ids_np = local_sp.encode("Hello world", return_type='numpy')
        
        # Verify base object is present
        self.assertIsNotNone(ids_np.base)
        # Verify it is our custom VectorBuffer inside memoryview
        self.assertEqual(type(ids_np.base.obj).__name__, "VectorBuffer")
        
        # Verify that memory addresses match (zero-copy verification)
        np_address = ids_np.__array_interface__['data'][0]
        cpp_address = ids_np.base.obj.data_address()
        self.assertEqual(np_address, cpp_address)
        
        # Delete processor
        del local_sp
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Access the array values. Should not segfault!
        self.assertEqual(ids_np.tolist(), self.sp.encode("Hello world", return_type=int))

    def test_numpy_empty_input(self):
        # Empty string encode
        empty_np = self.sp.encode("", return_type='numpy')
        self.assertIsInstance(empty_np, np.ndarray)
        self.assertEqual(len(empty_np), 0)
        
        # Empty array decode
        decoded = self.sp.decode(np.array([], dtype=np.int32))
        self.assertEqual(decoded, "")

    def test_decode_invalid_ids_numpy(self):
        # kOutOfRange should map to IndexError for numpy inputs
        with self.assertRaises(IndexError):
            self.sp.decode(np.array([10000], dtype=np.int32))
        with self.assertRaises(IndexError):
            self.sp.decode(np.array([0, 10000], dtype=np.int32))
        with self.assertRaises(IndexError):
            self.sp.decode([np.array([10000], dtype=np.int32)])

    def test_decode_numpy_batch_mixed(self):
        texts = ["Hello world", "This is another test."]
        ids_list = self.sp.encode(texts, return_type=int)
        
        # Mix of read-only and writable numpy arrays in a list
        arr1 = np.array(ids_list[0], dtype=np.int32)
        arr1.flags.writeable = False
        arr2 = np.array(ids_list[1], dtype=np.int32)
        
        decoded = self.sp.decode([arr1, arr2])
        self.assertEqual(decoded, texts)

    def test_native_batch_id_to_piece_numpy(self):
        vocab_size = self.sp.vocab_size()
        # Batch valid (numpy)
        ids = np.array([0, 1, 2], dtype=np.int32)
        pieces = self.sp.IdToPiece(ids)
        self.assertIsInstance(pieces, list)
        self.assertEqual(len(pieces), len(ids))
        for i, p in zip(ids, pieces):
            self.assertEqual(p, self.sp.IdToPiece(int(i)))

        # Invalid ID in numpy array
        with self.assertRaises(IndexError):
            self.sp.IdToPiece(np.array([0, -1], dtype=np.int32))
        with self.assertRaises(IndexError):
            self.sp.IdToPiece(np.array([0, vocab_size], dtype=np.int32))

    def test_native_batch_other_id_methods_numpy(self):
        vocab_size = self.sp.vocab_size()
        methods = [
            self.sp.GetScore,
            self.sp.IsUnknown,
            self.sp.IsControl,
            self.sp.IsUnused,
            self.sp.IsByte
        ]
        for method in methods:
            # Batch valid (numpy)
            res_numpy = method(np.array([0, 1], dtype=np.int32))
            self.assertIsInstance(res_numpy, list)
            self.assertEqual(len(res_numpy), 2)
            
            # Batch invalid (numpy)
            with self.assertRaises(IndexError):
                method(np.array([0, -1], dtype=np.int32))
            with self.assertRaises(IndexError):
                method(np.array([0, vocab_size], dtype=np.int32))

if __name__ == '__main__':
    unittest.main()
