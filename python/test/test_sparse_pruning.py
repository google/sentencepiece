"""Test that sparse training options are local to each training call."""

import io
import os
import tempfile
import unittest

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2


class TestSparsePruning(unittest.TestCase):
    """Exercise sparse options through both public training entry points."""

    def test_sparse_pruning_options(self):
        """Preserve per-call settings and Unicode round trips with byte fallback."""
        lines = ["hello world", "hello tokenizer", "ภาษาไทย 中文 日本語"] * 20
        for sparse in (True, False):
            with self.subTest(sparse=sparse):
                writer = io.BytesIO()
                spm.SentencePieceTrainer.train(
                    sentence_iterator=iter(lines),
                    model_type="unigram",
                    vocab_size=320,
                    hard_vocab_limit=False,
                    byte_fallback=True,
                    auto_character_coverage=sparse,
                    use_sparse_pruning=sparse,
                    post_l1_debias=False,
                    seed_piece_length_power=0.5,
                    min_freq_alpha=0.1,
                    normalization_rule_name="identity",
                    model_writer=writer,
                )
                model = writer.getvalue()
                proto = sentencepiece_model_pb2.ModelProto.FromString(model)
                self.assertEqual(proto.trainer_spec.use_sparse_pruning, sparse)
                self.assertFalse(proto.trainer_spec.post_l1_debias)
                self.assertAlmostEqual(proto.trainer_spec.seed_piece_length_power, 0.5)
                self.assertAlmostEqual(proto.trainer_spec.min_freq_alpha, 0.1)
                processor = spm.SentencePieceProcessor(model_proto=model)
                for line in lines[:3]:
                    self.assertEqual(processor.decode(processor.encode(line)), line)

        with self.assertRaisesRegex(RuntimeError, "use_sparse_pruning"):
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(lines),
                vocab_size=320,
                byte_fallback=True,
                auto_character_coverage=True,
            )

    def test_sparse_pruning_string_options(self):
        """Accept sparse flags in string arguments and preserve default options."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus = os.path.join(tmp_dir, "input.txt")
            prefix = os.path.join(tmp_dir, "model")
            with open(corpus, "w", encoding="utf-8") as output:
                output.write("hello world hello tokenizer ภาษาไทย 中文 日本語\n" * 20)
            spm.SentencePieceTrainer.train(
                f"--input={corpus} --model_prefix={prefix} --vocab_size=320 "
                "--hard_vocab_limit=false --byte_fallback=true "
                "--auto_character_coverage=true --use_sparse_pruning=true"
            )
            with open(prefix + ".model", "rb") as model:
                proto = sentencepiece_model_pb2.ModelProto.FromString(model.read())
            self.assertTrue(proto.trainer_spec.use_sparse_pruning)
            self.assertTrue(proto.trainer_spec.post_l1_debias)
            self.assertEqual(proto.trainer_spec.seed_piece_length_power, 1.0)
            self.assertEqual(proto.trainer_spec.min_freq_alpha, 0.0)
