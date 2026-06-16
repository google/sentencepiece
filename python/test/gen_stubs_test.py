# Copyright 2024 Google Sepm.
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
# limitations under the License.

"""Tests for tools/gen_stubs.py."""

import os
import sys
import types
import unittest

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.abspath(os.path.join(HERE, os.pardir, "tools"))
sys.path.insert(0, TOOLS_DIR)

import gen_stubs  # pylint: disable=g-import-not-at-top
import sentencepiece as spm  # pylint: disable=g-import-not-at-top

STUB_PATH = os.path.join(os.path.dirname(spm.__file__), "__init__.pyi")

pytestmark = pytest.mark.thread_unsafe


class GenStubsTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    with open(STUB_PATH, encoding="utf-8") as f:
      cls.committed_stub = f.read()
    cls.spp = spm.SentencePieceProcessor

  def test_committed_stub_in_sync(self):
    """Scenario A: The committed stub matches a fresh generation."""
    text, _ = gen_stubs.generate()
    self.assertEqual(text, self.committed_stub)

  def test_added_method(self):
    """Scenario B: A newly added public method is handled correctly."""
    # Mutate
    self.spp.BrandNewParallelThing = lambda self: None
    try:
      text, uncurated = gen_stubs.generate()

      self.assertIn("def BrandNewParallelThing(", text)
      self.assertNotIn("def brand_new_parallel_thing(", text)
      self.assertTrue(any("BrandNewParallelThing" in u for u in uncurated))
      self.assertIn(
          "def BrandNewParallelThing(self, *args: Any, **kwargs: Any) -> Any:"
          " ...",
          text,
      )
      self.assertNotEqual(text, self.committed_stub)
    finally:
      # Restore
      del self.spp.BrandNewParallelThing

    # Verify restored state
    text_restored, _ = gen_stubs.generate()
    self.assertEqual(text_restored, self.committed_stub)

  def test_removed_method(self):
    """Scenario C: A removed method is handled correctly."""
    original_encode = self.spp.Encode
    # Mutate
    del self.spp.Encode
    try:
      text, _ = gen_stubs.generate()
      self.assertNotIn("def Encode(", text)
      self.assertNotEqual(text, self.committed_stub)
    finally:
      # Restore
      self.spp.Encode = original_encode

    # Verify restored state
    text_restored, _ = gen_stubs.generate()
    self.assertEqual(text_restored, self.committed_stub)

  def test_added_property(self):
    """Scenario D: A new property is handled correctly."""
    self.spp.brand_new_prop = property(lambda self: 1)
    try:
      text, uncurated = gen_stubs.generate()
      self.assertIn("def brand_new_prop(self) -> Any: ...", text)
      self.assertTrue(any("brand_new_prop" in u for u in uncurated))
    finally:
      del self.spp.brand_new_prop

  def test_leaked_module_level_import_ignored(self):
    """Scenario E: Leaked module-level imports are ignored."""
    spm.some_leaked_module = types.ModuleType("some_leaked_module")
    spm.some_leaked_constant = 1234
    try:
      text, _ = gen_stubs.generate()
      self.assertNotIn("some_leaked_module", text)
      self.assertNotIn("some_leaked_constant", text)
    finally:
      del spm.some_leaked_module
      del spm.some_leaked_constant

  def test_determinism(self):
    """Scenario F: Output is deterministic."""
    first, _ = gen_stubs.generate()
    second, _ = gen_stubs.generate()
    self.assertEqual(first, second)


if __name__ == "__main__":
  unittest.main()
