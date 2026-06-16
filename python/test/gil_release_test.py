import os
import sys
import threading
import time
import sentencepiece as spm

# Re-use the same test model
HERE = os.path.dirname(os.path.abspath(__file__))

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
  """This thread increments the counter as long as the GIL is properly released."""
  while not stop_event.is_set():
    counter.increment()
    sys.stdout.write(".")
    sys.stdout.flush()
    time.sleep(0.01)  # Force context switch

def run_heavy_op_with_gil_check(op_func, name):
  counter = HeartbeatCounter()
  stop_event = threading.Event()
  bg_thread = threading.Thread(
      target=background_heartbeat, args=(stop_event, counter)
  )
  bg_thread.daemon = True
  bg_thread.start()

  # Wait a brief moment to ensure the background thread spins up
  time.sleep(0.1)

  print(f"\n[Main] Starting {name}...", flush=True)
  start_time = time.time()

  result = op_func()

  end_time = time.time()
  elapsed_time = end_time - start_time
  print(
      f"\n[Main] {name} finished. Elapsed time: {elapsed_time:.4f} seconds",
      flush=True,
  )

  # Stop and clean up the background thread
  stop_event.set()
  bg_thread.join()

  heartbeat_count = counter.get_count()
  print(
      f"[Main] Background thread executed {heartbeat_count} times during {name}."
  )

  is_gil_disabled = False
  if hasattr(sys, "_is_gil_enabled"):
    is_gil_disabled = not sys._is_gil_enabled()

  if is_gil_disabled:
    min_expected_heartbeats = 1
  else:
    is_mac = sys.platform == "darwin"
    sleep_interval = 0.01
    theoretical_max = elapsed_time / sleep_interval
    margin = 0.1 if is_mac else 0.5
    min_expected_heartbeats = int(theoretical_max * margin)

    if min_expected_heartbeats < 1:
      min_expected_heartbeats = 1

  # Floor of at least 2 to ensure we actually switched context
  if min_expected_heartbeats < 2:
    min_expected_heartbeats = 2

  print(
      f"[Main] Expected at least {min_expected_heartbeats} heartbeats."
  )

  assert heartbeat_count >= min_expected_heartbeats, (
      f"GIL Release Failure in {name}! The background thread was blocked.\n"
      f"Expected at least {min_expected_heartbeats} heartbeats, but only got {heartbeat_count}."
  )
  return result

def test_gil_release():
  print("[Main] Generating heavy dummy text...")
  # 200k repetitions is enough to run for ~0.5-1.0s on standard hardware
  heavy_text = (
      "Hello, world! Testing SentencePiece GIL release behavior. " * 20000
  )

  model_path = "test_model.model"
  sp = spm.SentencePieceProcessor(model_file=os.path.join(HERE, model_path))

  # 1. Test Encode
  tokens = run_heavy_op_with_gil_check(lambda: sp.encode(heavy_text), "Encode")
  assert len(tokens) > 0, "Tokenization result is empty"

  # 2. Test Decode
  decoded = run_heavy_op_with_gil_check(lambda: sp.decode(tokens), "Decode")
  assert len(decoded) > 0, "Decode result is empty"

  # 3. Test ParallelEncode
  heavy_texts = [heavy_text] * 4
  parallel_tokens = run_heavy_op_with_gil_check(
      lambda: sp.parallel_encode(heavy_texts, num_threads=4, chunk_len=1000),
      "ParallelEncode"
  )
  assert len(parallel_tokens) == 4, "ParallelEncode batch size mismatch"

  print("\n[Main] All GIL release tests passed successfully!")

if __name__ == "__main__":
  test_gil_release()
