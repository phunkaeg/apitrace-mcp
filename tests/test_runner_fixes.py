from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apitrace_mcp import runner


class JoinReadersStuckPipeTests(unittest.TestCase):
    """Regression: _join_readers must not hang on a reader stuck in a kernel read.

    The old fallback called stream.close() from the joining thread, which blocks
    on the BufferedReader's internal lock held by the blocked reader -- turning a
    bounded cleanup into an unbounded hang until the leaked pipe writer exits.
    """

    def test_join_readers_returns_promptly_with_blocked_reader(self) -> None:
        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, "rb")
        output = bytearray()
        state = {"truncated": False}
        reader = threading.Thread(
            target=runner._read_bounded,
            args=(stream, 64, output, state),
            daemon=True,
        )
        reader.start()
        try:
            # Let the reader park inside the blocking read on the open pipe.
            time.sleep(0.2)
            self.assertTrue(reader.is_alive())

            # Watchdog: call _join_readers from a helper thread so this test
            # itself can never deadlock if the regression comes back.
            done = threading.Event()

            def call() -> None:
                runner._join_readers([reader], timeout=1.0)
                done.set()

            caller = threading.Thread(target=call, daemon=True)
            start = time.monotonic()
            caller.start()
            finished = done.wait(timeout=8.0)
            elapsed = time.monotonic() - start
            self.assertTrue(
                finished,
                "_join_readers blocked behind a reader stuck in a kernel read",
            )
            self.assertLess(elapsed, 8.0)
            # The stuck reader is abandoned, not force-closed: closing its
            # stream from another thread is exactly what used to deadlock.
            self.assertTrue(reader.is_alive())
            self.assertFalse(stream.closed)
        finally:
            # Release the daemon reader by closing the write end (EOF).
            os.close(write_fd)
        reader.join(timeout=5.0)
        self.assertFalse(reader.is_alive())
        # _read_bounded closes its stream itself once the read returns.
        self.assertTrue(stream.closed)

    def test_join_readers_joins_finished_readers(self) -> None:
        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, "rb")
        output = bytearray()
        state = {"truncated": False}
        reader = threading.Thread(
            target=runner._read_bounded,
            args=(stream, 64, output, state),
            daemon=True,
        )
        reader.start()
        os.write(write_fd, b"hello")
        os.close(write_fd)
        runner._join_readers([reader], timeout=5.0)
        self.assertFalse(reader.is_alive())
        self.assertEqual(bytes(output), b"hello")


if __name__ == "__main__":
    unittest.main()
