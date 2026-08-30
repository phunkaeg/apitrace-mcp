"""Resource-boundary tests for the isolated apitrace pickle decoder."""

from __future__ import annotations

import base64
import io
import pickle
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from apitrace_mcp import pickleparse


def record_with(value: object) -> tuple:
    return (1, 0, "TestCall", [("value", value)], None, 0)


def producer_command(record: object, *, sleep_after: float = 0.0) -> list[str]:
    encoded = base64.b64encode(pickle.dumps(record, protocol=4)).decode("ascii")
    script = (
        "import base64,sys,time;"
        f"sys.stdout.buffer.write(base64.b64decode('{encoded}'));"
        "sys.stdout.buffer.flush();"
        f"time.sleep({sleep_after!r})"
    )
    return [sys.executable, "-c", script]


class WorkerLimitTests(unittest.TestCase):
    def decode(self, record: object, **kwargs: object) -> list[pickleparse.Call]:
        with tempfile.NamedTemporaryFile(suffix=".trace") as trace:
            command = producer_command(record)
            with mock.patch.object(pickleparse, "pickle_cmd", return_value=command):
                return list(pickleparse.iter_calls(None, trace.name, timeout=10, **kwargs))

    def test_worker_round_trip_preserves_structured_values(self) -> None:
        value = {"matrix": [float(i) for i in range(16)], "blob": b"abc"}
        calls = self.decode(record_with(value))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arg("value"), value)

    def test_worker_rejects_record_node_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "node budget"):
            self.decode(record_with(list(range(100))), max_record_nodes=20)

    def test_worker_rejects_oversized_blob_before_returning_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "blob budget"):
            self.decode(record_with(b"x" * 4096), max_record_blob_bytes=1024)

    def test_worker_rejects_record_larger_than_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame exceeds"):
            self.decode(
                record_with(b"x" * 4096),
                max_record_blob_bytes=8192,
                max_frame_bytes=1024,
            )

    def test_worker_rejects_deep_record(self) -> None:
        nested: object = 1.0
        for _ in range(20):
            nested = [nested]
        with self.assertRaisesRegex(ValueError, "nesting depth"):
            self.decode(record_with(nested), max_record_depth=8)

    def test_worker_rejects_cyclic_record(self) -> None:
        cyclic: list[object] = []
        cyclic.append(cyclic)
        with self.assertRaisesRegex(ValueError, "container cycle"):
            self.decode(record_with(cyclic))

    def test_parent_rejects_oversized_frame_header_without_body_allocation(self) -> None:
        stream = io.BytesIO(struct.pack(">I", 10_000_000))
        with self.assertRaisesRegex(ValueError, "invalid frame size"):
            pickleparse._read_frame(stream, 1024)

    def test_timeout_kills_decoder_and_apitrace_process_trees(self) -> None:
        seen: list[int] = []
        seen_lock = threading.Lock()
        real_kill = pickleparse._kill_process_tree

        def tracked_kill(
            proc: subprocess.Popen, job: pickleparse._DecoderJob | None = None
        ) -> None:
            with seen_lock:
                seen.append(proc.pid)
            real_kill(proc, job)

        command = [sys.executable, "-c", "import time;time.sleep(30)"]
        with tempfile.NamedTemporaryFile(suffix=".trace") as trace:
            with mock.patch.object(pickleparse, "pickle_cmd", return_value=command):
                with mock.patch.object(
                    pickleparse, "_kill_process_tree", side_effect=tracked_kill
                ):
                    with self.assertRaisesRegex(RuntimeError, "timed out"):
                        list(pickleparse.iter_calls(None, trace.name, timeout=0.15))

        self.assertGreaterEqual(len(set(seen)), 2)


class TraversalBudgetTests(unittest.TestCase):
    def test_json_conversion_detects_cycle_and_obeys_shared_node_budget(self) -> None:
        cyclic: list[object] = []
        cyclic.append(cyclic)
        self.assertEqual(pickleparse.to_jsonable(cyclic), "<cycle>")
        rendered = pickleparse.to_jsonable([1, 2, 3, 4], max_nodes=2)
        self.assertIn("<node-budget-exceeded>", rendered)

    def test_json_blob_summary_does_not_copy_mutable_blob(self) -> None:
        blob = bytearray(b"abcdef")
        rendered = pickleparse.to_jsonable(blob)
        self.assertEqual(rendered["blob_bytes"], 6)
        self.assertEqual(rendered["head_hex"], "616263646566")

    def test_harvest_rejects_cycles_depth_nodes_and_float_overflow(self) -> None:
        cyclic: list[object] = []
        cyclic.append(cyclic)
        with self.assertRaisesRegex(ValueError, "container cycle"):
            pickleparse.harvest_floats(cyclic)

        nested: object = 1.0
        for _ in range(10):
            nested = [nested]
        with self.assertRaisesRegex(ValueError, "nesting depth"):
            pickleparse.harvest_floats(nested, max_depth=4)
        with self.assertRaisesRegex(ValueError, "node budget"):
            pickleparse.harvest_floats([1.0, 2.0, 3.0], max_nodes=2)
        with self.assertRaisesRegex(ValueError, "float budget"):
            pickleparse.harvest_floats([1.0, 2.0], max_floats=1)

    def test_harvest_preserves_order_without_recursive_stack_use(self) -> None:
        value = {"a": [1.0, (2.0,)], "b": {"c": 3.0}}
        self.assertEqual(pickleparse.harvest_floats(value), [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
