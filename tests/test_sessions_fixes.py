"""Regression tests for cross-instance session persistence fixes.

Covers: merge-before-write so one server instance's save cannot erase
another's live session; retry on transient sharing violations during
os.replace; and proportionate handling of persistence failures in start()
(no kill of a healthy launch) and stop() (warning, not a replaced result).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apitrace_mcp import sessions  # noqa: E402


class _FakeProc:
    def __init__(self, pid: int, code: int | None = None) -> None:
        self.pid = pid
        self.code = code
        self.waited = False
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.code

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return 0 if self.code is None else self.code

    def terminate(self) -> None:
        self.terminated = True
        self.code = -15

    def kill(self) -> None:
        self.killed = True
        self.code = -9


def _sleeper_cmd() -> list[str]:
    return [sys.executable, "-c", "import time; time.sleep(60)"]


def _kill_pid(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        creationflags=sessions.CREATE_NO_WINDOW,
        check=False,
    )


def _wait_deletable(directory: Path, timeout: float = 10.0) -> None:
    """Block until every log file in the directory can be unlinked.

    A taskkill'd child releases its inherited log handle asynchronously; a
    fixed sleep flakes under load, so retry the unlink itself -- once it
    succeeds, TemporaryDirectory cleanup has nothing left to race.
    """
    deadline = time.monotonic() + timeout
    pending = list(directory.glob("*.log"))
    while pending and time.monotonic() < deadline:
        remaining = []
        for path in pending:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                remaining.append(path)
        pending = remaining
        if pending:
            time.sleep(0.1)


@unittest.skipUnless(os.name == "nt", "session manager targets Windows processes")
class CrossInstancePersistenceTests(unittest.TestCase):
    def test_stop_in_one_instance_preserves_other_instances_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_dir = Path(temp)
            manager_a = sessions.SessionManager()
            manager_b = sessions.SessionManager()
            first = second = None
            try:
                first = manager_a.start("trace", _sleeper_cmd(), log_dir=log_dir)
                second = manager_b.start("replay", _sleeper_cmd(), log_dir=log_dir)

                result = manager_a.stop(first.id, force=True)
                self.assertFalse(result["running"])

                records = json.loads(
                    (log_dir / "sessions.json").read_text(encoding="utf-8")
                )
                ids = {rec["id"] for rec in records}
                # A's save must carry forward B's still-running session
                # instead of overwriting the store with A's snapshot alone.
                self.assertIn(second.id, ids)
                self.assertNotIn(first.id, ids)
            finally:
                for manager, session in (
                    (manager_a, first),
                    (manager_b, second),
                ):
                    if session is None:
                        continue
                    try:
                        if session.running:
                            manager.stop(session.id, force=True)
                    except (RuntimeError, KeyError):
                        _kill_pid(session.pid)
                    session._close_log_handle()
                _wait_deletable(log_dir)

    def test_save_retries_transient_sharing_violations_on_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_dir = Path(temp)
            manager = sessions.SessionManager()
            manager._store = log_dir / "sessions.json"
            proc = _FakeProc(100)
            session = sessions.Session(
                "trace-live",
                "trace",
                ["tool.exe"],
                100,
                log_dir / "live.log",
                proc=proc,
                process_created_filetime=1000,
                process_executable=sessions._normalise_executable("tool.exe"),
            )
            manager.sessions[session.id] = session

            real_replace = os.replace
            attempts = {"count": 0}

            def flaky_replace(src, dst):
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise OSError(22, "sharing violation", None, 32)
                return real_replace(src, dst)

            with mock.patch(
                "apitrace_mcp.sessions.os.replace", side_effect=flaky_replace
            ):
                with mock.patch("apitrace_mcp.sessions.time.sleep"):
                    manager._save()

            self.assertEqual(attempts["count"], 3)
            records = json.loads(manager._store.read_text(encoding="utf-8"))
            self.assertEqual([rec["id"] for rec in records], ["trace-live"])
            self.assertFalse(list(log_dir.glob(".sessions.json.*.tmp")))


@unittest.skipUnless(os.name == "nt", "session manager targets Windows processes")
class ProportionateFailureHandlingTests(unittest.TestCase):
    def test_start_save_failure_keeps_session_alive_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_dir = Path(temp)
            manager = sessions.SessionManager()
            proc = _FakeProc(100)
            identity = sessions.ProcessIdentity(
                1000, sessions._normalise_executable("tool.exe")
            )
            with mock.patch("apitrace_mcp.sessions.subprocess.Popen", return_value=proc):
                with mock.patch(
                    "apitrace_mcp.sessions._query_process_identity",
                    return_value=identity,
                ):
                    with mock.patch("apitrace_mcp.sessions.subprocess.run") as run:
                        with mock.patch.object(
                            manager,
                            "_save_locked",
                            side_effect=RuntimeError("store unavailable"),
                        ):
                            session = manager.start(
                                "trace", ["tool.exe"], log_dir=log_dir
                            )

            # The freshly launched process must not be killed over a
            # transient persistence failure.
            run.assert_not_called()
            self.assertFalse(proc.terminated)
            self.assertFalse(proc.killed)
            self.assertIn(session.id, manager.sessions)
            self.assertIsNotNone(session.persist_warning)
            self.assertIn("not persisted", session.persist_warning)
            status = session.status()
            self.assertTrue(status["running"])
            self.assertIn("persist_warning", status)
            session._close_log_handle()

    def test_stop_save_failure_reports_warning_not_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_dir = Path(temp)
            manager = sessions.SessionManager()
            manager._store = log_dir / "sessions.json"
            proc = _FakeProc(100, 7)  # already exited
            session = sessions.Session(
                "trace-done",
                "trace",
                ["tool.exe"],
                100,
                log_dir / "done.log",
                proc=proc,
            )
            manager.sessions[session.id] = session

            with mock.patch.object(
                manager,
                "_save_locked",
                side_effect=RuntimeError("store unavailable"),
            ):
                result = manager.stop(session.id, force=True)

            # The successful stop outcome survives; the persistence failure
            # is attached as a warning instead of replacing the result.
            self.assertEqual(result["session"], "trace-done")
            self.assertFalse(result["running"])
            self.assertIn("persist_warning", result)
            self.assertIn("store unavailable", result["persist_warning"])


if __name__ == "__main__":
    unittest.main()
