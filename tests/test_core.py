from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from apitrace_mcp import config, pe, sessions, wrappers
from apitrace_mcp.config import ApitraceRoot


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WrapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root_dir = self.base / "apitrace-14.0-win64"
        (self.root_dir / "lib" / "wrappers").mkdir(parents=True)
        (self.root_dir / "lib" / "wrappers" / "d3d9.dll").write_bytes(
            b"apitrace-wrapper"
        )
        self.root = ApitraceRoot(self.root_dir, 64, "14.0")
        self.game = self.base / "Gäme 100%"
        self.game.mkdir()
        self.exe = self.game / "gäme 100%.exe"
        self.exe.write_bytes(b"game")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_install_and_uninstall_restore_original_transactionally(self) -> None:
        target = self.game / "d3d9.dll"
        target.write_bytes(b"original-shim")
        trace = self.base / "träces" / "100% run.trace"

        result = wrappers.install(self.root, self.exe, "d3d9", trace_file=trace)

        backup = self.game / "d3d9.dll.apitrace-backup"
        manifest_path = self.game / wrappers.MANIFEST_NAME
        launcher = self.game / wrappers.LAUNCHER_NAME
        self.assertEqual(target.read_bytes(), b"apitrace-wrapper")
        self.assertEqual(backup.read_bytes(), b"original-shim")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], 2)
        self.assertEqual(manifest["state"], "installed")
        self.assertEqual(manifest["files"][0]["installed_sha256"], sha256(target))
        self.assertEqual(manifest["files"][0]["original_sha256"], sha256(backup))
        launcher_bytes = launcher.read_bytes()
        self.assertTrue(launcher_bytes.startswith(b"\xef\xbb\xbf"))
        launcher_text = launcher_bytes.decode("utf-8-sig")
        self.assertIn('pushd "%~dp0"', launcher_text)
        self.assertIn("100%% run.trace", launcher_text)
        self.assertIn('if exist "%TRACE_FILE%"', launcher_text)
        self.assertIn("Refusing to overwrite existing apitrace capture", launcher_text)
        self.assertIn('"gäme 100%%.exe" %*', launcher_text)
        self.assertEqual(result["launcher"], str(launcher))

        removal = wrappers.uninstall(self.exe)
        self.assertFalse(removal["manifest_retained"])
        self.assertEqual(target.read_bytes(), b"original-shim")
        self.assertFalse(backup.exists())
        self.assertFalse(launcher.exists())
        self.assertFalse(manifest_path.exists())

    def test_force_never_overwrites_manifest_or_backup(self) -> None:
        target = self.game / "d3d9.dll"
        target.write_bytes(b"original-shim")
        wrappers.install(self.root, self.exe, "d3d9")
        manifest_path = self.game / wrappers.MANIFEST_NAME
        backup = self.game / "d3d9.dll.apitrace-backup"
        before_manifest = manifest_path.read_bytes()
        before_backup = backup.read_bytes()

        with self.assertRaisesRegex(RuntimeError, "force never overwrites"):
            wrappers.install(self.root, self.exe, "d3d9", force=True)

        self.assertEqual(manifest_path.read_bytes(), before_manifest)
        self.assertEqual(backup.read_bytes(), before_backup)

    def test_corrupt_manifest_is_not_treated_as_absent(self) -> None:
        manifest_path = self.game / wrappers.MANIFEST_NAME
        manifest_path.write_text("not json", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "corrupt"):
            wrappers.install(self.root, self.exe, "d3d9", force=True)
        self.assertEqual(manifest_path.read_text(encoding="utf-8"), "not json")

    def test_uninstall_refuses_changed_wrapper_and_retains_manifest(self) -> None:
        target = self.game / "d3d9.dll"
        target.write_bytes(b"original-shim")
        wrappers.install(self.root, self.exe, "d3d9")
        target.write_bytes(b"user changed this after install")

        result = wrappers.uninstall(self.game)

        self.assertTrue(result["manifest_retained"])
        self.assertIn("changed since installation", " ".join(result["problems"]))
        self.assertEqual(target.read_bytes(), b"user changed this after install")
        self.assertEqual(
            (self.game / "d3d9.dll.apitrace-backup").read_bytes(), b"original-shim"
        )

    def test_uninstall_revalidates_replacement_after_preflight(self) -> None:
        """A file swapped after initial hashes must never be quarantined/deleted."""
        target = self.game / "d3d9.dll"
        target.write_bytes(b"original-shim")
        wrappers.install(self.root, self.exe, "d3d9")
        real_update = wrappers._update_manifest
        raced = False

        def replace_after_preflight(path: Path, manifest: dict) -> None:
            nonlocal raced
            real_update(path, manifest)
            if manifest["state"] == "uninstalling" and not raced:
                raced = True
                target.write_bytes(b"user replacement during uninstall")

        with mock.patch(
            "apitrace_mcp.wrappers._update_manifest",
            side_effect=replace_after_preflight,
        ):
            result = wrappers.uninstall(self.game)

        self.assertTrue(raced)
        self.assertTrue(result["manifest_retained"])
        self.assertIn("changed during the transaction", " ".join(result["problems"]))
        self.assertEqual(target.read_bytes(), b"user replacement during uninstall")
        self.assertEqual(
            (self.game / "d3d9.dll.apitrace-backup").read_bytes(), b"original-shim"
        )
        self.assertTrue((self.game / wrappers.MANIFEST_NAME).exists())

    def test_wrapper_manifestless_probe_ignores_held_lock(self) -> None:
        # A probe of a directory with no manifest must answer without
        # contending for (or creating) the transaction lock; contention on a
        # real uninstall is covered in test_wrappers_fixes.
        with wrappers._transaction_lock(self.game):
            result = wrappers.uninstall(self.game)
        self.assertIn("nothing to undo", result["note"])

    def test_uninstall_does_not_unlink_replaced_manifest(self) -> None:
        target = self.game / "d3d9.dll"
        target.write_bytes(b"original-shim")
        wrappers.install(self.root, self.exe, "d3d9")
        manifest_path = self.game / wrappers.MANIFEST_NAME
        replacement = b'{"owned_by": "someone else"}\n'
        real_update = wrappers._update_manifest

        def replace_final_manifest(path: Path, manifest: dict) -> None:
            real_update(path, manifest)
            if manifest["state"] == "uninstalled":
                path.write_bytes(replacement)

        with mock.patch(
            "apitrace_mcp.wrappers._update_manifest",
            side_effect=replace_final_manifest,
        ):
            result = wrappers.uninstall(self.game)

        self.assertTrue(result["manifest_retained"])
        self.assertEqual(manifest_path.read_bytes(), replacement)
        self.assertEqual(target.read_bytes(), b"original-shim")

    def test_manifest_traversal_is_rejected(self) -> None:
        wrappers.install(self.root, self.exe, "d3d9")
        manifest_path = self.game / wrappers.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][0]["installed_to"] = "../victim.dll"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        victim = self.base / "victim.dll"
        victim.write_bytes(b"keep")

        with self.assertRaisesRegex(RuntimeError, "filename"):
            wrappers.uninstall(self.game)
        self.assertEqual(victim.read_bytes(), b"keep")

    def test_failed_install_rolls_back_but_keeps_recovery_manifest(self) -> None:
        target = self.game / "d3d9.dll"
        target.write_bytes(b"original-shim")
        real_rename = wrappers._rename_no_replace
        failed = False

        def fail_stage_once(source: str | os.PathLike, destination: str | os.PathLike) -> None:
            nonlocal failed
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                not failed
                and source_path.name.startswith(".apitrace-mcp-stage-")
                and destination_path.name == "d3d9.dll"
            ):
                failed = True
                raise OSError("simulated copy activation failure")
            real_rename(Path(source), Path(destination))

        with mock.patch(
            "apitrace_mcp.wrappers._rename_no_replace", side_effect=fail_stage_once
        ):
            with self.assertRaisesRegex(RuntimeError, "recovery manifest retained"):
                wrappers.install(self.root, self.exe, "d3d9")

        self.assertEqual(target.read_bytes(), b"original-shim")
        self.assertFalse((self.game / "d3d9.dll.apitrace-backup").exists())
        manifest = wrappers.read_manifest(self.game)
        self.assertEqual(manifest["state"], "failed")
        self.assertIn("simulated copy activation failure", manifest["install_error"])
        cleanup = wrappers.uninstall(self.game)
        self.assertFalse(cleanup["manifest_retained"])


class _TrackingBytesIO(io.BytesIO):
    def __init__(self, value: bytes) -> None:
        super().__init__(value)
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)


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


class SessionTests(unittest.TestCase):
    def test_log_tail_reads_only_requested_bytes(self) -> None:
        stream = _TrackingBytesIO(b"a" * 10000 + b"TAIL")
        session = sessions.Session("trace-x", "trace", ["x"], 1, Path("unused"))
        with mock.patch.object(Path, "open", return_value=stream):
            value = session.log_tail(4)
        self.assertEqual(value, "TAIL")
        self.assertEqual(stream.read_sizes, [4])

    def test_natural_exit_finalizes_process_and_log_handles(self) -> None:
        proc = _FakeProc(10, 7)
        log_handle = mock.Mock()
        session = sessions.Session(
            "trace-x", "trace", ["x"], 10, Path("unused"), proc=proc, log_handle=log_handle
        )
        self.assertFalse(session.running)
        self.assertEqual(session.exit_code, 7)
        self.assertIsNone(session.proc)
        self.assertTrue(proc.waited)
        log_handle.close.assert_called_once_with()

    def test_detached_stop_rejects_recycled_pid_before_taskkill(self) -> None:
        manager = sessions.SessionManager()
        session = sessions.Session(
            "trace-x",
            "trace",
            [r"C:\Tools\apitrace.exe"],
            123,
            Path("unused"),
            detached=True,
            process_created_filetime=111,
            process_executable=r"C:\Tools\apitrace.exe",
        )
        manager.sessions[session.id] = session
        recycled = sessions.ProcessIdentity(222, sessions._normalise_executable(r"C:\Other.exe"))
        with mock.patch("apitrace_mcp.sessions._query_process_identity", return_value=recycled):
            with mock.patch("apitrace_mcp.sessions.subprocess.run") as run:
                with self.assertRaisesRegex(RuntimeError, "refusing to stop"):
                    manager.stop(session.id, force=True)
                run.assert_not_called()

    def test_in_memory_stop_rechecks_identity_after_running_poll(self) -> None:
        manager = sessions.SessionManager()
        proc = _FakeProc(123, None)
        recorded_exe = sessions._normalise_executable(r"C:\Tools\apitrace.exe")
        session = sessions.Session(
            "trace-live",
            "trace",
            [r"C:\Tools\apitrace.exe"],
            123,
            Path("unused"),
            proc=proc,
            process_created_filetime=111,
            process_executable=recorded_exe,
        )
        manager.sessions[session.id] = session
        recycled = sessions.ProcessIdentity(
            222, sessions._normalise_executable(r"C:\Other.exe")
        )

        with mock.patch(
            "apitrace_mcp.sessions._query_process_identity", return_value=recycled
        ):
            with mock.patch("apitrace_mcp.sessions.subprocess.run") as run:
                with self.assertRaisesRegex(RuntimeError, "no longer has the recorded"):
                    manager.stop(session.id, force=True)
                run.assert_not_called()

    def test_session_records_require_creation_time_and_executable(self) -> None:
        record = {
            "id": "trace-old",
            "kind": "trace",
            "cmd": ["tool.exe"],
            "pid": 123,
            "log_path": "trace.log",
            "started_at": 1.0,
        }
        with self.assertRaises(TypeError):
            sessions.Session.from_record(record)

    def test_save_is_atomic_and_ids_do_not_reuse_log_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_dir = Path(temp)
            manager = sessions.SessionManager()
            fake_processes = [_FakeProc(100), _FakeProc(101)]
            identities = [
                sessions.ProcessIdentity(1000, sessions._normalise_executable("tool.exe")),
                sessions.ProcessIdentity(1001, sessions._normalise_executable("tool.exe")),
            ]
            with mock.patch(
                "apitrace_mcp.sessions.subprocess.Popen", side_effect=fake_processes
            ):
                with mock.patch(
                    "apitrace_mcp.sessions._query_process_identity", side_effect=identities
                ):
                    first = manager.start("trace", ["tool.exe"], log_dir=log_dir)
                    second = manager.start("trace", ["tool.exe"], log_dir=log_dir)
            self.assertNotEqual(first.id, second.id)
            self.assertNotEqual(first.log_path, second.log_path)
            records = json.loads((log_dir / "sessions.json").read_text(encoding="utf-8"))
            self.assertEqual({item["id"] for item in records}, {first.id, second.id})
            self.assertFalse(list(log_dir.glob(".sessions.json.*.tmp")))
            first._close_log_handle()
            second._close_log_handle()

    def test_save_failure_is_surfaced_and_original_store_survives(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_dir = Path(temp)
            store = log_dir / "sessions.json"
            original = b"[]\n"
            store.write_bytes(original)
            manager = sessions.SessionManager()
            manager._store = store
            proc = _FakeProc(100)
            session = sessions.Session(
                "trace-live", "trace", ["tool.exe"], 100, log_dir / "live.log", proc=proc
            )
            manager.sessions[session.id] = session

            with mock.patch(
                "apitrace_mcp.sessions.os.replace", side_effect=OSError("disk failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot durably save sessions"):
                    manager._save()

            self.assertEqual(store.read_bytes(), original)
            self.assertFalse(list(log_dir.glob(".sessions.json.*.tmp")))

    # A save failure during start() must NOT terminate the freshly launched
    # process -- that behaviour (asserted by a test formerly here) let a purely
    # transient sessions.json write race kill a healthy game. The session now
    # survives with a persist warning; see
    # tests/test_sessions_fixes.py::test_start_save_failure_keeps_session_alive_with_warning.

    def test_start_identity_failure_terminates_process_without_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_dir = Path(temp)
            manager = sessions.SessionManager()
            proc = _FakeProc(100)
            with mock.patch("apitrace_mcp.sessions.subprocess.Popen", return_value=proc):
                with mock.patch(
                    "apitrace_mcp.sessions._query_process_identity", return_value=None
                ):
                    with mock.patch("apitrace_mcp.sessions.time.sleep"):
                        with self.assertRaisesRegex(
                            RuntimeError, "could not obtain a durable process identity"
                        ):
                            manager.start("trace", ["tool.exe"], log_dir=log_dir)

            self.assertTrue(proc.terminated)
            self.assertFalse(manager.sessions)
            self.assertFalse((log_dir / "sessions.json").exists())

    def test_concurrent_starts_do_not_lose_persisted_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_dir = Path(temp)
            manager = sessions.SessionManager()
            next_pid = 100
            pid_lock = threading.Lock()

            def make_proc(*args: object, **kwargs: object) -> _FakeProc:
                nonlocal next_pid
                with pid_lock:
                    proc = _FakeProc(next_pid)
                    next_pid += 1
                    return proc

            def identity(pid: int) -> sessions.ProcessIdentity:
                return sessions.ProcessIdentity(
                    pid * 10, sessions._normalise_executable("tool.exe")
                )

            with mock.patch(
                "apitrace_mcp.sessions.subprocess.Popen", side_effect=make_proc
            ):
                with mock.patch(
                    "apitrace_mcp.sessions._query_process_identity", side_effect=identity
                ):
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        created = list(
                            executor.map(
                                lambda _: manager.start(
                                    "trace", ["tool.exe"], log_dir=log_dir
                                ),
                                range(12),
                            )
                        )

            records = json.loads((log_dir / "sessions.json").read_text(encoding="utf-8"))
            self.assertEqual(len(created), 12)
            self.assertEqual(len({session.id for session in created}), 12)
            self.assertEqual({record["id"] for record in records}, set(manager.sessions))
            for session in created:
                session._close_log_handle()

    def test_concurrent_double_stop_is_rejected_atomically(self) -> None:
        manager = sessions.SessionManager()
        session = sessions.Session(
            "trace-live", "trace", ["tool.exe"], 100, Path("unused"), proc=_FakeProc(100)
        )
        manager.sessions[session.id] = session
        entered = threading.Event()
        release = threading.Event()

        def held_stop(current: sessions.Session, force: bool) -> dict:
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            return {"session": current.id}

        with mock.patch.object(manager, "_stop_session", side_effect=held_stop):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(manager.stop, session.id)
                self.assertTrue(entered.wait(timeout=5))
                with self.assertRaisesRegex(RuntimeError, "already being stopped"):
                    manager.stop(session.id)
                release.set()
                self.assertEqual(future.result(timeout=5), {"session": session.id})


class ConfigTests(unittest.TestCase):
    def test_semantic_version_extraction_and_order(self) -> None:
        self.assertEqual(config._version_of(Path("apitrace-14.0-win64")), "14.0")
        self.assertEqual(config._version_of(Path("apitrace-15.0.0-rc1-win32")), "15.0.0-rc1")
        self.assertGreater(config._version_key("14.0"), config._version_key("9.0"))
        self.assertGreater(config._version_key("15.0.0"), config._version_key("15.0.0-rc1"))

    def _make_root(self, parent: Path, name: str) -> Path:
        root = parent / name
        (root / "bin").mkdir(parents=True)
        (root / "lib" / "wrappers").mkdir(parents=True)
        (root / "bin" / "apitrace.exe").write_bytes(b"fake")
        return root

    def test_explicit_root_architecture_mismatch_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self._make_root(Path(temp), "apitrace-14.0-win32")
            env = {config.ENV_ROOT_WIN64: str(root)}
            with mock.patch.dict(os.environ, env, clear=True):
                with mock.patch("apitrace_mcp.config.read_pe", return_value=SimpleNamespace(bits=32)):
                    with mock.patch.object(config, "SEARCH_GLOBS", []):
                        with self.assertRaisesRegex(RuntimeError, "configured as win64"):
                            config.discover_roots()

    def test_invalid_explicit_root_does_not_silently_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing"
            with mock.patch.dict(
                os.environ, {config.ENV_ROOT_WIN64: str(missing)}, clear=True
            ):
                with mock.patch.object(config, "SEARCH_GLOBS", []):
                    with self.assertRaisesRegex(RuntimeError, "not an apitrace directory"):
                        config.discover_roots()

    def test_auto_discovery_chooses_highest_semantic_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            old = self._make_root(parent, "apitrace-9.0-win64")
            new = self._make_root(parent, "apitrace-14.0-win64")
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(config, "SEARCH_GLOBS", ["fake-pattern"]):
                    with mock.patch("apitrace_mcp.config.glob.glob", return_value=[str(old), str(new)]):
                        with mock.patch(
                            "apitrace_mcp.config.read_pe", return_value=SimpleNamespace(bits=64)
                        ):
                            roots = config.discover_roots()
            self.assertEqual(roots[64].path, new.resolve())
            self.assertEqual(roots[64].version, "14.0")

    def test_auto_discovery_ranks_versions_across_all_search_globs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            old = self._make_root(parent, "apitrace-9.0-win64")
            new = self._make_root(parent, "apitrace-14.0-win64")

            def hits(pattern: str) -> list[str]:
                return [str(old)] if pattern == "first-location" else [str(new)]

            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch.object(
                    config, "SEARCH_GLOBS", ["first-location", "second-location"]
                ):
                    with mock.patch(
                        "apitrace_mcp.config.glob.glob", side_effect=hits
                    ):
                        with mock.patch(
                            "apitrace_mcp.config.read_pe",
                            return_value=SimpleNamespace(bits=64),
                        ):
                            roots = config.discover_roots()

            self.assertEqual(roots[64].path, new.resolve())
            self.assertEqual(roots[64].version, "14.0")

    def test_explicit_root_still_beats_newer_automatic_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            explicit = self._make_root(parent, "apitrace-9.0-win64")
            automatic = self._make_root(parent, "apitrace-14.0-win64")
            with mock.patch.dict(
                os.environ, {config.ENV_ROOT_WIN64: str(explicit)}, clear=True
            ):
                with mock.patch.object(config, "SEARCH_GLOBS", ["automatic"]):
                    with mock.patch(
                        "apitrace_mcp.config.glob.glob", return_value=[str(automatic)]
                    ):
                        with mock.patch(
                            "apitrace_mcp.config.read_pe",
                            return_value=SimpleNamespace(bits=64),
                        ):
                            roots = config.discover_roots()

            self.assertEqual(roots[64].path, explicit.resolve())
            self.assertEqual(roots[64].version, "9.0")


def build_pe32_delay_import(
    *, attrs: int, directory_size: int = 64, second_outside_directory: bool = False
) -> bytes:
    image_base = 0x400000
    pe_offset = 0x80
    optional_size = 224
    raw_offset = 0x200
    raw_size = 0x400
    section_rva = 0x1000
    data = bytearray(raw_offset + raw_size)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, pe_offset)
    data[pe_offset : pe_offset + 4] = b"PE\0\0"
    struct.pack_into(
        "<HHIIIHH",
        data,
        pe_offset + 4,
        0x14C,
        1,
        0,
        0,
        0,
        optional_size,
        0x0002,
    )
    optional = pe_offset + 24
    struct.pack_into("<H", data, optional, 0x10B)
    struct.pack_into("<I", data, optional + 28, image_base)
    struct.pack_into("<I", data, optional + 60, raw_offset)
    struct.pack_into("<I", data, optional + 92, 16)
    struct.pack_into(
        "<II", data, optional + 96 + 13 * 8, section_rva, directory_size
    )
    section = optional + optional_size
    data[section : section + 8] = b".rdata\0\0"
    struct.pack_into("<IIII", data, section + 8, 0x800, section_rva, raw_size, raw_offset)

    name_rva = section_rva + 0x100
    name_value = name_rva if attrs & 1 else image_base + name_rva
    struct.pack_into("<II", data, raw_offset, attrs, name_value)
    data[raw_offset + 0x100 : raw_offset + 0x109] = b"d3d9.dll\0"
    if second_outside_directory:
        second_name_rva = section_rva + 0x120
        struct.pack_into("<II", data, raw_offset + 32, 1, second_name_rva)
        data[raw_offset + 0x120 : raw_offset + 0x12D] = b"opengl32.dll\0"
    return bytes(data)


class PETests(unittest.TestCase):
    def _read(self, payload: bytes) -> pe.PEInfo:
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as handle:
            handle.write(payload)
            path = handle.name
        try:
            return pe.read_pe(path)
        finally:
            Path(path).unlink()

    def test_old_delay_import_va_is_rebased_to_rva(self) -> None:
        info = self._read(build_pe32_delay_import(attrs=0))
        self.assertEqual(info.delay_imports, ["d3d9.dll"])

    def test_new_delay_import_rva_is_mapped_directly(self) -> None:
        info = self._read(build_pe32_delay_import(attrs=1))
        self.assertEqual(info.delay_imports, ["d3d9.dll"])

    def test_delay_descriptor_walk_is_bounded_by_directory_size(self) -> None:
        info = self._read(
            build_pe32_delay_import(
                attrs=1, directory_size=32, second_outside_directory=True
            )
        )
        self.assertEqual(info.delay_imports, ["d3d9.dll"])

    def test_rva_mapping_rejects_unbacked_virtual_tail(self) -> None:
        sections_list = [pe.Section(".x", 0x1000, 0x800, 0x200, 0x100)]
        self.assertIsNone(pe._rva_to_offset(0x1200, sections_list, file_size=0x1000))
        self.assertEqual(pe._rva_to_offset(0x1080, sections_list, file_size=0x1000), 0x280)


if __name__ == "__main__":
    unittest.main()
