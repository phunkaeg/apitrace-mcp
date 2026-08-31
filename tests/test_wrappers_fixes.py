from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apitrace_mcp import wrappers
from apitrace_mcp.config import ApitraceRoot


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WrapperFixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root_dir = self.base / "apitrace-14.0-win64"
        (self.root_dir / "lib" / "wrappers").mkdir(parents=True)
        (self.root_dir / "lib" / "wrappers" / "d3d9.dll").write_bytes(
            b"apitrace-wrapper"
        )
        self.root = ApitraceRoot(self.root_dir, 64, "14.0")
        self.game = self.base / "Game"
        self.game.mkdir()
        self.exe = self.game / "game.exe"
        self.exe.write_bytes(b"game")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def file_names(self) -> set[str]:
        return {entry.name for entry in self.game.iterdir()}

    def test_uninstall_completes_after_manual_copy_restore(self) -> None:
        target = self.game / "d3d9.dll"
        target.write_bytes(b"original-shim")
        wrappers.install(self.root, self.exe, "d3d9")
        backup = self.game / "d3d9.dll.apitrace-backup"
        self.assertEqual(target.read_bytes(), b"apitrace-wrapper")
        self.assertEqual(backup.read_bytes(), b"original-shim")

        # The user copy-restored the original by hand, leaving the backup.
        target.write_bytes(b"original-shim")

        result = wrappers.uninstall(self.game)

        self.assertEqual(result["problems"], [])
        self.assertFalse(result["manifest_retained"])
        self.assertIn(str(target), result["restored"])
        self.assertEqual(target.read_bytes(), b"original-shim")
        self.assertFalse(backup.exists())
        self.assertFalse((self.game / wrappers.MANIFEST_NAME).exists())

        # The directory is fully back to its pre-install state, so a fresh
        # install must not be blocked by leftovers.
        wrappers.install(self.root, self.exe, "d3d9")
        self.assertEqual(target.read_bytes(), b"apitrace-wrapper")

    def test_probe_uninstall_leaves_no_new_files(self) -> None:
        before = self.file_names()

        result = wrappers.uninstall(self.game)

        self.assertIn("nothing to undo", result["note"])
        self.assertEqual(self.file_names(), before)
        self.assertFalse((self.game / wrappers.LOCK_NAME).exists())

    def test_real_uninstall_still_rejects_concurrent_operation(self) -> None:
        target = self.game / "d3d9.dll"
        target.write_bytes(b"original-shim")
        wrappers.install(self.root, self.exe, "d3d9")

        with wrappers._transaction_lock(self.game):
            with self.assertRaisesRegex(RuntimeError, "already active"):
                wrappers.uninstall(self.game)
            # The loser must not have unlinked the held lock file.
            self.assertTrue((self.game / wrappers.LOCK_NAME).exists())
        self.assertFalse((self.game / wrappers.LOCK_NAME).exists())

    def test_install_uninstall_cycle_restores_exact_file_set(self) -> None:
        target = self.game / "d3d9.dll"
        target.write_bytes(b"original-shim")
        trace = self.base / "traces" / "run.trace"
        before = self.file_names()

        wrappers.install(self.root, self.exe, "d3d9", trace_file=trace)
        result = wrappers.uninstall(self.game)

        self.assertEqual(result["problems"], [])
        self.assertEqual(self.file_names(), before)
        self.assertFalse((self.game / wrappers.LOCK_NAME).exists())
        self.assertEqual(target.read_bytes(), b"original-shim")

    def test_launcher_has_no_bom_and_starts_with_echo_off(self) -> None:
        # cmd.exe does not skip a UTF-8 BOM in a batch file: the bytes join the
        # first command, so "@echo off" is never recognised. A real Morrowind
        # capture failed to launch on exactly this -- the file was present and
        # its text read correctly, but the batch would not run.
        trace = Path("D:/traces/mw.trace")
        payload = wrappers._launcher_bytes("Morrowind.exe", trace)
        self.assertFalse(
            payload.startswith(bytes([0xEF, 0xBB, 0xBF])), "launcher carries a BOM"
        )
        self.assertTrue(payload.startswith(b"@echo off" + bytes([13, 10])))
        self.assertIn(b"chcp 65001", payload)
        text = payload.decode("utf-8")
        self.assertIn(str(trace), text)


if __name__ == "__main__":
    unittest.main()
