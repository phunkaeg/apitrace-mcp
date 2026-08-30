from __future__ import annotations

import json
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import register as registration


class RegisterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.claude = self.root / "claude_desktop_config.json"
        self.codex = self.root / "config.toml"
        self.server = self.root / "apitrace-mcp.exe"
        self.server.write_bytes(b"server")
        self.globals = mock.patch.multiple(
            registration,
            CLAUDE_DESKTOP=self.claude,
            CODEX=self.codex,
            SERVER_EXE=self.server,
        )
        self.globals.start()

    def tearDown(self) -> None:
        self.globals.stop()
        self.temp.cleanup()

    def backups_for(self, path: Path) -> list[Path]:
        return sorted(path.parent.glob(path.name + ".bak.*"))

    def test_backups_are_timestamped_unique_and_never_overwritten(self) -> None:
        self.codex.write_bytes(b"first")
        first = registration.backup(self.codex)
        self.codex.write_bytes(b"second")
        second = registration.backup(self.codex)

        self.assertNotEqual(first, second)
        self.assertRegex(first.name, r"config\.toml\.bak\.\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{8}")
        self.assertEqual(first.read_bytes(), b"first")
        self.assertEqual(second.read_bytes(), b"second")

    def test_backup_name_collision_preserves_existing_backup_and_retries(self) -> None:
        self.codex.write_bytes(b"current")
        stamp = "20260810T120000.000000Z"
        collision = self.root / f"config.toml.bak.{stamp}-aaaaaaaa"
        collision.write_bytes(b"existing backup")
        fake_datetime = mock.Mock()
        fake_datetime.now.return_value.strftime.return_value = stamp
        uuids = [
            SimpleNamespace(hex="aaaaaaaa00000000"),
            SimpleNamespace(hex="bbbbbbbb00000000"),
        ]

        with mock.patch.object(registration.dt, "datetime", fake_datetime):
            with mock.patch("register.uuid.uuid4", side_effect=uuids):
                created = registration.backup(self.codex)

        self.assertEqual(collision.read_bytes(), b"existing backup")
        self.assertEqual(created.name, f"config.toml.bak.{stamp}-bbbbbbbb")
        self.assertEqual(created.read_bytes(), b"current")

    def test_atomic_temp_collision_is_not_deleted_or_overwritten(self) -> None:
        self.codex.write_bytes(b"old")
        collision = self.root / ".config.toml.aaaaaaaa00000000.tmp"
        collision.write_bytes(b"someone else's temp")
        uuids = [
            SimpleNamespace(hex="aaaaaaaa00000000"),
            SimpleNamespace(hex="bbbbbbbb00000000"),
        ]

        with mock.patch("register.uuid.uuid4", side_effect=uuids):
            registration._atomic_write(self.codex, b"new")

        self.assertEqual(collision.read_bytes(), b"someone else's temp")
        self.assertEqual(self.codex.read_bytes(), b"new")

    def test_claude_add_remove_preserves_unrelated_entries(self) -> None:
        original = {
            "theme": "dark",
            "mcpServers": {
                "other": {"command": "other.exe", "args": ["--keep"]},
            },
            "unrelated": {"nested": [1, 2, 3]},
        }
        self.claude.write_text(json.dumps(original), encoding="utf-8")

        message = registration.claude_desktop(False)
        added = json.loads(self.claude.read_text(encoding="utf-8"))
        self.assertIn("registered", message)
        self.assertEqual(added["theme"], "dark")
        self.assertEqual(added["unrelated"], original["unrelated"])
        self.assertEqual(added["mcpServers"]["other"], original["mcpServers"]["other"])
        self.assertEqual(
            added["mcpServers"][registration.NAME]["command"], str(self.server)
        )
        self.assertEqual(len(self.backups_for(self.claude)), 1)

        message = registration.claude_desktop(True)
        removed = json.loads(self.claude.read_text(encoding="utf-8"))
        self.assertIn("removed", message)
        self.assertNotIn(registration.NAME, removed["mcpServers"])
        self.assertEqual(removed["mcpServers"]["other"], original["mcpServers"]["other"])
        self.assertEqual(removed["unrelated"], original["unrelated"])
        self.assertEqual(len(self.backups_for(self.claude)), 2)

    def test_claude_does_not_overwrite_different_existing_entry(self) -> None:
        payload = {"mcpServers": {registration.NAME: {"command": "custom.exe"}}}
        original = json.dumps(payload).encode("utf-8")
        self.claude.write_bytes(original)

        with self.assertRaisesRegex(registration.RegistrationError, "refusing to overwrite"):
            registration.claude_desktop(False)

        self.assertEqual(self.claude.read_bytes(), original)
        self.assertFalse(self.backups_for(self.claude))

    def test_claude_reports_json_location_without_writing(self) -> None:
        self.claude.write_text('{"mcpServers": ', encoding="utf-8")
        with self.assertRaisesRegex(registration.RegistrationError, r"line 1, column"):
            registration.claude_desktop(False)
        self.assertFalse(self.backups_for(self.claude))

    def test_codex_add_preserves_original_bytes_and_newline_style(self) -> None:
        original = b'theme = "dark"\r\n\r\n[mcp_servers.other]\r\ncommand = "other.exe"\r\n'
        self.codex.write_bytes(original)

        message = registration.codex(False)

        result = self.codex.read_bytes()
        self.assertIn("registered", message)
        self.assertTrue(result.startswith(original))
        self.assertNotIn(b"\n", result.replace(b"\r\n", b""))
        parsed = tomllib.loads(result.decode("utf-8"))
        self.assertEqual(parsed["theme"], "dark")
        self.assertEqual(parsed["mcp_servers"]["other"]["command"], "other.exe")
        self.assertEqual(
            parsed["mcp_servers"][registration.NAME]["command"], str(self.server)
        )
        self.assertEqual(parsed["mcp_servers"][registration.NAME]["args"], [])

    def test_codex_remove_includes_nested_target_tables_only(self) -> None:
        prefix = 'theme = "dark"\n\n'
        target = (
            '[mcp_servers.apitrace]\ncommand = "old.exe"\n'
            '[mcp_servers.apitrace.env]\nKEEP_OUT = "remove with target"\n\n'
        )
        unrelated = (
            '[mcp_servers.other]\ncommand = "other.exe"\n\n'
            '[tui]\nnotifications = true\n'
        )
        self.codex.write_text(prefix + target + unrelated, encoding="utf-8", newline="")

        registration.codex(True)

        result = self.codex.read_text(encoding="utf-8")
        self.assertEqual(result, prefix + unrelated)
        parsed = tomllib.loads(result)
        self.assertEqual(parsed["mcp_servers"]["other"]["command"], "other.exe")
        self.assertNotIn(registration.NAME, parsed["mcp_servers"])
        self.assertTrue(parsed["tui"]["notifications"])

    def test_codex_remove_handles_quoted_table_name_at_eof(self) -> None:
        original = (
            'keep = "yes"\n\n[mcp_servers."apitrace"]\ncommand = "old.exe"'
        )
        self.codex.write_text(original, encoding="utf-8", newline="")

        registration.codex(True)

        self.assertEqual(self.codex.read_text(encoding="utf-8"), 'keep = "yes"\n\n')

    def test_codex_remove_keeps_comment_block_of_following_table(self) -> None:
        prefix = 'theme = "dark"\n\n'
        target = '[mcp_servers.apitrace]\ncommand = "old.exe"\n\n'
        unrelated = (
            '# other server settings\n'
            '# keep this documentation\n'
            '[mcp_servers.other]\ncommand = "other.exe"\n'
        )
        self.codex.write_text(prefix + target + unrelated, encoding="utf-8", newline="")

        registration.codex(True)

        result = self.codex.read_text(encoding="utf-8")
        self.assertEqual(result, prefix + unrelated)
        parsed = tomllib.loads(result)
        self.assertNotIn(registration.NAME, parsed["mcp_servers"])
        self.assertEqual(parsed["mcp_servers"]["other"]["command"], "other.exe")

    def test_codex_remove_without_comments_still_removes_cleanly(self) -> None:
        prefix = 'theme = "dark"\n\n'
        target = '[mcp_servers.apitrace]\ncommand = "old.exe"\n\n'
        unrelated = '[mcp_servers.other]\ncommand = "other.exe"\n'
        self.codex.write_text(prefix + target + unrelated, encoding="utf-8", newline="")

        registration.codex(True)

        result = self.codex.read_text(encoding="utf-8")
        self.assertEqual(result, prefix + unrelated)
        parsed = tomllib.loads(result)
        self.assertNotIn(registration.NAME, parsed["mcp_servers"])

    def test_codex_refuses_inline_target_it_cannot_remove_safely(self) -> None:
        original = (
            '[mcp_servers]\n'
            'apitrace = { command = "custom.exe", args = ["keep"] }\n'
            'other = { command = "other.exe" }\n'
        ).encode("utf-8")
        self.codex.write_bytes(original)

        with self.assertRaisesRegex(registration.RegistrationError, "cannot safely remove"):
            registration.codex(True)

        self.assertEqual(self.codex.read_bytes(), original)
        self.assertFalse(self.backups_for(self.codex))

    def test_codex_different_existing_entry_is_not_overwritten(self) -> None:
        original = b'[mcp_servers.apitrace]\ncommand = "custom.exe"\n'
        self.codex.write_bytes(original)
        with self.assertRaisesRegex(registration.RegistrationError, "refusing to overwrite"):
            registration.codex(False)
        self.assertEqual(self.codex.read_bytes(), original)

    def test_atomic_replace_failure_leaves_original_and_backup(self) -> None:
        original = b'theme = "dark"\n'
        self.codex.write_bytes(original)
        with mock.patch("register.os.replace", side_effect=OSError("locked")):
            with self.assertRaisesRegex(
                registration.RegistrationError, "original remains in place"
            ):
                registration.codex(False)

        self.assertEqual(self.codex.read_bytes(), original)
        backups = self.backups_for(self.codex)
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        self.assertFalse(list(self.root.glob(".config.toml.*.tmp")))


if __name__ == "__main__":
    unittest.main()
