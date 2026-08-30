"""Focused regression tests for the MCP boundary and CLI orchestration."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apitrace_mcp import server  # noqa: E402
from apitrace_mcp.runner import CommandError, run  # noqa: E402
from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402


class _Root:
    apitrace_exe = Path(sys.executable)

    def retrace_exe(self, api: str) -> Path:
        return Path(sys.executable)


def _sync(tool):
    """Return the synchronous function wrapped by @guard."""
    return tool.__wrapped__


class RunnerTests(unittest.TestCase):
    def test_run_drains_both_streams_with_hard_memory_caps(self) -> None:
        script = (
            "import sys;"
            "sys.stdout.write('A'*200000);sys.stdout.flush();"
            "sys.stderr.write('B'*200000);sys.stderr.flush()"
        )
        stdout, stderr, truncated = run(
            [sys.executable, "-c", script],
            max_bytes=1024,
            max_stderr_bytes=2048,
            timeout=20,
        )
        self.assertTrue(truncated)
        self.assertLessEqual(len(stdout.encode()), 1024)
        self.assertIn("stderr truncated", stderr)
        self.assertLessEqual(len(stderr.encode()), 2200)

    def test_run_raises_command_error_with_stderr_tail(self) -> None:
        with self.assertRaises(CommandError) as caught:
            run(
                [
                    sys.executable,
                    "-c",
                    "import sys;sys.stderr.write('specific failure');sys.exit(7)",
                ],
                timeout=20,
            )
        self.assertEqual(caught.exception.returncode, 7)
        self.assertIn("specific failure", caught.exception.stderr)

    @unittest.skipUnless(os.name == "nt", "Windows taskkill process-tree regression")
    def test_timeout_kills_spawned_child_process_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "child-survived.txt"
            child = (
                "import time;from pathlib import Path;"
                "time.sleep(1.0);"
                f"Path({str(marker)!r}).write_text('survived')"
            )
            parent = (
                "import subprocess,sys,time;"
                f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
                "time.sleep(30)"
            )
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                run([sys.executable, "-c", parent], timeout=0.2)
            time.sleep(1.2)
            self.assertFalse(marker.exists(), "spawned child survived timeout")


class GuardAndSchemaTests(unittest.TestCase):
    def test_guard_is_nonblocking_and_raises_protocol_tool_error(self) -> None:
        @server.guard
        def slow() -> dict[str, object]:
            time.sleep(0.05)
            return {"ok": True}

        async def exercise() -> dict[str, object]:
            task = asyncio.create_task(slow())
            await asyncio.sleep(0.005)
            self.assertFalse(task.done(), "blocking function ran on the event loop")
            return await task

        self.assertEqual(asyncio.run(exercise()), {"ok": True})

        @server.guard
        def fail() -> dict[str, object]:
            raise ValueError("bad input")

        with self.assertRaises(ToolError):
            asyncio.run(fail())

    def test_registered_tools_have_structured_schema_docs_and_annotations(self) -> None:
        tools = server.mcp._tool_manager._tools
        dump = tools["dump_calls"]
        self.assertIn("0-1000/draw", dump.description)
        self.assertEqual(dump.output_schema["type"], "object")
        self.assertTrue(dump.annotations.readOnlyHint)
        self.assertFalse(tools["trace_launch"].annotations.readOnlyHint)
        self.assertIn("gltrim_trace", tools)
        self.assertIn("leak_report", tools)
        self.assertIn("repack_trace", tools)
        for name in ("diff_traces", "diff_state", "diff_images", "sed_trace"):
            self.assertIn(name, tools)
            self.assertEqual(tools[name].output_schema["type"], "object")
        self.assertTrue(tools["diff_traces"].annotations.readOnlyHint)
        self.assertTrue(tools["diff_state"].annotations.readOnlyHint)
        self.assertFalse(tools["diff_images"].annotations.readOnlyHint)
        self.assertFalse(tools["sed_trace"].annotations.readOnlyHint)

    def test_legacy_and_native_argv_are_safe_and_quote_aware(self) -> None:
        self.assertEqual(
            server._coerce_argv(
                r'--name "two words" --path "C:\Program Files\Game"', label="args"
            ),
            ["--name", "two words", "--path", r"C:\Program Files\Game"],
        )
        native = ["--name", "two words", "--literal=$HOME"]
        self.assertEqual(server._coerce_argv(native, label="args"), native)
        with self.assertRaises(ValueError):
            server._coerce_argv(["bad\0value"], label="args")

    def test_import_repairs_fastmcp_forward_ref_without_warning(self) -> None:
        source = Path(__file__).resolve().parents[1] / "src"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(source), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        completed = subprocess.run(
            [sys.executable, "-W", "default", "-c", "import apitrace_mcp.server"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("IncompleteFieldDefinitionWarning", completed.stderr)

    def test_guard_bounds_concurrent_foreground_work(self) -> None:
        lock = threading.Lock()
        running = 0
        peak = 0

        @server.guard
        def work() -> dict[str, bool]:
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.04)
            with lock:
                running -= 1
            return {"ok": True}

        async def exercise() -> None:
            await asyncio.gather(*(work() for _ in range(8)))

        asyncio.run(exercise())
        self.assertEqual(peak, server.MAX_CONCURRENT_FOREGROUND_TOOLS)

    def test_hard_response_and_structured_call_caps(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_bytes"):
            _sync(server.dump_calls)(
                "unused.trace", max_bytes=server.MAX_TOOL_OUTPUT_BYTES + 1
            )
        with self.assertRaisesRegex(ValueError, "limit"):
            _sync(server.get_calls)(
                "unused.trace", limit=server.MAX_STRUCTURED_CALLS + 1
            )
        with self.assertRaisesRegex(ValueError, "max_list"):
            _sync(server.get_calls)(
                "unused.trace", max_list=server.MAX_STRUCTURED_LIST_ITEMS + 1
            )
        with self.assertRaisesRegex(ValueError, "max_bytes"):
            _sync(server.leak_report)(
                "unused.trace", max_bytes=server.MAX_TOOL_OUTPUT_BYTES + 1
            )

    def test_output_allocators_are_unique_under_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            target = directory / "result.trace"
            prefix = directory / "frame"

            with ThreadPoolExecutor(max_workers=16) as pool:
                files = list(pool.map(lambda _: server._unique_file(target), range(32)))
                prefixes = list(
                    pool.map(lambda _: server._unique_prefix(prefix), range(32))
                )

        self.assertEqual(len(set(files)), 32)
        self.assertEqual(len(set(prefixes)), 32)
        self.assertNotIn(target.resolve(), files)
        self.assertNotIn(prefix.resolve(), prefixes)

    def test_wrapper_install_uses_unique_default_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            exe = directory / "game.exe"
            exe.write_bytes(b"game")
            root = SimpleNamespace(bits=64)
            with (
                patch.object(server, "read_pe", return_value=SimpleNamespace(bits=64)),
                patch.object(server.ENV, "root_for_bits", return_value=root),
                patch.object(server, "trace_dir", return_value=directory),
                patch.object(server.wrappers, "install", return_value={}) as install,
            ):
                result = _sync(server.install_wrapper)(str(exe), api="d3d9")

        selected = Path(install.call_args.kwargs["trace_file"])
        self.assertEqual(result["trace_file"], str(selected))
        self.assertNotEqual(selected, directory / "game.trace")
        self.assertEqual(selected.suffix, ".trace")

    def test_wrapper_install_refuses_explicit_existing_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            exe = directory / "game.exe"
            exe.write_bytes(b"game")
            existing = directory / "capture.trace"
            existing.write_bytes(b"old capture")
            with (
                patch.object(server, "read_pe", return_value=SimpleNamespace(bits=64)),
                patch.object(server.ENV, "root_for_bits", return_value=object()),
                patch.object(server, "trace_dir", return_value=directory),
                patch.object(server.wrappers, "install") as install,
            ):
                with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                    _sync(server.install_wrapper)(
                        str(exe), api="d3d9", trace_file=str(existing)
                    )
            install.assert_not_called()


class FrameAndInfoTests(unittest.TestCase):
    FRAMES = [
        {"FirstCallId": 0, "LastCallId": 9, "TotalCalls": 10, "SizeInBytes": 100},
        {"FirstCallId": 10, "LastCallId": 24, "TotalCalls": 15, "SizeInBytes": 200},
        {"FirstCallId": 25, "LastCallId": 29, "TotalCalls": 5, "SizeInBytes": 50},
    ]

    def test_trace_info_recognizes_uppercase_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "sample.trace"
            trace.write_bytes(b"trace")
            info = {"API": "OpenGL", "FramesCount": 3, "Frames": self.FRAMES}
            with (
                patch.object(server, "_resolve_trace", return_value=trace),
                patch.object(server.ENV, "any_root", return_value=object()),
                patch.object(server.analysis, "trace_info", return_value=info),
            ):
                result = _sync(server.trace_info)(str(trace), per_frame=True)
        self.assertEqual(result["frame_count"], 3)
        self.assertEqual(result["frames_head"], self.FRAMES)
        self.assertNotIn("Frames", result["info"])

    def test_list_frames_uses_info_ranges_and_paginates(self) -> None:
        with (
            patch.object(server, "_resolve_trace", return_value=Path("trace.trace")),
            patch.object(server.ENV, "any_root", return_value=object()),
            patch.object(
                server.analysis, "trace_info", return_value={"Frames": self.FRAMES}
            ),
        ):
            result = _sync(server.list_frames)("trace.trace", start=1, count=1)
            empty = _sync(server.list_frames)("trace.trace", start=0, count=0)
        self.assertEqual(result["frames"][0]["calls"], "10-24")
        self.assertEqual(result["frames"][0]["call_count"], 15)
        self.assertTrue(result["has_more"])
        self.assertEqual(result["next_start"], 2)
        self.assertEqual(empty["frames"], [])
        self.assertEqual(empty["total_frames"], 3)
        self.assertIsNone(empty["next_start"])


class ApiDetectionTests(unittest.TestCase):
    def test_api_of_never_silently_defaults(self) -> None:
        with (
            patch.object(server.ENV, "any_root", return_value=object()),
            patch.object(server.analysis, "trace_info", return_value={"API": "Mystery"}),
        ):
            with self.assertRaises(RuntimeError):
                server._api_of(Path("mystery.trace"))

    def test_directx_label_selects_the_d3d_retracer(self) -> None:
        # apitrace labels every Direct3D flavour (DirectDraw, D3D8, D3D9, DXGI)
        # simply "DirectX". Not mapping that broke dump_state / dump_images /
        # replay_trace for every D3D trace -- this server's primary target.
        for label in ("DirectX", "directx", "D3D9", "Direct3D 9", "DXGI"):
            with (
                patch.object(server.ENV, "any_root", return_value=object()),
                patch.object(
                    server.analysis, "trace_info", return_value={"API": label}
                ),
            ):
                self.assertEqual(
                    server._api_of(Path("game.trace")), "d3d", f"label {label!r}"
                )

    def test_opengl_label_still_selects_the_gl_retracer(self) -> None:
        with (
            patch.object(server.ENV, "any_root", return_value=object()),
            patch.object(
                server.analysis,
                "trace_info",
                return_value={"API": "OpenGL + GLX/WGL/CGL"},
            ),
        ):
            self.assertEqual(server._api_of(Path("game.trace")), "gl")

    def test_d3d12_target_does_not_recommend_dxgi_wrapper(self) -> None:
        pe = SimpleNamespace(
            bits=64,
            machine="amd64",
            all_imports=["dxgi.dll", "d3d12.dll"],
        )
        apis = [
            {
                "dll": "dxgi.dll",
                "api": "dxgi",
                "wrapper_dll": "dxgi.dll",
                "supported": True,
            },
            {
                "dll": "d3d12.dll",
                "api": None,
                "wrapper_dll": None,
                "supported": False,
            },
        ]
        with (
            patch.object(server, "_resolved_file", return_value=Path(sys.executable)),
            patch.object(server, "read_pe", return_value=pe),
            patch.object(server, "graphics_apis", return_value=apis),
            patch.object(server.ENV, "_roots", {64: _Root()}),
        ):
            result = _sync(server.detect_target)(sys.executable)
        self.assertNotIn("recommended_api", result)
        self.assertNotIn("recommended_wrapper", result)
        self.assertIn("D3D12 is not supported", result["recommendation"])


class ReplayOutputTests(unittest.TestCase):
    def test_dump_state_parses_complete_json_then_summarizes_payloads(self) -> None:
        state = {
            "framebuffer": {
                "image": {
                    "data": "x" * (server.MAX_STATE_STRING_CHARS + 1),
                    "pixels": list(range(server.MAX_STATE_SEQUENCE_ITEMS + 1)),
                }
            }
        }
        with (
            patch.object(server, "_resolve_trace", return_value=Path("sample.trace")),
            patch.object(server.ENV, "any_root", return_value=_Root()),
            patch.object(server, "run", return_value=(json.dumps(state), "", False)),
        ):
            result = _sync(server.dump_state)("sample.trace", 12, api="gl")
        image = result["state"]["framebuffer"]["image"]
        self.assertEqual(image["data"]["chars"], server.MAX_STATE_STRING_CHARS + 1)
        self.assertEqual(
            image["pixels"]["item_count"], server.MAX_STATE_SEQUENCE_ITEMS + 1
        )

    def test_dump_state_accepts_apitrace_literal_control_characters(self) -> None:
        raw = '{"shaders":{"compiler_log":"line one\nline two\tend"}}'
        with (
            patch.object(server, "_resolve_trace", return_value=Path("sample.trace")),
            patch.object(server.ENV, "any_root", return_value=_Root()),
            patch.object(server, "run", return_value=(raw, "", False)),
        ):
            result = _sync(server.dump_state)("sample.trace", 12, api="gl")
        self.assertEqual(
            result["state"]["shaders"]["compiler_log"], "line one\nline two\tend"
        )

    def test_dump_state_reports_truncation_truthfully(self) -> None:
        with (
            patch.object(server, "_resolve_trace", return_value=Path("sample.trace")),
            patch.object(server.ENV, "any_root", return_value=_Root()),
            patch.object(server, "run", return_value=("{}", "", True)),
        ):
            with self.assertRaisesRegex(RuntimeError, "exceeded"):
                _sync(server.dump_state)("sample.trace", 12, api="gl")

    def test_dump_images_uses_unique_prefix_and_mrt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            trace = directory / "sample.trace"
            trace.write_bytes(b"trace")
            requested = directory / "capture"
            (directory / "capture-old.png").write_bytes(b"old")
            seen: list[str] = []

            def fake_run(cmd, **kwargs):
                seen.extend(cmd)
                prefix = Path(next(item.split("=", 1)[1] for item in cmd if item.startswith("--output=")))
                for index in range(205):
                    prefix.with_name(prefix.name + f"-{index:06}.png").write_bytes(b"png")
                return "ok", "", False

            with (
                patch.object(server, "_resolve_trace", return_value=trace),
                patch.object(server.ENV, "any_root", return_value=_Root()),
                patch.object(server, "run", side_effect=fake_run),
            ):
                result = _sync(server.dump_images)(
                    str(trace), out_prefix=str(requested), mrt=True
                )
        self.assertIn("--mrt", seen)
        self.assertNotEqual(result["output_prefix"], str(requested))
        self.assertEqual(result["image_count"], 205)
        self.assertEqual(len(result["images"]), 200)
        self.assertTrue(result["images_truncated"])

    def test_leak_report_uses_the_stream_apitrace_writes(self) -> None:
        with (
            patch.object(server, "_resolve_trace", return_value=Path("sample.trace")),
            patch.object(server.ENV, "any_root", return_value=_Root()),
            patch.object(server, "run", return_value=("", "object leaked", False)),
        ):
            result = _sync(server.leak_report)("sample.trace")
        self.assertEqual(result["stream"], "stderr")
        self.assertEqual(result["report"], "object leaked")


class CompareAndEditTests(unittest.TestCase):
    def test_diff_traces_builds_bounded_semantic_diff_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reference = directory / "reference.trace"
            source = directory / "source.trace"
            reference.write_bytes(b"reference")
            source.write_bytes(b"source")
            seen: list[str] = []

            def fake_run(cmd, **kwargs):
                seen.extend(cmd)
                self.assertEqual(kwargs["max_bytes"], 12345)
                return "semantic difference", "", False

            with (
                patch.object(server.ENV, "any_root", return_value=_Root()),
                patch.object(server, "run", side_effect=fake_run),
            ):
                result = _sync(server.diff_traces)(
                    str(reference),
                    str(source),
                    calls="4-20",
                    reference_calls="4-10",
                    source_calls="8-20",
                    call_numbers=True,
                    suppress_common_lines=True,
                    width=120,
                    max_bytes=12345,
                )

        self.assertIn("diff", seen)
        self.assertIn(f"--apitrace={_Root.apitrace_exe}", seen)
        self.assertIn("--tool=python", seen)
        self.assertIn("--calls=4-20", seen)
        self.assertIn("--ref-calls=4-10", seen)
        self.assertIn("--src-calls=8-20", seen)
        self.assertIn("--call-nos", seen)
        self.assertIn("--suppress-common-lines", seen)
        self.assertIn("--width=120", seen)
        self.assertEqual(result["diff"], "semantic difference")

    def test_diff_state_canonicalizes_literal_controls_for_jsondiff(self) -> None:
        trace = Path("sample.trace")
        raw_reference = '{"log":"a\x01b"}'
        raw_source = '{"log":"a\x02b"}'
        captures = [
            (raw_reference, {"log": "a\x01b"}, "gl"),
            (raw_source, {"log": "a\x02b"}, "gl"),
        ]
        seen: list[str] = []

        def fake_run(cmd, **kwargs):
            seen.extend(cmd)
            ref_text = Path(cmd[-2]).read_text(encoding="utf-8")
            src_text = Path(cmd[-1]).read_text(encoding="utf-8")
            self.assertIn(r"\u0001", ref_text)
            self.assertIn(r"\u0002", src_text)
            self.assertEqual(json.loads(ref_text), {"log": "a\x01b"})
            self.assertEqual(json.loads(src_text), {"log": "a\x02b"})
            return "log differs", "", False

        with (
            patch.object(server, "_resolve_trace", return_value=trace),
            patch.object(server, "_capture_state_json", side_effect=captures),
            patch.object(server.ENV, "any_root", return_value=_Root()),
            patch.object(server, "run", side_effect=fake_run),
        ):
            result = _sync(server.diff_state)(
                str(trace), 10, 20, ignore_added=True, keep_images=True
            )

        self.assertIn("diff-state", seen)
        self.assertIn("--ignore-added", seen)
        self.assertIn("--keep-images", seen)
        self.assertTrue(result["differences_found"])
        self.assertEqual(
            result["reference"]["state_json_bytes"], len(raw_reference.encode())
        )

    def test_diff_images_reports_missing_pillow_cleanly(self) -> None:
        with patch.object(server.importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "requires Pillow"):
                _sync(server.diff_images)("reference", "source")

    def test_diff_images_accepts_exit_one_with_complete_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reference = directory / "ref" / "shot"
            source = directory / "src" / "shot"
            reference.parent.mkdir()
            source.parent.mkdir()
            reference.with_name("shot000.png").write_bytes(b"png")
            source.with_name("shot000.png").write_bytes(b"png")
            package = directory / "apitrace"
            snapdiff = package / "lib" / "scripts" / "snapdiff.py"
            snapdiff.parent.mkdir(parents=True)
            snapdiff.write_text("# upstream helper", encoding="utf-8")
            apitrace_exe = package / "bin" / "apitrace.exe"
            apitrace_exe.parent.mkdir()
            apitrace_exe.write_bytes(b"exe")
            root = SimpleNamespace(apitrace_exe=apitrace_exe)
            seen: list[str] = []

            def fake_run(cmd, **kwargs):
                seen.extend(cmd)
                locks = list(source.parent.glob(".apitrace-mcp-diff-images-*.lock"))
                self.assertEqual(len(locks), 1)
                report = Path(
                    next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("--output="))
                )
                report.write_text("<html><body>different</body></html>", encoding="utf-8")
                source.with_name("shot000.diff.png").write_bytes(b"diff")
                raise CommandError(cmd, 1, "images differ")

            with (
                patch.object(server.importlib.util, "find_spec", return_value=object()),
                patch.object(server.ENV, "any_root", return_value=root),
                patch.object(server, "run", side_effect=fake_run),
            ):
                result = _sync(server.diff_images)(
                    str(reference),
                    str(source),
                    output=str(directory / "report.html"),
                )
            remaining_locks = list(
                source.parent.glob(".apitrace-mcp-diff-images-*.lock")
            )

        self.assertEqual(result["result"], "different_or_missing")
        self.assertEqual(result["sidecar_count"], 1)
        self.assertEqual(seen[:2], [sys.executable, str(snapdiff.resolve())])
        self.assertEqual(result["command"][:2], seen[:2])
        self.assertNotIn("--overwrite", seen)
        self.assertFalse(remaining_locks)

    def test_diff_images_lock_is_cross_process_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reference = directory / "reference" / "shot"
            source = directory / "source" / "shot"
            reference.parent.mkdir()
            source.parent.mkdir()
            lock, token = server._acquire_diff_images_lock(reference, source)
            try:
                with self.assertRaisesRegex(RuntimeError, "another apitrace-mcp"):
                    server._acquire_diff_images_lock(reference, source)
            finally:
                server._release_diff_images_lock(lock, token)
            self.assertFalse(lock.exists())

    def test_diff_images_refuses_existing_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reference = directory / "ref" / "shot"
            source = directory / "src" / "shot"
            reference.parent.mkdir()
            source.parent.mkdir()
            reference.with_name("shot000.png").write_bytes(b"png")
            source.with_name("shot000.png").write_bytes(b"png")
            source.with_name("shot000.diff.png").write_bytes(b"existing")
            with patch.object(
                server.importlib.util, "find_spec", return_value=object()
            ):
                with self.assertRaisesRegex(FileExistsError, "overwrite"):
                    _sync(server.diff_images)(str(reference), str(source))

    def test_sed_trace_uses_safe_args_and_never_overwrites_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            trace = directory / "input.trace"
            trace.write_bytes(b"trace")
            requested = directory / "edited.trace"
            requested.write_bytes(b"keep")
            seen: list[str] = []

            def fake_run(cmd, **kwargs):
                seen.extend(cmd)
                output = Path(
                    next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("--output="))
                )
                output.write_bytes(b"edited")
                return "", "", False

            with (
                patch.object(server.ENV, "any_root", return_value=_Root()),
                patch.object(server, "run", side_effect=fake_run),
            ):
                result = _sync(server.sed_trace)(
                    str(trace),
                    expressions=["s/OLD/NEW/"],
                    properties=["author=codex"],
                    calls="1-9",
                    output=str(requested),
                    extra_args=["--verbose"],
                )
            requested_contents = requested.read_bytes()

        self.assertEqual(requested_contents, b"keep")
        self.assertNotEqual(result["output"], str(requested))
        self.assertIn("-e", seen)
        self.assertIn("s/OLD/NEW/", seen)
        self.assertIn("--property=author=codex", seen)
        self.assertIn("--calls=1-9", seen)
        self.assertIn("--verbose", seen)


class VerifiedDefectRegressionTests(unittest.TestCase):
    DDRAW = {
        "dll": "ddraw.dll",
        "api": "d3d7",
        "wrapper_dll": "ddraw.dll",
        "supported": True,
    }
    D3D9 = {
        "dll": "d3d9.dll",
        "api": "d3d9",
        "wrapper_dll": "d3d9.dll",
        "supported": True,
    }

    def test_diff_images_failure_releases_sidecar_reservations_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reference = directory / "ref" / "shot"
            source = directory / "src" / "shot"
            reference.parent.mkdir()
            source.parent.mkdir()
            reference.with_name("shot000.png").write_bytes(b"png")
            source.with_name("shot000.png").write_bytes(b"png")
            package = directory / "apitrace"
            snapdiff = package / "lib" / "scripts" / "snapdiff.py"
            snapdiff.parent.mkdir(parents=True)
            snapdiff.write_text("# upstream helper", encoding="utf-8")
            apitrace_exe = package / "bin" / "apitrace.exe"
            apitrace_exe.parent.mkdir()
            apitrace_exe.write_bytes(b"exe")
            root = SimpleNamespace(apitrace_exe=apitrace_exe)
            attempts = {"count": 0}

            def fake_run(cmd, **kwargs):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("command timed out after 600s")
                report = Path(
                    next(arg.split("=", 1)[1] for arg in cmd if arg.startswith("--output="))
                )
                report.write_text("<html><body>match</body></html>", encoding="utf-8")
                return "", "", False

            with (
                patch.object(server.importlib.util, "find_spec", return_value=object()),
                patch.object(server.ENV, "any_root", return_value=root),
                patch.object(server, "run", side_effect=fake_run),
            ):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    _sync(server.diff_images)(
                        str(reference),
                        str(source),
                        output=str(directory / "first.html"),
                    )
                # Before the fix this retry died with FileExistsError because the
                # first attempt's sidecar reservations were never released.
                result = _sync(server.diff_images)(
                    str(reference),
                    str(source),
                    output=str(directory / "second.html"),
                )

        self.assertEqual(result["result"], "match")
        self.assertEqual(attempts["count"], 2)

    def test_dump_images_finds_pngs_for_bracketed_trace_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            trace = directory / "Game [USA].trace"
            trace.write_bytes(b"trace")
            requested = directory / "Game [USA]"

            def fake_run(cmd, **kwargs):
                prefix = Path(
                    next(item.split("=", 1)[1] for item in cmd if item.startswith("--output="))
                )
                for index in range(3):
                    prefix.with_name(prefix.name + f"{index:010}.png").write_bytes(b"png")
                return "ok", "", False

            with (
                patch.object(server, "_resolve_trace", return_value=trace),
                patch.object(server.ENV, "any_root", return_value=_Root()),
                patch.object(server, "run", side_effect=fake_run),
            ):
                result = _sync(server.dump_images)(
                    str(trace), out_prefix=str(requested)
                )

        self.assertEqual(result["image_count"], 3)
        self.assertEqual(len(result["images"]), 3)

    def test_unique_prefix_collision_check_sees_bracketed_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            prefix = (directory / "Game [USA]").resolve()
            fixed = uuid.UUID(int=1)
            colliding = prefix.with_name(f"{prefix.name}-{fixed.hex[:12]}-000001.png")
            colliding.write_bytes(b"png")
            found = server._names_with_prefix(directory, prefix.name)
            self.assertIn(colliding, [entry.resolve() for entry in found])
            # With every candidate colliding, the allocator must refuse rather
            # than silently reuse the taken prefix (the old glob no-op did).
            with patch.object(server.uuid, "uuid4", return_value=fixed):
                with self.assertRaisesRegex(RuntimeError, "unique output prefix"):
                    server._unique_prefix(prefix)

    def test_status_tools_bypass_work_slots_and_busy_error_names_cap(self) -> None:
        for _ in range(server.MAX_CONCURRENT_FOREGROUND_TOOLS):
            self.assertTrue(server._FOREGROUND_WORK_SLOTS.acquire(timeout=1))
        try:
            manager = SimpleNamespace(
                ensure_loaded=lambda log_dir: None,
                get=lambda session: SimpleNamespace(status=lambda: {"id": session}),
            )
            with (
                patch.object(server, "MANAGER", manager),
                patch.object(
                    server, "trace_dir", return_value=Path(tempfile.gettempdir())
                ),
            ):
                start = time.monotonic()
                result = asyncio.run(server.trace_status("abc"))
                elapsed = time.monotonic() - start
            self.assertEqual(result, {"id": "abc"})
            self.assertLess(elapsed, 5.0, "trace_status blocked on a work slot")

            @server.guard
            def slotted() -> dict[str, bool]:
                return {"ok": True}

            with patch.object(server, "WORK_SLOT_TIMEOUT_SECONDS", 0.05):
                pattern = (
                    f"busy.*{server.MAX_CONCURRENT_FOREGROUND_TOOLS} concurrent"
                )
                with self.assertRaisesRegex(ToolError, pattern):
                    asyncio.run(slotted())
        finally:
            for _ in range(server.MAX_CONCURRENT_FOREGROUND_TOOLS):
                server._FOREGROUND_WORK_SLOTS.release()

    def test_dump_calls_no_longer_accepts_blobs(self) -> None:
        tool = server.mcp._tool_manager._tools["dump_calls"]
        self.assertNotIn("blobs", tool.parameters.get("properties", {}))
        self.assertNotIn(
            "blobs", inspect.signature(_sync(server.dump_calls)).parameters
        )
        self.assertIn("extract_blobs", _sync(server.dump_calls).__doc__)

    def test_api_autodetect_prefers_d3d9_over_earlier_ddraw_import(self) -> None:
        with patch.object(
            server, "graphics_apis", return_value=[dict(self.DDRAW), dict(self.D3D9)]
        ):
            self.assertEqual(server._detect_trace_api(object()), "d3d9")
        with patch.object(server, "graphics_apis", return_value=[dict(self.DDRAW)]):
            self.assertEqual(server._detect_trace_api(object()), "d3d7")

    def test_detect_target_ranks_apis_and_reports_runners_up(self) -> None:
        pe = SimpleNamespace(
            bits=32,
            machine="i386",
            all_imports=["ddraw.dll", "d3d9.dll"],
        )
        with (
            patch.object(server, "_resolved_file", return_value=Path(sys.executable)),
            patch.object(server, "read_pe", return_value=pe),
            patch.object(
                server,
                "graphics_apis",
                return_value=[dict(self.DDRAW), dict(self.D3D9)],
            ),
            patch.object(server.ENV, "_roots", {32: _Root()}),
        ):
            result = _sync(server.detect_target)(sys.executable)
        self.assertEqual(result["recommended_api"], "d3d9")
        self.assertEqual(result["runner_up_apis"], ["d3d7"])
        self.assertIn("ddraw", result["api_ranking_note"])

    def test_extract_blobs_docstring_mentions_walk_cap(self) -> None:
        tool = server.mcp._tool_manager._tools["extract_blobs"]
        self.assertIn("walk", tool.description.lower())
        self.assertIn("20000", tool.description)


if __name__ == "__main__":
    unittest.main()
