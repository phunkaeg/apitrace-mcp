"""Focused tests for stateful matrix extraction and pickle safety."""

from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apitrace_mcp import analysis, matrices as mx, pickleparse  # noqa: E402
from apitrace_mcp.pickleparse import CALL_FLAG_END_FRAME, Call  # noqa: E402
from apitrace_mcp.runner import CommandError  # noqa: E402


def call(no, name, args=(), *, flags=0, thread=0, ret=None):
    return Call(no=no, thread_id=thread, name=name, args=list(args), ret=ret, flags=flags)


def view_matrix(eye):
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -eye[0], -eye[1], -eye[2], 1.0,
    ]


def blob(matrix):
    return struct.pack("<16f", *matrix)


def identity_matrix():
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def scale_matrix(x, y, z):
    return [
        float(x), 0.0, 0.0, 0.0,
        0.0, float(y), 0.0, 0.0,
        0.0, 0.0, float(z), 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def translation_matrix(x, y, z):
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        float(x), float(y), float(z), 1.0,
    ]


def transpose_matrix(matrix):
    return [matrix[column * 4 + row] for row in range(4) for column in range(4)]


def gl_multiply(left, right):
    return [
        sum(left[k * 4 + row] * right[column * 4 + k] for k in range(4))
        for column in range(4)
        for row in range(4)
    ]


def gl_frustum(left, right, bottom, top, near, far):
    return [
        2.0 * near / (right - left), 0.0, 0.0, 0.0,
        0.0, 2.0 * near / (top - bottom), 0.0, 0.0,
        (right + left) / (right - left),
        (top + bottom) / (top - bottom),
        -(far + near) / (far - near), -1.0,
        0.0, 0.0, -(2.0 * far * near) / (far - near), 0.0,
    ]


def gl_ortho(left, right, bottom, top, near, far):
    return [
        2.0 / (right - left), 0.0, 0.0, 0.0,
        0.0, 2.0 / (top - bottom), 0.0, 0.0,
        0.0, 0.0, -2.0 / (far - near), 0.0,
        -(right + left) / (right - left),
        -(top + bottom) / (top - bottom),
        -(far + near) / (far - near), 1.0,
    ]


class AnalysisTests(unittest.TestCase):
    def test_temporal_changes_use_final_value_per_frame(self):
        slot = analysis.MatrixSlot("SetTransform(D3DTS_VIEW)", "SetTransform")
        decode = mx.classify(view_matrix((1, 2, 3)), "d3d")
        slot.add(analysis.MatrixSample(1, 0, view_matrix((1, 2, 3)), decode), 2)
        slot.add(analysis.MatrixSample(2, 0, view_matrix((4, 5, 6)), decode), 2)
        slot.add(analysis.MatrixSample(3, 1, view_matrix((4, 5, 6)), decode), 2)
        stable = slot.to_dict()
        self.assertEqual(stable["temporal_changes"], 0)
        self.assertTrue(stable["explicit_view"])
        slot.add(analysis.MatrixSample(4, 2, view_matrix((7, 8, 9)), decode), 2)
        self.assertEqual(slot.to_dict()["temporal_changes"], 1)

    def test_gl_uniform_source_includes_program(self):
        extractor = analysis._MatrixEventExtractor()
        extractor.feed(call(1, "glUseProgram", [("program", 42)]))
        events = extractor.feed(
            call(
                2,
                "glUniformMatrix4fv",
                [("location", 7), ("count", 1), ("transpose", False), ("value", view_matrix((1.0, 2.0, 3.0)))],
            )
        )
        self.assertEqual(events[0].source, "glUniformMatrix4fv(program=42,location=7)")

    def test_gl_fixed_function_multiplies_current_column_major_matrix(self):
        extractor = analysis._MatrixEventExtractor()
        scale = scale_matrix(2, 3, 4)
        translation = translation_matrix(5, 6, 7)
        extractor.feed(call(1, "glMatrixMode", [("mode", "GL_MODELVIEW")]))
        extractor.feed(call(2, "glLoadMatrixf", [("m", scale)]))
        event = extractor.feed(call(3, "glMultMatrixf", [("m", translation)]))[0]
        self.assertEqual(event.matrix, gl_multiply(scale, translation))
        self.assertEqual(event.matrix[12:15], [10.0, 18.0, 28.0])
        self.assertEqual(
            event.source, "glMultMatrixf[GL_MODELVIEW](thread=0)"
        )

    def test_gl_transpose_load_and_mult_apply_transposed_input(self):
        extractor = analysis._MatrixEventExtractor()
        scale = scale_matrix(2, 3, 4)
        translation = translation_matrix(5, 6, 7)
        supplied = transpose_matrix(translation)

        loaded = extractor.feed(
            call(1, "glLoadTransposeMatrixf", [("m", supplied)])
        )[0]
        self.assertEqual(loaded.matrix, translation)

        extractor.feed(call(2, "glLoadMatrixf", [("m", scale)]))
        multiplied = extractor.feed(
            call(3, "glMultTransposeMatrixf", [("m", supplied)])
        )[0]
        self.assertEqual(multiplied.matrix, gl_multiply(scale, translation))

    def test_gl_modes_stacks_and_threads_are_isolated(self):
        extractor = analysis._MatrixEventExtractor()
        modelview = translation_matrix(1, 2, 3)
        projection = scale_matrix(2, 3, 4)
        delta = translation_matrix(5, 6, 7)

        extractor.feed(call(1, "glLoadMatrixf", [("m", modelview)], thread=1))
        extractor.feed(
            call(2, "glMatrixMode", [("mode", "GL_PROJECTION")], thread=1)
        )
        extractor.feed(call(3, "glLoadMatrixf", [("m", projection)], thread=1))
        extractor.feed(call(4, "glPushMatrix", thread=1))
        composed = extractor.feed(
            call(5, "glMultMatrixf", [("m", delta)], thread=1)
        )[0]
        self.assertEqual(composed.matrix, gl_multiply(projection, delta))
        restored = extractor.feed(call(6, "glPopMatrix", thread=1))[0]
        self.assertEqual(restored.matrix, projection)

        extractor.feed(
            call(7, "glMatrixMode", [("mode", "GL_MODELVIEW")], thread=1)
        )
        thread_one = extractor.feed(
            call(8, "glMultMatrixf", [("m", identity_matrix())], thread=1)
        )[0]
        self.assertEqual(thread_one.matrix, modelview)

        thread_two = extractor.feed(
            call(9, "glMultMatrixf", [("m", delta)], thread=2)
        )[0]
        self.assertEqual(thread_two.matrix, delta)
        self.assertIn("thread=1", thread_one.source)
        self.assertIn("thread=2", thread_two.source)

    def test_gl_context_state_follows_make_current_across_threads(self):
        extractor = analysis._MatrixEventExtractor()
        scale = scale_matrix(2, 3, 4)
        translation = translation_matrix(5, 6, 7)
        extractor.feed(
            call(
                1,
                "wglMakeCurrent",
                [("hdc", 10), ("hglrc", 77)],
                thread=1,
                ret=True,
            )
        )
        extractor.feed(call(2, "glLoadMatrixf", [("m", scale)], thread=1))
        extractor.feed(
            call(
                3,
                "wglMakeCurrent",
                [("hdc", 11), ("hglrc", 77)],
                thread=2,
                ret=True,
            )
        )
        event = extractor.feed(
            call(4, "glMultMatrixf", [("m", translation)], thread=2)
        )[0]
        self.assertEqual(event.matrix, gl_multiply(scale, translation))
        self.assertIn("context=wgl:77", event.source)

    def test_gl_frustum_and_ortho_multiply_projection_stack(self):
        extractor = analysis._MatrixEventExtractor()
        current = scale_matrix(2, 3, 4)
        extractor.feed(call(1, "glMatrixMode", [("mode", "GL_PROJECTION")]))
        extractor.feed(call(2, "glLoadMatrixf", [("m", current)]))

        frustum_args = [-1.0, 2.0, -0.5, 1.0, 1.0, 100.0]
        frustum = extractor.feed(
            call(3, "glFrustum", [("planes", frustum_args)])
        )[0]
        self.assertEqual(
            frustum.matrix, gl_multiply(current, gl_frustum(*frustum_args))
        )
        self.assertEqual(frustum.parameters["right"], 2.0)

        extractor.feed(call(4, "glLoadMatrixf", [("m", current)]))
        ortho_args = [-4.0, 6.0, -3.0, 5.0, -2.0, 20.0]
        ortho = extractor.feed(call(5, "glOrtho", [("planes", ortho_args)]))[0]
        self.assertEqual(ortho.matrix, gl_multiply(current, gl_ortho(*ortho_args)))
        self.assertEqual(len(ortho.matrix), 16)

    def test_buffer_sources_include_bound_resource_and_d3d_destination(self):
        extractor = analysis._MatrixEventExtractor()
        extractor.feed(call(1, "glBindBuffer", [("target", "GL_UNIFORM_BUFFER"), ("buffer", 17)]))
        gl_events = extractor.feed(
            call(
                2,
                "glBufferData",
                [("target", "GL_UNIFORM_BUFFER"), ("size", 64), ("data", blob(view_matrix((1, 2, 3))))],
            )
        )
        self.assertIn("buffer=17", gl_events[0].source)

        first = extractor.feed(
            call(
                3,
                "ID3D11DeviceContext::UpdateSubresource",
                [("pDstResource", "resource-A"), ("DstSubresource", 0), ("pSrcData", blob(view_matrix((1, 2, 3))))],
            )
        )
        second = extractor.feed(
            call(
                4,
                "ID3D11DeviceContext::UpdateSubresource",
                [("pDstResource", "resource-B"), ("DstSubresource", 0), ("pSrcData", blob(view_matrix((1, 2, 3))))],
            )
        )
        self.assertNotEqual(first[0].source, second[0].source)

    def test_track_camera_handles_buffer_source(self):
        calls = [
            call(0, "glBindBuffer", [("target", "GL_UNIFORM_BUFFER"), ("buffer", 7)]),
            call(1, "glBufferData", [("target", "GL_UNIFORM_BUFFER"), ("data", blob(view_matrix((1, 2, 3))))]),
            call(2, "wglSwapBuffers", flags=CALL_FLAG_END_FRAME),
            call(3, "glBufferData", [("target", "GL_UNIFORM_BUFFER"), ("data", blob(view_matrix((4, 5, 6))))]),
            call(4, "wglSwapBuffers", flags=CALL_FLAG_END_FRAME),
        ]
        with patch.object(analysis, "iter_calls", side_effect=lambda *args, **kwargs: iter(calls)):
            result = analysis.track_camera(None, "dummy.trace", max_frames=10)
        self.assertEqual(result["frame_count"], 2)
        self.assertEqual(result["frames"][0]["eye"], [1.0, 2.0, 3.0])
        self.assertEqual(result["frames"][1]["eye"], [4.0, 5.0, 6.0])
        self.assertEqual(result["frames"][0]["view_z_axis"], [0.0, 0.0, 1.0])
        self.assertEqual(
            result["frames"][0]["forward_candidates"]["right_handed"],
            [-0.0, -0.0, -1.0],
        )
        self.assertNotIn("forward", result["frames"][0])
        self.assertTrue(result["camera_moves"])

    def test_frustum_filters_mode_and_invalid_planes_and_records_real_matrix(self):
        self.assertIsNone(analysis._frustum_event("glFrustum", [-1, 1, -1, 1, 1, 10], "GL_MODELVIEW"))
        self.assertIsNone(analysis._frustum_event("glFrustum", [1, 1, -1, 1, 1, 10], "GL_PROJECTION"))
        self.assertIsNone(analysis._frustum_event("glFrustum", [-1, 1, 1, 1, 1, 10], "GL_PROJECTION"))
        slots = {}
        analysis.record_frustum(slots, "glFrustum", [-1, 1, -0.5, 0.5, 1, 10], 3, 0, "GL_PROJECTION")
        sample = next(iter(slots.values())).to_dict()["samples"][0]
        self.assertIn("parameters", sample)
        self.assertEqual(len(sample["rows"]), 4)
        self.assertEqual(
            analysis._frustum_event(
                "glFrustum", [-1, 1, -0.5, 0.5, 1, 10], "GL_PROJECTION"
            ).matrix,
            gl_frustum(-1, 1, -0.5, 0.5, 1, 10),
        )

    def test_d3d8_byte_blob_constants_fill_register_files(self):
        extractor = analysis._MatrixEventExtractor()
        vs_events = extractor.feed(
            call(
                1,
                "IDirect3DDevice8::SetVertexShaderConstant",
                [
                    ("Register", 0),
                    ("pConstantData", blob(view_matrix((1, 2, 3)))),
                    ("ConstantCount", 4),
                ],
            )
        )
        vs_sources = [event.source for event in vs_events]
        self.assertIn("vs_c[0..3]", vs_sources)
        vs_event = vs_events[vs_sources.index("vs_c[0..3]")]
        self.assertEqual(vs_event.matrix, view_matrix((1.0, 2.0, 3.0)))

        ps_events = extractor.feed(
            call(
                2,
                "IDirect3DDevice8::SetPixelShaderConstant",
                [
                    ("Register", 4),
                    ("pConstantData", blob(view_matrix((4, 5, 6)))),
                    ("ConstantCount", 4),
                ],
            )
        )
        ps_sources = [event.source for event in ps_events]
        self.assertIn("ps_c[4..7]", ps_sources)
        ps_event = ps_events[ps_sources.index("ps_c[4..7]")]
        self.assertEqual(ps_event.matrix, view_matrix((4.0, 5.0, 6.0)))

    def test_arb_suffixed_uniform_and_use_program_calls_are_recognized(self):
        extractor = analysis._MatrixEventExtractor()
        extractor.feed(call(1, "glUseProgramObjectARB", [("programObj", 42)]))
        events = extractor.feed(
            call(
                2,
                "glUniformMatrix4fvARB",
                [
                    ("location", 7),
                    ("count", 1),
                    ("transpose", False),
                    ("value", view_matrix((1.0, 2.0, 3.0))),
                ],
            )
        )
        self.assertEqual(
            events[0].source, "glUniformMatrix4fvARB(program=42,location=7)"
        )
        self.assertEqual(events[0].matrix, view_matrix((1.0, 2.0, 3.0)))

    def test_gl_ortho_top_left_extents_multiply_projection_stack(self):
        extractor = analysis._MatrixEventExtractor()
        extractor.feed(call(1, "glMatrixMode", [("mode", "GL_PROJECTION")]))
        extractor.feed(call(2, "glLoadIdentity"))
        ortho_args = [0.0, 640.0, 480.0, 0.0, -1.0, 1.0]
        events = extractor.feed(call(3, "glOrtho", [("planes", ortho_args)]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].matrix, gl_ortho(*ortho_args))
        self.assertEqual(events[0].parameters["top"], 0.0)

        # The simulated stack must carry the ortho multiply forward.
        delta = translation_matrix(5, 6, 7)
        multiplied = extractor.feed(call(4, "glMultMatrixf", [("m", delta)]))[0]
        self.assertEqual(
            multiplied.matrix, gl_multiply(gl_ortho(*ortho_args), delta)
        )

    def test_gl_buffer_bindings_are_per_context(self):
        extractor = analysis._MatrixEventExtractor()
        extractor.feed(
            call(1, "glBindBuffer", [("target", "GL_UNIFORM_BUFFER"), ("buffer", 17)], thread=1)
        )
        extractor.feed(
            call(2, "glBindBuffer", [("target", "GL_UNIFORM_BUFFER"), ("buffer", 99)], thread=2)
        )
        events = extractor.feed(
            call(
                3,
                "glBufferSubData",
                [
                    ("target", "GL_UNIFORM_BUFFER"),
                    ("offset", 0),
                    ("size", 64),
                    ("data", blob(view_matrix((1, 2, 3)))),
                ],
                thread=1,
            )
        )
        self.assertIn("buffer=17", events[0].source)
        self.assertNotIn("buffer=99", events[0].source)

    def test_list_shaders_bounds_multi_string_preview_and_zero_limit(self):
        fragments = ["a" * 599, "b" * 1000]
        shader_call = call(1, "glShaderSource", [("string", fragments)])
        with patch.object(analysis, "iter_calls", return_value=iter([shader_call])):
            result = analysis.list_shaders(None, "dummy.trace", max_shaders=1)
        shader = result["shaders"][0]
        self.assertEqual(shader["source_preview"], "a" * 599 + "\n")
        self.assertEqual(shader["source_chars"], 1600)

        with patch.object(
            analysis, "iter_calls", side_effect=AssertionError("walked trace")
        ):
            self.assertEqual(
                analysis.list_shaders(None, "dummy.trace", max_shaders=0),
                {"shaders": [], "count": 0, "truncated": False},
            )
            self.assertEqual(
                analysis.list_shaders(None, "dummy.trace", max_shaders=-1),
                {"shaders": [], "count": 0, "truncated": False},
            )

    def test_extract_blobs_sanitizes_names_and_does_not_overwrite(self):
        malicious = call(1, "../bad\\name", [("../../payload", b"abc")])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(analysis, "iter_calls", return_value=iter([malicious])):
                first = analysis.extract_blobs(None, "dummy", calls="0", out_dir=directory, limit=1)
            with patch.object(analysis, "iter_calls", return_value=iter([malicious])):
                second = analysis.extract_blobs(None, "dummy", calls="0", out_dir=directory, limit=1)
            first_path = Path(first["blobs"][0]["path"])
            second_path = Path(second["blobs"][0]["path"])
            self.assertEqual(first_path.parent, Path(directory).resolve())
            self.assertEqual(second_path.parent, Path(directory).resolve())
            self.assertNotEqual(first_path, second_path)
            self.assertNotIn("..", first_path.name)
            self.assertEqual(first["calls_scanned"], 1)

    def test_extract_blobs_reports_walk_cap_truncation(self):
        def fake_iter(root, trace, *, calls=None, limit=None):
            for no in range(limit):
                yield call(no, "glClear")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(analysis, "iter_calls", side_effect=fake_iter):
                capped = analysis.extract_blobs(
                    None, "dummy", calls="0", out_dir=directory, limit=1
                )
            self.assertTrue(capped["truncated"])
            self.assertEqual(capped["calls_scanned"], 20000)
            self.assertIn("note", capped)

            short_walk = [call(0, "glClear"), call(1, "glFlush")]
            with patch.object(analysis, "iter_calls", return_value=iter(short_walk)):
                finished = analysis.extract_blobs(
                    None, "dummy", calls="0", out_dir=directory, limit=1
                )
            self.assertFalse(finished["truncated"])
            self.assertEqual(finished["calls_scanned"], 2)
            self.assertNotIn("note", finished)


class PickleParseTests(unittest.TestCase):
    def test_schema_and_thread_zero(self):
        parsed = pickleparse._record_to_call((0, 0, "fn", [("x", 1)], None, 0))
        self.assertEqual(pickleparse.call_to_dict(parsed)["thread"], 0)
        with self.assertRaises(ValueError):
            pickleparse._record_to_call((0, 0, "fn", ["not-a-pair"], None, 0))

    def test_limit_zero_does_not_spawn(self):
        with tempfile.NamedTemporaryFile() as trace:
            with patch.object(pickleparse, "pickle_cmd", side_effect=AssertionError("spawned")):
                self.assertEqual(list(pickleparse.iter_calls(None, trace.name, limit=0)), [])

    def test_nonzero_process_reports_stderr(self):
        command = [sys.executable, "-c", "import sys;sys.stderr.write('pickle failed');sys.exit(7)"]
        with tempfile.NamedTemporaryFile() as trace:
            with patch.object(pickleparse, "pickle_cmd", return_value=command):
                with self.assertRaises(CommandError) as raised:
                    list(pickleparse.iter_calls(None, trace.name, timeout=10))
        self.assertIn("pickle failed", raised.exception.stderr)

    def test_pickle_process_timeout(self):
        command = [sys.executable, "-c", "import time;time.sleep(5)"]
        with tempfile.NamedTemporaryFile() as trace:
            with patch.object(pickleparse, "pickle_cmd", return_value=command):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    list(pickleparse.iter_calls(None, trace.name, timeout=0.1))


if __name__ == "__main__":
    unittest.main()


class RealTraceRegressionTests(unittest.TestCase):
    """Defects found by running against a real UE3 / D3D9 capture (Dishonored).

    The numbers here are taken verbatim from that trace: a float4x3 view matrix
    in vs_c[6..8] and the projection scales in vs_c[0..3].
    """

    VIEW_4X3 = [
        0.57944, 0.00590, 0.81439, 16.37790,
        -0.06425, 0.99670, 0.03849, 2.61454,
        -0.81187, -0.07467, 0.57819, 4.99379,
    ]

    def test_float4x3_upload_yields_a_view_matrix(self) -> None:
        # D3D9 engines upload world/view matrices as three registers to save
        # constant space. Scanning only 4-register windows missed these
        # entirely -- on the real trace this slot did not exist at all.
        bank = analysis._RegisterFile()
        written = bank.write(6, list(self.VIEW_4X3))
        self.assertEqual(written, (6, 8))
        windows = {(base, size): flat for base, size, flat in bank.windows(*written)}
        self.assertIn((6, 3), windows, "3-register window was not enumerated")

        flat = windows[(6, 3)]
        self.assertEqual(len(flat), 16)
        self.assertEqual(flat[12:], [0.0, 0.0, 0.0, 1.0], "implicit last row")

        decode = mx.classify(flat, "d3d")
        self.assertIsNotNone(decode)
        self.assertEqual(decode.kind, "rigid")
        self.assertEqual(decode.layout, "transposed")
        for actual, expected in zip(decode.eye, (-5.2677, -2.3297, -16.326)):
            self.assertLess(abs(actual - expected), 1e-3)

    def test_windows_are_upload_aligned(self) -> None:
        # A 144-register bone-matrix upload used to yield ~140 overlapping
        # candidates, nearly all of them straddling two real matrices.
        bank = analysis._RegisterFile()
        written = bank.write(6, [float(i) for i in range(12 * 4)])
        bases = {(base, size) for base, size, _ in bank.windows(*written)}

        # Aligned starts only, measured from the upload's own StartRegister.
        self.assertIn((6, 4), bases)
        self.assertIn((6, 3), bases)
        self.assertIn((9, 3), bases)
        self.assertIn((10, 4), bases)
        for stray in ((7, 4), (8, 4), (7, 3), (8, 3), (11, 3)):
            self.assertNotIn(stray, bases, f"unaligned window {stray} was emitted")

    def test_repeated_rows_are_rejected(self) -> None:
        # The signature of a window slid across packed matrices.
        repeated = [
            0.57944, 0.00590, 0.81439, 16.37790,
            -0.06425, 0.99670, 0.03849, 2.61454,
            -0.81187, -0.07467, 0.57819, 4.99379,
            0.57944, 0.00590, 0.81439, 16.37790,
        ]
        self.assertIsNone(mx.classify(repeated, "d3d"))

    def test_mixed_slot_reports_kind_breakdown(self) -> None:
        slot = analysis.MatrixSlot(source="vs_c[9..11]", function="SetVertexShaderConstantF")
        slot.kinds.update({"viewproj": 5, "rigid": 3})
        data = slot.to_dict(include_matrix=False)
        self.assertEqual(data["kind"], "viewproj")
        self.assertEqual(data["kind_breakdown"], {"viewproj": 5, "rigid": 3})

        single = analysis.MatrixSlot(source="vs_c[0..3]", function="SetVertexShaderConstantF")
        single.kinds.update({"projection": 4})
        self.assertNotIn("kind_breakdown", single.to_dict(include_matrix=False))

    def test_track_camera_follows_the_write_that_moves(self) -> None:
        # A per-draw world*view*projection slot: several writes per frame, each
        # a different object. On a real SWAT 4 capture the write at a fixed
        # ordinal was a static HUD quad, so sampling one position per frame
        # reported "camera is static" while the player was walking.
        calls = []
        no = 0
        for frame in range(4):
            # Write 0: never moves (the HUD/skybox trap).
            calls.append(
                call(no, "IDirect3DDevice9::SetVertexShaderConstantF",
                     [("StartRegister", 0), ("pConstantData", view_matrix((9.0, 9.0, 9.0))),
                      ("Vector4fCount", 4)]))
            no += 1
            # Write 1: travels a unit per frame.
            calls.append(
                call(no, "IDirect3DDevice9::SetVertexShaderConstantF",
                     [("StartRegister", 0),
                      ("pConstantData", view_matrix((float(frame), 0.0, 0.0))),
                      ("Vector4fCount", 4)]))
            no += 1
            calls.append(call(no, "IDirect3DDevice9::Present", flags=CALL_FLAG_END_FRAME))
            no += 1

        with patch.object(analysis, "iter_calls", side_effect=lambda *a, **k: iter(calls)):
            result = analysis.track_camera(
                None, "dummy.trace", source="vs_c[0..3]", max_frames=10
            )

        self.assertTrue(result["camera_moves"], "motion in a later write was missed")
        self.assertEqual(result["sample_ordinal"], 1, "did not follow the moving write")
        self.assertEqual(result["writes_per_frame"], 2.0)
        self.assertIn("per-draw", result["multi_write_note"])
        self.assertGreater(result["eye_path_length"], 2.0)
        self.assertEqual([f["eye"][0] for f in result["frames"]], [0.0, 1.0, 2.0, 3.0])

    def test_track_camera_single_write_slot_stays_simple(self) -> None:
        # A dedicated camera constant written once per frame must not grow the
        # multi-write reporting.
        calls = []
        no = 0
        for frame in range(3):
            calls.append(
                call(no, "IDirect3DDevice9::SetVertexShaderConstantF",
                     [("StartRegister", 0),
                      ("pConstantData", view_matrix((float(frame), 0.0, 0.0))),
                      ("Vector4fCount", 4)]))
            no += 1
            calls.append(call(no, "IDirect3DDevice9::Present", flags=CALL_FLAG_END_FRAME))
            no += 1
        with patch.object(analysis, "iter_calls", side_effect=lambda *a, **k: iter(calls)):
            result = analysis.track_camera(
                None, "dummy.trace", source="vs_c[0..3]", max_frames=10
            )
        self.assertTrue(result["camera_moves"])
        self.assertNotIn("multi_write_note", result)
        self.assertNotIn("writes_per_frame", result)

    def test_single_jump_is_not_reported_as_camera_motion(self) -> None:
        # Morrowind's D3DTS_VIEW flips once between its UI and world pass and is
        # otherwise identity. Treating "differs from frame 0" as motion reported
        # that as a moving camera and sent the caller chasing a slot that can
        # never yield a trajectory.
        calls = []
        no = 0
        for frame in range(12):
            eye = (0.0, 0.0, 600.0) if frame == 0 else (0.0, 0.0, 0.0)
            calls.append(
                call(no, "IDirect3DDevice8::SetTransform",
                     [("State", "D3DTS_VIEW"), ("pMatrix", view_matrix(eye))]))
            no += 1
            calls.append(call(no, "IDirect3DDevice8::Present", flags=CALL_FLAG_END_FRAME))
            no += 1
        with patch.object(analysis, "iter_calls", side_effect=lambda *a, **k: iter(calls)):
            result = analysis.track_camera(
                None, "dummy.trace", source="SetTransform(D3DTS_VIEW)", max_frames=20
            )
        self.assertEqual(result["moving_transitions"], 1)
        self.assertEqual(result["frame_transitions"], 11)
        self.assertFalse(result["camera_moves"], "a single jump is not travel")
        self.assertIn("pass or mode switch", result["note"])

    def test_fixed_function_multi_write_note_is_not_called_a_worldviewproj(self) -> None:
        calls = []
        no = 0
        for frame in range(4):
            for k in range(3):  # same transform state set three times a frame
                calls.append(
                    call(no, "IDirect3DDevice8::SetTransform",
                         [("State", "D3DTS_VIEW"),
                          ("pMatrix", view_matrix((float(frame + k), 0.0, 0.0)))]))
                no += 1
            calls.append(call(no, "IDirect3DDevice8::Present", flags=CALL_FLAG_END_FRAME))
            no += 1
        with patch.object(analysis, "iter_calls", side_effect=lambda *a, **k: iter(calls)):
            result = analysis.track_camera(
                None, "dummy.trace", source="SetTransform(D3DTS_VIEW)", max_frames=10
            )
        note = result["multi_write_note"]
        self.assertIn("separate render passes", note)
        self.assertNotIn("per-draw world*view*projection", note)
