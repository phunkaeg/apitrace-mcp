"""Ground-truth unit tests for matrix classification and camera decoding.

The suite uses only the standard library so it runs both under ``unittest`` and
pytest, and the module is safe to import during test discovery.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apitrace_mcp import matrices as mx  # noqa: E402

FOV_Y = math.radians(60.0)
ASPECT = 16.0 / 9.0
ZN, ZF = 0.1, 1000.0


def d3d_perspective_lh(fov_y=FOV_Y, aspect=ASPECT, zn=ZN, zf=ZF) -> list[float]:
    ys = 1.0 / math.tan(fov_y / 2.0)
    xs = ys / aspect
    q = zf / (zf - zn)
    return [xs, 0, 0, 0, 0, ys, 0, 0, 0, 0, q, 1, 0, 0, -zn * q, 0]


def d3d_perspective_rh(fov_y=FOV_Y, aspect=ASPECT, zn=ZN, zf=ZF) -> list[float]:
    ys = 1.0 / math.tan(fov_y / 2.0)
    xs = ys / aspect
    q = zf / (zn - zf)
    return [xs, 0, 0, 0, 0, ys, 0, 0, 0, 0, q, -1, 0, 0, zn * q, 0]


def d3d_offcentre_lh(left, right, bottom, top, zn=ZN, zf=ZF) -> list[float]:
    return [
        2 * zn / (right - left), 0, 0, 0,
        0, 2 * zn / (top - bottom), 0, 0,
        (left + right) / (left - right), (top + bottom) / (bottom - top),
        zf / (zf - zn), 1,
        0, 0, -zn * zf / (zf - zn), 0,
    ]


def gl_perspective(fov_y=FOV_Y, aspect=ASPECT, zn=ZN, zf=ZF) -> list[float]:
    f = 1.0 / math.tan(fov_y / 2.0)
    return [
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (zf + zn) / (zn - zf), -1,
        0, 0, 2 * zf * zn / (zn - zf), 0,
    ]


def d3d_lookat_lh(eye, at, up=(0.0, 1.0, 0.0)) -> list[float]:
    def sub(a, b):
        return tuple(a[i] - b[i] for i in range(3))

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def norm(v):
        length = math.sqrt(sum(component * component for component in v))
        return tuple(component / length for component in v)

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    z = norm(sub(at, eye))
    x = norm(cross(up, z))
    y = cross(z, x)
    return [
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
    ]


def mul(a: list[float], b: list[float]) -> list[float]:
    return [
        sum(a[row * 4 + k] * b[k * 4 + col] for k in range(4))
        for row in range(4)
        for col in range(4)
    ]


class MatrixDecodeTests(unittest.TestCase):
    def assertClose(self, actual, expected, tolerance=1e-3):
        self.assertIsNotNone(actual)
        self.assertLessEqual(abs(actual - expected), tolerance * max(1.0, abs(expected)))

    def test_d3d_lh_projection(self):
        decode = mx.classify(d3d_perspective_lh(), "d3d")
        self.assertIsNotNone(decode)
        self.assertEqual(decode.kind, "projection")
        self.assertEqual(decode.layout, "as_stored")
        self.assertEqual(decode.handedness, "left")
        self.assertClose(decode.fov_y_deg, 60.0)
        self.assertClose(decode.aspect, ASPECT)
        self.assertClose(decode.near, ZN)
        self.assertClose(decode.far, ZF)

    def test_d3d_rh_projection(self):
        decode = mx.classify(d3d_perspective_rh(), "d3d")
        self.assertIsNotNone(decode)
        self.assertEqual(decode.handedness, "right")
        self.assertClose(decode.near, ZN)
        self.assertClose(decode.far, ZF)

    def test_gl_projection_and_ambiguous_depth(self):
        decode = mx.classify(gl_perspective(), "gl")
        self.assertIsNotNone(decode)
        self.assertClose(decode.fov_y_deg, 60.0)
        self.assertClose(decode.near, ZN)
        self.assertClose(decode.far, ZF)
        self.assertIsNotNone(decode.alt_near_far)
        ambiguous = mx.classify(gl_perspective())
        self.assertIn("unknown", ambiguous.depth_range)

    def test_offcentre_fov_uses_both_side_angles(self):
        left, right, bottom, top = -0.05, 0.12, -0.07, 0.08
        decode = mx.classify(d3d_offcentre_lh(left, right, bottom, top), "d3d")
        self.assertIsNotNone(decode)
        expected_x = math.degrees(math.atan(right / ZN) - math.atan(left / ZN))
        expected_y = math.degrees(math.atan(top / ZN) - math.atan(bottom / ZN))
        self.assertClose(decode.fov_x_deg, expected_x)
        self.assertClose(decode.fov_y_deg, expected_y)
        symmetric_shortcut = math.degrees(2 * math.atan(1 / abs(decode and d3d_offcentre_lh(left, right, bottom, top)[0])))
        self.assertGreater(abs(decode.fov_x_deg - symmetric_shortcut), 1.0)
        self.assertTrue(any("off-centre" in note for note in decode.notes))

    def test_infinite_far_and_reversed_z(self):
        standard_inf = d3d_perspective_lh()
        standard_inf[10], standard_inf[14] = 1.0, -ZN
        decode = mx.classify(standard_inf, "d3d")
        self.assertClose(decode.near, ZN)
        self.assertTrue(math.isinf(decode.far))
        self.assertFalse(decode.reversed_z)

        reversed_inf = d3d_perspective_lh()
        reversed_inf[10], reversed_inf[14] = 0.0, ZN
        decode = mx.classify(reversed_inf, "d3d")
        self.assertClose(decode.near, ZN)
        self.assertTrue(math.isinf(decode.far))
        self.assertTrue(decode.reversed_z)

        reversed_finite = d3d_perspective_lh()
        reversed_finite[10] = ZN / (ZN - ZF)
        reversed_finite[14] = -ZN * ZF / (ZN - ZF)
        decode = mx.classify(reversed_finite, "d3d")
        self.assertClose(decode.near, ZN)
        self.assertClose(decode.far, ZF)
        self.assertTrue(decode.reversed_z)

    def test_gl_infinite_far(self):
        matrix = gl_perspective()
        matrix[10], matrix[14] = -1.0, -2 * ZN
        decode = mx.classify(matrix, "gl")
        self.assertClose(decode.near, ZN)
        self.assertTrue(math.isinf(decode.far))

    def test_transposed_projection(self):
        decode = mx.classify(mx.transpose16(d3d_perspective_lh()), "d3d")
        self.assertEqual(decode.kind, "projection")
        self.assertEqual(decode.layout, "transposed")
        self.assertClose(decode.fov_y_deg, 60.0)

    def test_transposed_infinite_far_layout_tiebreak(self):
        # Transposed infinite-far LH projection with zn=1: the flat array also
        # decodes as-stored (as a reversed-Z RH frustum with near=0.5, far=1.0)
        # at exactly the same score, so the tie must fall to the physically
        # plausible transposed reading rather than to whichever layout ran first.
        inf_far = d3d_perspective_lh(zn=1.0)
        inf_far[10], inf_far[14] = 1.0, -1.0
        decode = mx.classify(mx.transpose16(inf_far), "d3d")
        self.assertIsNotNone(decode)
        self.assertEqual(decode.kind, "projection")
        self.assertEqual(decode.layout, "transposed")
        self.assertEqual(decode.handedness, "left")
        self.assertClose(decode.near, 1.0)
        self.assertTrue(math.isinf(decode.far))
        self.assertFalse(decode.reversed_z)

    def test_transposed_finite_far_unrounded_scores(self):
        # Finite variant of the same trap: the wrong-order reading scores
        # ~0.9996, which the old round-to-3 comparison flattened into a tie.
        decode = mx.classify(mx.transpose16(d3d_perspective_lh(zn=1.0, zf=10000.0)), "d3d")
        self.assertIsNotNone(decode)
        self.assertEqual(decode.kind, "projection")
        self.assertEqual(decode.layout, "transposed")
        self.assertEqual(decode.handedness, "left")
        self.assertClose(decode.near, 1.0)
        self.assertClose(decode.far, 10000.0)
        self.assertFalse(decode.reversed_z)

    def test_projection_structural_rejections(self):
        translated = d3d_perspective_lh()
        translated[12] = 4.0
        decode = mx.classify(translated, "d3d")
        self.assertTrue(decode is None or decode.kind != "projection")
        huge_shift = d3d_perspective_lh()
        huge_shift[8] = 20.0
        decode = mx.classify(huge_shift, "d3d")
        self.assertTrue(decode is None or decode.kind != "projection")
        invalid_depth = d3d_perspective_lh()
        invalid_depth[10], invalid_depth[14] = 0.5, 1.0
        decode = mx.classify(invalid_depth, "d3d")
        self.assertTrue(decode is None or decode.kind != "projection")

    def test_view_eye_and_view_projection(self):
        eye = (-5.0, 3.0, 12.5)
        view = d3d_lookat_lh(eye, (100.0, -40.0, -7.0))
        decode = mx.classify(view, "d3d")
        self.assertEqual(decode.kind, "rigid")
        for actual, expected in zip(decode.eye, eye):
            self.assertClose(actual, expected)

        viewproj = mul(view, d3d_perspective_lh())
        decode = mx.classify(viewproj, "d3d")
        self.assertEqual(decode.kind, "viewproj")
        for actual, expected in zip(decode.eye, eye):
            self.assertClose(actual, expected, 5e-3)
        transposed = mx.classify(mx.transpose16(viewproj), "d3d")
        self.assertEqual(transposed.kind, "viewproj")

    def test_negative_cases_and_ortho(self):
        self.assertTrue(mx.is_identity([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]))
        noise = [0.37, 1.9, -4.2, 8.0, 0.5, 2.2, 3.1, -0.9, 7.7, 1.1, -2.3, 0.4, 9.9, -1.2, 3.3, 5.5]
        decode = mx.classify(noise, "d3d")
        self.assertTrue(decode is None or decode.kind != "projection")
        self.assertIsNone(mx.classify([float("nan")] * 16))
        self.assertIsNone(mx.classify([0.0] * 16))
        scaled = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 5, 5, 5, 1]
        decode = mx.classify(scaled, "d3d")
        self.assertTrue(decode is None or decode.kind != "rigid")
        ortho = [2 / 1280, 0, 0, 0, 0, 2 / 720, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        self.assertEqual(mx.classify(ortho, "d3d").kind, "ortho")


if __name__ == "__main__":
    unittest.main()
