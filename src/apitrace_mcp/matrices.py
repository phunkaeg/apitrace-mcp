"""Recognising and decoding 4x4 matrices found in a trace.

This is the part a VR modder actually needs. Finding the projection matrix tells
you the FOV the game thinks it has; finding the view matrix tells you where the
camera is and which way it points. Both are the prerequisites for stereo
rendering, and in a D3D8/D3D9 game they are sitting in plain sight in
SetTransform or in a block of vertex shader constants.

Conventions handled:

* D3D fixed function -- row-vector, row-major storage, m[row*4+col].
* OpenGL -- column-vector, column-major storage, which is bit-identical to the
  transpose of the D3D layout, so the same code path covers it.
* Shader constants -- D3D9 games usually upload the *transpose* of a matrix so
  that c0..c3 can be dotted against the position, so every candidate is tested
  both ways round.

No numpy: keeping the dependency list at "mcp" makes this trivial to install
next to the other RE servers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Sequence

EPS_ZERO = 1e-4
EPS_ONE = 1e-3


# --------------------------------------------------------------------------
# small vector helpers
# --------------------------------------------------------------------------

Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]


def _dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length3(v: Sequence[float]) -> float:
    return math.sqrt(max(_dot3(v, v), 0.0))


def _normalize3(v: Sequence[float]) -> Vec3:
    n = _length3(v)
    if n < 1e-20:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def transpose16(m: Sequence[float]) -> list[float]:
    return [m[col * 4 + row] for row in range(4) for col in range(4)]


def _round(value: float, digits: int = 6) -> float:
    if not math.isfinite(value):
        return value
    return round(value, digits)


def _finite(m: Sequence[float]) -> bool:
    return all(math.isfinite(v) for v in m)


def _solve3(rows: Sequence[Sequence[float]], rhs: Sequence[float]) -> Vec3 | None:
    """Cramer's rule for a 3x3 system; None if near-singular."""

    def det3(a: Sequence[Sequence[float]]) -> float:
        return (
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        )

    base = det3(rows)
    if abs(base) < 1e-12:
        return None
    out = []
    for col in range(3):
        swapped = [list(r) for r in rows]
        for r in range(3):
            swapped[r][col] = rhs[r]
        out.append(det3(swapped) / base)
    return (out[0], out[1], out[2])


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


@dataclass
class MatrixDecode:
    kind: str  # projection | rigid | viewproj | ortho | unknown
    confidence: float
    layout: str  # as_stored | transposed
    convention: str | None = None  # d3d | gl
    notes: list[str] = field(default_factory=list)

    fov_y_deg: float | None = None
    fov_x_deg: float | None = None
    aspect: float | None = None
    near: float | None = None
    far: float | None = None
    depth_range: str | None = None
    alt_near_far: str | None = None
    handedness: str | None = None
    reversed_z: bool | None = None

    eye: Vec3 | None = None
    axis_x: Vec3 | None = None
    axis_y: Vec3 | None = None
    axis_z: Vec3 | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None and v != []}


def _score_zeros(values: Sequence[float], tol: float = EPS_ZERO) -> float:
    """1.0 when every value is ~0, decaying as they deviate."""
    if not values:
        return 1.0
    worst = max(abs(v) for v in values)
    if worst <= tol:
        return 1.0
    return max(0.0, 1.0 - min(worst / (tol * 100.0), 1.0))


@dataclass(frozen=True)
class _DepthDecode:
    """Physical clip distances, independent of the depth-buffer direction."""

    near: float
    far: float
    reversed_z: bool = False


def _positive_distance(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _ordered_depth(a: float, b: float) -> _DepthDecode | None:
    """Order the distances at the two NDC endpoints into physical near/far."""
    if not (_positive_distance(a) and _positive_distance(b)):
        return None
    if abs(a - b) <= max(abs(a), abs(b), 1.0) * 1e-10:
        return None
    if a < b:
        return _DepthDecode(a, b, False)
    return _DepthDecode(b, a, True)


def _near_far_d3d(q: float, m14: float, left_handed: bool) -> _DepthDecode | None:
    """Decode D3D's 0..1 depth mapping, including reversed-Z and infinite far.

    With positive view distance ``d`` the matrix maps depth as ``A + B/d``.
    The distances at NDC 0 and 1 identify the physical planes; their ordering
    tells us whether the buffer is reversed.  This formulation handles both
    handednesses without separate, fragile near/far formulae.
    """
    w = 1.0 if left_handed else -1.0
    a = q / w
    b = m14
    tol = 1e-7

    # Conventional infinite far: depth approaches 1 as d -> infinity.
    if abs(a - 1.0) <= tol:
        near = -b / a if abs(a) > tol else math.nan
        return _DepthDecode(near, math.inf, False) if _positive_distance(near) else None

    # Reversed-Z infinite far: depth approaches 0 as d -> infinity.
    if abs(a) <= tol:
        near = b / (1.0 - a)
        return _DepthDecode(near, math.inf, True) if _positive_distance(near) else None

    at_zero = -b / a
    at_one = b / (1.0 - a)
    return _ordered_depth(at_zero, at_one)


def _near_far_gl(q: float, m14: float) -> _DepthDecode | None:
    """Decode OpenGL's -1..1 mapping, including reversed-Z and infinite far."""
    # OpenGL perspective matrices have w=-1, so NDC depth tends to A=-q.
    a = -q
    b = m14
    tol = 1e-7

    # Conventional far maps to +1; reversed-Z far maps to -1.
    if abs(a - 1.0) <= tol:
        near = b / (-1.0 - a)
        return _DepthDecode(near, math.inf, False) if _positive_distance(near) else None
    if abs(a + 1.0) <= tol:
        near = b / (1.0 - a)
        return _DepthDecode(near, math.inf, True) if _positive_distance(near) else None

    at_minus_one = b / (-1.0 - a)
    at_plus_one = b / (1.0 - a)
    return _ordered_depth(at_minus_one, at_plus_one)


def _perspective_span(scale: float, shift: float, w: float) -> tuple[float, float, float] | None:
    """Return total FOV and the two side angles for one projection axis.

    ``shift`` is m[8] or m[9].  Dividing it by the perspective-w sign gives
    the NDC centre shift for both D3D row-major and GL column-major layouts.
    This is the piece the symmetric ``2*atan(1/scale)`` shortcut loses.
    """
    if abs(scale) < 1e-9 or abs(w) < 1e-9:
        return None
    centre = shift / w
    if not math.isfinite(centre) or abs(centre) > 2.0:
        return None
    side_a = math.atan((-1.0 - centre) / scale)
    side_b = math.atan((1.0 - centre) / scale)
    low, high = sorted((side_a, side_b))
    span = math.degrees(high - low)
    if not math.isfinite(span) or not (1.0 < span < 179.0):
        return None
    return span, math.degrees(low), math.degrees(high)


def _decode_projection(
    m: Sequence[float], convention: str | None = None
) -> MatrixDecode | None:
    """Perspective projection in D3D row-major storage.

        [ xs  0   0   0 ]
        [ 0   ys  0   0 ]
        [ 0   0   q  +-1]
        [ 0   0  zq   0 ]

    An OpenGL projection stored column-major has the identical flat layout, and
    the two differ only in how depth is encoded (NDC 0..1 vs -1..1). That is not
    recoverable from the numbers alone, so `convention` decides -- callers that
    know the API (from the function name, or the trace) should pass it, and when
    they cannot, both readings are reported.
    """
    xs, ys, q, w = m[0], m[5], m[10], m[11]
    if abs(m[15]) > EPS_ONE or abs(abs(w) - 1.0) > 0.05:
        return None
    if abs(xs) < 1e-9 or abs(ys) < 1e-9:
        return None

    # A perspective projection has only scale, centre shift, depth and w terms.
    # m12/m13 used to be omitted here, which let translated affine/noise blocks
    # masquerade as projections at high confidence.
    off_diag = [m[1], m[2], m[3], m[4], m[6], m[7], m[12], m[13]]
    structural_tol = max(EPS_ZERO, max(abs(xs), abs(ys), 1.0) * 1e-5)
    if max(abs(v) for v in off_diag) > structural_tol:
        return None

    x_span = _perspective_span(xs, m[8], w)
    y_span = _perspective_span(ys, m[9], w)
    if x_span is None or y_span is None:
        return None

    d3d_depth = _near_far_d3d(q, m[14], w > 0)
    gl_depth = _near_far_gl(q, m[14])
    if convention == "d3d" and d3d_depth is None:
        return None
    if convention == "gl" and gl_depth is None:
        return None
    if convention is None and d3d_depth is None and gl_depth is None:
        return None

    score = _score_zeros(off_diag) * 0.6 + _score_zeros([m[15]], EPS_ONE) * 0.2
    score += (1.0 - min(abs(abs(w) - 1.0) / 0.05, 1.0)) * 0.2

    dec = MatrixDecode(kind="projection", confidence=round(score, 3), layout="as_stored")
    dec._raw_confidence = score  # unrounded; classify() compares this, not `confidence`
    dec.handedness = "left" if w > 0 else "right"

    dec.fov_x_deg = _round(x_span[0], 4)
    dec.fov_y_deg = _round(y_span[0], 4)
    dec.aspect = _round(abs(ys / xs), 6)

    # Off-centre projections (VR-style, or a game rendering to a sub-rect) put a
    # non-zero shift in row 2; worth surfacing because it breaks naive stereo.
    if abs(m[8]) > EPS_ZERO or abs(m[9]) > EPS_ZERO:
        dec.notes.append(
            f"off-centre projection: x_shift={_round(m[8])} y_shift={_round(m[9])}"
        )
        dec.notes.append(
            "frustum side angles: "
            f"left={_round(x_span[1], 4)} right={_round(x_span[2], 4)} "
            f"bottom={_round(y_span[1], 4)} top={_round(y_span[2], 4)}"
        )

    def _apply(depth: _DepthDecode | None) -> None:
        if depth is None:
            return
        dec.near, dec.far = _round(depth.near, 5), _round(depth.far, 5)
        dec.reversed_z = depth.reversed_z
        if depth.reversed_z:
            dec.notes.append("reversed-Z depth")
        if math.isinf(depth.far):
            dec.notes.append("infinite far plane")

    def _depth_text(depth: _DepthDecode, label: str) -> str:
        suffix = ", reversed-Z" if depth.reversed_z else ""
        return (
            f"{label}: near={_round(depth.near, 5)} "
            f"far={_round(depth.far, 5)}{suffix}"
        )

    if convention == "gl":
        dec.convention = "gl"
        dec.depth_range = "-1..1 (OpenGL)"
        _apply(gl_depth)
        if d3d_depth:
            dec.alt_near_far = _depth_text(d3d_depth, "if read as D3D depth")
    elif convention == "d3d":
        dec.convention = "d3d"
        dec.depth_range = "0..1 (Direct3D)"
        _apply(d3d_depth)
        if gl_depth:
            dec.alt_near_far = _depth_text(gl_depth, "if read as OpenGL depth")
    else:
        primary = d3d_depth or gl_depth
        dec.depth_range = (
            "unknown -- near/far shown for D3D depth mapping"
            if d3d_depth
            else "unknown -- matrix only has a plausible OpenGL depth mapping"
        )
        _apply(primary)
        if gl_depth and d3d_depth:
            dec.alt_near_far = (
                _depth_text(gl_depth, "if this is an OpenGL projection")
                + ". FOV and aspect are unaffected."
            )
    return dec


def _decode_ortho(m: Sequence[float], convention: str | None = None) -> MatrixDecode | None:
    """Orthographic projection -- overwhelmingly a HUD/UI or shadow-map pass."""
    if abs(m[15] - 1.0) > EPS_ONE or abs(m[11]) > EPS_ZERO:
        return None
    off_diag = [m[1], m[2], m[3], m[4], m[6], m[7], m[8], m[9]]
    if max(abs(v) for v in off_diag) > EPS_ZERO:
        return None
    if abs(m[0]) < 1e-9 or abs(m[5]) < 1e-9 or abs(m[10]) < 1e-12:
        return None
    dec = MatrixDecode(
        kind="ortho",
        confidence=round(_score_zeros(off_diag), 3),
        layout="as_stored",
        convention=convention,
    )
    dec._raw_confidence = _score_zeros(off_diag)
    dec.notes.append(
        f"width={_round(2.0 / m[0])} height={_round(2.0 / m[5])} "
        "(typical of HUD/UI or shadow passes)"
    )
    return dec


def _decode_rigid(m: Sequence[float], convention: str | None = None) -> MatrixDecode | None:
    """Orthonormal affine transform -- a view matrix, or a model matrix."""
    if abs(m[3]) > EPS_ZERO or abs(m[7]) > EPS_ZERO or abs(m[11]) > EPS_ZERO:
        return None
    if abs(m[15] - 1.0) > EPS_ONE:
        return None

    ax = (m[0], m[4], m[8])
    ay = (m[1], m[5], m[9])
    az = (m[2], m[6], m[10])
    lengths = [_length3(ax), _length3(ay), _length3(az)]
    if min(lengths) < 1e-6:
        return None
    unit_err = max(abs(v - 1.0) for v in lengths)
    ortho_err = max(abs(_dot3(ax, ay)), abs(_dot3(ax, az)), abs(_dot3(ay, az)))
    if unit_err > 0.02 or ortho_err > 0.02:
        return None

    raw_confidence = 1.0 - min((unit_err + ortho_err) / 0.04, 1.0) * 0.5
    dec = MatrixDecode(
        kind="rigid",
        confidence=round(raw_confidence, 3),
        layout="as_stored",
        convention=convention,
    )
    dec._raw_confidence = raw_confidence
    dec.axis_x = tuple(_round(v) for v in ax)  # type: ignore[assignment]
    dec.axis_y = tuple(_round(v) for v in ay)  # type: ignore[assignment]
    dec.axis_z = tuple(_round(v) for v in az)  # type: ignore[assignment]

    # If this is a view matrix, the translation row holds -(eye . axes), so the
    # world-space eye is -(R . t) with R read row-major.
    t = (m[12], m[13], m[14])
    eye = (
        -(m[0] * t[0] + m[1] * t[1] + m[2] * t[2]),
        -(m[4] * t[0] + m[5] * t[1] + m[6] * t[2]),
        -(m[8] * t[0] + m[9] * t[1] + m[10] * t[2]),
    )
    dec.eye = tuple(_round(v, 4) for v in eye)  # type: ignore[assignment]
    dec.notes.append(
        "eye/axes are only meaningful if this is a VIEW matrix; for a WORLD/model "
        "matrix the translation row is the object position instead"
    )
    return dec


def _decode_viewproj(
    m: Sequence[float], convention: str | None = None
) -> MatrixDecode | None:
    """A combined view*projection -- what shader-era games upload as constants.

    Uses Gribb-Hartmann plane extraction, which is exact for any world->clip
    matrix, then recovers the eye as the apex where left/right/top meet and the
    FOV as the angle between opposing plane pairs.
    """
    col = [tuple(m[row * 4 + c] for row in range(4)) for c in range(4)]
    c0, c1, c2, c3 = col

    # A pure projection or a rigid transform is handled by the other decoders;
    # this one wants the genuinely combined case.
    if max(abs(v) for v in (m[3], m[7], m[11])) < EPS_ZERO:
        return None

    left = tuple(c3[i] + c0[i] for i in range(4))
    right = tuple(c3[i] - c0[i] for i in range(4))
    bottom = tuple(c3[i] + c1[i] for i in range(4))
    top = tuple(c3[i] - c1[i] for i in range(4))

    nl, nr = _normalize3(left), _normalize3(right)
    nb, nt = _normalize3(bottom), _normalize3(top)
    if min(_length3(left), _length3(right), _length3(top), _length3(bottom)) < 1e-12:
        return None

    dec = MatrixDecode(
        kind="viewproj", confidence=0.6, layout="as_stored", convention=convention
    )

    def _angle(a: Vec3, b: Vec3) -> float | None:
        d = max(-1.0, min(1.0, _dot3(a, b)))
        ang = math.pi - math.acos(d)
        return math.degrees(ang) if 0.0 < ang < math.pi else None

    # Everything from here to the eye solve is a plausibility gate, not a decode.
    # This is by far the loosest of the four decoders -- it accepts any matrix
    # with a perspective w-column -- so without gates it happily "finds"
    # view-projections in vertex buffers, reporting 180-degree fields of view and
    # aspect ratios in the hundreds. Scanning a real trace's buffer uploads
    # produced 2104 such slots before these checks existed.
    fov_y = _angle(nt, nb)
    fov_x = _angle(nl, nr)
    if fov_y is None or fov_x is None:
        return None
    if not (5.0 < fov_y < 175.0 and 5.0 < fov_x < 175.0):
        return None
    try:
        aspect = math.tan(math.radians(fov_x) / 2) / math.tan(math.radians(fov_y) / 2)
    except (ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(aspect) or not (0.1 < aspect < 10.0):
        return None

    dec.fov_y_deg = _round(fov_y, 4)
    dec.fov_x_deg = _round(fov_x, 4)
    dec.aspect = _round(aspect, 5)

    # Eye = the point common to left/right/top -- the frustum apex.
    #
    # Three planes always intersect somewhere, so a solution alone proves
    # nothing: feeding this decoder a transposed matrix (where the planes should
    # have come from the rows instead of the columns) still yields a plausible
    # looking point. The bottom plane is deliberately held back from the solve
    # and used to verify it -- a genuine frustum apex lies on all four side
    # planes, a spurious one does not. That residual is what lets classify()
    # tell the two storage orders apart.
    score = 0.6

    eye = _solve3(
        [nl, nr, nt],
        [-left[3] / _length3(left), -right[3] / _length3(right), -top[3] / _length3(top)],
    )
    if eye is None or not all(math.isfinite(v) for v in eye):
        return None
    residual = abs(_dot3(nb, eye) + bottom[3] / _length3(bottom))
    if residual / max(1.0, _length3(eye)) >= 1e-3:
        # The four side planes do not meet at one point, so this is not a
        # world->clip matrix however plausible the individual numbers looked.
        return None
    dec.eye = tuple(_round(v, 4) for v in eye)  # type: ignore[assignment]
    score += 0.15

    # Storage-order discriminator. In a row-vector world->clip matrix, clip.w is
    # v . column3, and column3's xyz is the view matrix's forward axis -- a unit
    # vector. Read the same matrix in the wrong order and column3 becomes the
    # translation row, whose length is arbitrary. Both orders can still produce a
    # consistent frustum apex (the transpose of a projective matrix is also
    # projective), so this length test is what actually separates them.
    #
    # It is a preference, not a veto: a world*view*projection legitimately
    # carries object scale here, and those are worth reporting too.
    view_dir = c3[:3]
    dir_len = _length3(view_dir)
    if not (1e-4 < dir_len < 1e4):
        return None
    dec.axis_z = tuple(_round(v) for v in _normalize3(view_dir))  # type: ignore[assignment]
    if abs(dir_len - 1.0) < 0.02:
        score += 0.15
    else:
        dec.notes.append(
            f"w-column length is {_round(dir_len, 4)}, not 1 -- either a "
            "world*view*projection carrying object scale, or this is the "
            "wrong storage order"
        )
    dec.confidence = round(score, 3)
    dec._raw_confidence = score

    dec.notes.append(
        "combined world/view->clip matrix; FOV and eye derived from frustum planes"
    )
    return dec


# Order matters: an axis-aligned view matrix is structurally identical to a
# unit-scale orthographic projection, and both decoders score it 1.0. Ties go to
# whichever runs first, and "rigid" is the reading that is useful here -- real
# HUD/shadow ortho matrices have scales far from 1 and fail the rigid test
# anyway, so they still land on _decode_ortho.
_DECODERS = (_decode_projection, _decode_rigid, _decode_ortho, _decode_viewproj)

# Raw scores this close count as a tie; larger gaps (e.g. the |w|-1 penalty a
# transposed finite-far projection picks up) decide the layout outright.
_TIE_EPS = 1e-9


def _projection_plausibility(dec: MatrixDecode) -> int:
    """Rank tied projection readings by whether they look like a game camera.

    A transposed infinite-far projection also decodes in the wrong storage
    order -- as a reversed-Z frustum with near=0.5, far=1.0 -- and both readings
    can score exactly 1.0. No real camera has its far plane at twice the near
    plane, so prefer conventional depth and a far/near ratio above ~2.
    """
    score = 0
    if not dec.reversed_z:
        score += 1
    if (
        dec.near is not None
        and dec.far is not None
        and dec.near > 0.0
        and (math.isinf(dec.far) or dec.far > dec.near * 2.0)
    ):
        score += 1
    return score


def _alt_reading_note(dec: MatrixDecode) -> str:
    """One-line summary of a tied alternate-layout projection reading."""
    suffix = ", reversed-Z" if dec.reversed_z else ""
    return (
        f"equally plausible {dec.layout} reading: handedness={dec.handedness} "
        f"near={dec.near} far={dec.far}{suffix}"
    )


def classify(m16: Sequence[float], convention: str | None = None) -> MatrixDecode | None:
    """Best interpretation of 16 floats, trying both storage orders.

    `convention` ('d3d' or 'gl') only affects how depth is read out of a
    projection matrix; pass it when the source API is known.
    """
    if len(m16) != 16 or not _finite(m16):
        return None
    if all(abs(v) < 1e-12 for v in m16):
        return None

    best: MatrixDecode | None = None
    best_raw = -math.inf
    for layout, candidate in (("as_stored", list(m16)), ("transposed", transpose16(m16))):
        for decoder in _DECODERS:
            try:
                dec = decoder(candidate, convention)
            except (ValueError, ZeroDivisionError, OverflowError):
                dec = None
            if dec is None:
                continue
            dec.layout = layout
            if layout == "transposed":
                dec.notes.append(
                    "matched after transpose -- i.e. stored column-major (OpenGL "
                    "layout, or a D3D9 shader constant holding the transpose)"
                )
            # Compare unrounded scores: rounding to 3 digits erased the small
            # |w|-1 penalty that separates the two storage orders of a
            # projection, letting whichever layout ran first win the tie.
            raw = getattr(dec, "_raw_confidence", dec.confidence)
            if best is None or raw > best_raw + _TIE_EPS:
                best, best_raw = dec, raw
            elif (
                abs(raw - best_raw) <= _TIE_EPS
                and dec.kind == "projection"
                and best.kind == "projection"
            ):
                # Exact tie between the two storage orders of a projection
                # (e.g. the transpose of an infinite-far matrix with zn=1):
                # prefer the physically plausible reading, and surface the
                # ambiguity when neither dominates.
                held, challenger = _projection_plausibility(best), _projection_plausibility(dec)
                if challenger > held:
                    best, best_raw = dec, raw
                elif challenger == held:
                    best.notes.append(_alt_reading_note(dec))
    return best


def is_identity(m16: Sequence[float], tol: float = 1e-5) -> bool:
    ident = [1.0 if i % 5 == 0 else 0.0 for i in range(16)]
    return all(abs(a - b) <= tol for a, b in zip(m16, ident))


def format_matrix(m16: Sequence[float], digits: int = 5) -> list[str]:
    """Pretty 4x4 rows, in storage order."""
    return [
        "  ".join(f"{m16[r * 4 + c]:>{digits + 6}.{digits}f}" for c in range(4))
        for r in range(4)
    ]
