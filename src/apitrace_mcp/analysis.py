"""Trace analysis built on the pickle stream.

The tools here answer the questions that come up when reverse-engineering an
old game's renderer, rather than just re-exposing apitrace's CLI:

* what does this frame actually do (histogram, draw list)?
* where does the camera live (matrix slots, per-frame camera track)?
* which shader constant register holds the view-projection matrix?

Everything streams and everything is bounded -- a 4 GB trace must never turn
into a 4 GB reply.
"""

from __future__ import annotations

import json
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import matrices as mx
from .config import ApitraceRoot
from .pickleparse import Call, harvest_floats, iter_calls
from .runner import run

# ---------------------------------------------------------------------------
# function name sets (substring, case-insensitive -- covers D3D7/8/9 and the
# Ex/versioned variants without enumerating every interface name)
# ---------------------------------------------------------------------------

FIXED_FUNCTION_MATRIX = ("settransform", "multiplytransform")

VS_CONSTANT_SETTERS = (
    "setvertexshaderconstantf",
    "setvertexshaderconstant",  # D3D8 spelling
)
PS_CONSTANT_SETTERS = ("setpixelshaderconstantf", "setpixelshaderconstant")

GL_FIXED_MATRIX_CALLS = (
    "glloadmatrixf",
    "glloadmatrixd",
    "glmultmatrixf",
    "glmultmatrixd",
    "glloadtransposematrixf",
    "glloadtransposematrixd",
    "glmulttransposematrixf",
    "glmulttransposematrixd",
)
GL_UNIFORM_MATRIX_CALLS = (
    "gluniformmatrix4fv",
    "gluniformmatrix4dv",
    "glprogramuniformmatrix4fv",
    "glprogramuniformmatrix4dv",
)
GL_MATRIX_CALLS = GL_FIXED_MATRIX_CALLS + GL_UNIFORM_MATRIX_CALLS
# Vendor-suffixed GL entry points (glUniformMatrix4fvARB, glUseProgramObjectARB,
# glLoadTransposeMatrixfARB, ...) behave like the core call. Anchored at the end
# so only a trailing suffix token is stripped, never letters inside a name.
_GL_VENDOR_SUFFIX = re.compile(r"(?:arb|ext|nv)$")
GL_FRUSTUM_CALLS = ("glfrustum", "glortho")
GL_PROGRAM_PARAM = (
    "glprogramlocalparameter4fv",
    "glprogramenvparameter4fv",
    "glprogramlocalparameters4fv",
    "glprogramenvparameters4fv",
)

# Draw calls are identified by apitrace's own RENDER flag, not by name matching.

# Buffer uploads that plausibly carry matrices: GL uniform buffers, D3D10/11
# constant buffers, and the synthetic memcpy calls apitrace emits for writes to
# mapped memory (which is how D3D9 dynamic constant/vertex buffers show up).
BUFFER_UPLOADS = (
    "glbufferdata",
    "glbuffersubdata",
    "glnamedbufferdata",
    "glnamedbuffersubdata",
    "updatesubresource",
    "memcpy",
)

MAX_CONST_REGISTERS = 256

# Matrices live in small constant/uniform buffers. Capping the blob size keeps
# multi-megabyte vertex and index uploads -- which are pure false-positive fuel
# and slow to scan -- out of the search.
MAX_BLOB_SCAN_BYTES = 64 * 1024

# Buffer targets that never hold matrices. Geometry is the worst false-positive
# source there is: slide a 16-float window along a vertex buffer and some of
# those windows will satisfy any structural test you care to write.
NON_CONSTANT_TARGETS = {
    "GL_ARRAY_BUFFER",
    "GL_ELEMENT_ARRAY_BUFFER",
    "GL_PIXEL_PACK_BUFFER",
    "GL_PIXEL_UNPACK_BUFFER",
    "GL_COPY_READ_BUFFER",
    "GL_COPY_WRITE_BUFFER",
    "GL_TEXTURE_BUFFER",
    "GL_DRAW_INDIRECT_BUFFER",
    "GL_QUERY_BUFFER",
}

# Blob candidates arrive with no context -- no function name saying "this is the
# view matrix", no register number. The only thing vouching for them is their own
# structure, so they have to clear the top confidence bar (which requires the
# w-column to be a unit view direction) rather than the general one.
BLOB_MIN_CONFIDENCE = 0.9
MAX_MATRIX_SLOTS = 4096
MAX_DISTINCT_VALUES_PER_SLOT = 512
MAX_BLOB_EYE_DISTANCE = 100_000.0
MAX_GL_MATRIX_STATES = 1024
MAX_GL_CONTEXT_BINDINGS = 4096
MAX_GL_MATRIX_STACK_DEPTH = 256


def matches(name: str, needles: Iterable[str]) -> bool:
    low = name.lower()
    return any(n in low for n in needles)


def convention_of(name: str) -> str | None:
    """Which depth convention a call implies -- decides how near/far is read."""
    low = name.lower()
    if low.startswith("gl") or low.startswith("wgl") or low.startswith("egl"):
        return "gl"
    if "direct3d" in low or low.startswith("idirect3d") or "d3d" in low:
        return "d3d"
    return None


# ---------------------------------------------------------------------------
# frame index
# ---------------------------------------------------------------------------


_LONE_BACKSLASH = re.compile(r"\\\\|\\")


def _loads_lenient(text: str) -> dict:
    """Parse apitrace's JSON, working around unescaped Windows paths.

    `apitrace info` writes the trace path into its JSON verbatim -- so on Windows
    you get "FileName": "C:\\Program Files\\..." with single backslashes, which is not
    legal JSON. Upstream bug; strict parsing is tried first so correct output is
    never touched, and only the fallback doubles lone backslashes.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    repaired = _LONE_BACKSLASH.sub(
        lambda m: m.group(0) if m.group(0) == "\\\\" else "\\\\", text
    )
    return json.loads(repaired)


def trace_info(root: ApitraceRoot, trace: str | Path, dump_frames: bool = False) -> dict:
    cmd = [str(root.apitrace_exe), "info"]
    if dump_frames:
        cmd.append("--dump-frames")
    cmd.append(str(trace))
    stdout, stderr, _ = run(cmd, timeout=600, max_bytes=8_000_000)
    try:
        return _loads_lenient(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"apitrace info returned unparseable JSON: {stdout[:400]!r} / {stderr[:400]!r}"
        ) from exc


# ---------------------------------------------------------------------------
# histogram / frame summary
# ---------------------------------------------------------------------------


def histogram(
    root: ApitraceRoot,
    trace: str | Path,
    *,
    calls: str | None = None,
    limit: int | None = None,
    top: int = 60,
) -> dict:
    counts: Counter[str] = Counter()
    draws = frames = total = 0
    first_no = last_no = None
    for call in iter_calls(root, trace, calls=calls, limit=limit):
        counts[call.name] += 1
        total += 1
        if call.is_draw:
            draws += 1
        if call.is_frame_end:
            frames += 1
        if first_no is None:
            first_no = call.no
        last_no = call.no
    return {
        "calls_scanned": total,
        "call_range": [first_no, last_no],
        "frames_ended": frames,
        "draw_calls": draws,
        "distinct_functions": len(counts),
        "top_functions": [
            {"function": name, "count": n} for name, n in counts.most_common(top)
        ],
        "truncated": limit is not None and total >= limit,
    }


def frame_summary(
    root: ApitraceRoot,
    trace: str | Path,
    *,
    calls: str,
    limit: int | None = None,
) -> dict:
    """What a frame is made of: draws, render-target switches, shader binds."""
    counts: Counter[str] = Counter()
    draws: list[dict] = []
    rt_switches: list[dict] = []
    shader_sets: Counter[str] = Counter()
    total = 0
    draw_count = 0
    rt_switch_count = 0

    for call in iter_calls(root, trace, calls=calls, limit=limit):
        total += 1
        counts[call.name] += 1
        low = call.name.lower()
        if call.is_draw:
            draw_count += 1
            if len(draws) < 200:
                draws.append(
                    {
                        "no": call.no,
                        "function": call.name,
                        "args": {
                            k: v
                            for k, v in ((n, val) for n, val in call.args)
                            if isinstance(v, (int, float, str))
                        },
                    }
                )
        if "setrendertarget" in low or "omsetrendertargets" in low or "bindframebuffer" in low:
            rt_switch_count += 1
            if len(rt_switches) < 100:
                rt_switches.append({"no": call.no, "function": call.name})
        if "setvertexshader" in low or "setpixelshader" in low or "useprogram" in low:
            shader_sets[call.name] += 1

    return {
        "calls_scanned": total,
        "draw_call_count": draw_count,
        "draw_calls": draws,
        "draw_calls_truncated": draw_count > len(draws),
        "render_target_switch_count": rt_switch_count,
        "render_target_switches": rt_switches,
        "render_target_switches_truncated": rt_switch_count > len(rt_switches),
        "shader_binds": dict(shader_sets),
        "top_functions": [
            {"function": n, "count": c} for n, c in counts.most_common(30)
        ],
    }


# ---------------------------------------------------------------------------
# matrix discovery
# ---------------------------------------------------------------------------


@dataclass
class MatrixSample:
    call_no: int
    frame: int
    matrix: list[float]
    decode: mx.MatrixDecode | None
    parameters: dict[str, float] | None = None


@dataclass
class MatrixSlot:
    """One place matrices come from -- a transform state, or a register range."""

    source: str
    function: str
    hits: int = 0
    distinct: set = field(default_factory=set)
    kinds: Counter = field(default_factory=Counter)
    samples: list[MatrixSample] = field(default_factory=list)
    first_call: int | None = None
    last_call: int | None = None
    frames_seen: int = 0
    distinct_truncated: bool = False
    _current_frame: int | None = None
    _current_value: tuple[float, ...] | None = None
    _last_final_value: tuple[float, ...] | None = None
    _frame_comparisons: int = 0
    _frame_changes: int = 0

    def add(self, sample: MatrixSample, max_samples: int) -> None:
        self.hits += 1
        value = tuple(round(v, 4) for v in sample.matrix)
        if value not in self.distinct:
            if len(self.distinct) < MAX_DISTINCT_VALUES_PER_SLOT:
                self.distinct.add(value)
            else:
                self.distinct_truncated = True
        if sample.decode:
            self.kinds[sample.decode.kind] += 1
        if self.first_call is None:
            self.first_call = sample.call_no
        self.last_call = sample.call_no
        if self._current_frame is None:
            self._current_frame = sample.frame
            self._current_value = value
            self.frames_seen = 1
        elif sample.frame != self._current_frame:
            if self._last_final_value is not None and self._current_value is not None:
                self._frame_comparisons += 1
                if self._current_value != self._last_final_value:
                    self._frame_changes += 1
            self._last_final_value = self._current_value
            self._current_frame = sample.frame
            self._current_value = value
            self.frames_seen += 1
        else:
            # The last write in a frame is the state carried into its draw calls.
            self._current_value = value
        if max_samples <= 0:
            return
        if len(self.samples) < max_samples:
            self.samples.append(sample)
        else:
            self.samples[-1] = sample  # keep first N-1 plus the most recent

    def temporal_stats(self) -> tuple[int, int]:
        comparisons = self._frame_comparisons
        changes = self._frame_changes
        if self._last_final_value is not None and self._current_value is not None:
            comparisons += 1
            if self._current_value != self._last_final_value:
                changes += 1
        return comparisons, changes

    def to_dict(self, include_matrix: bool = True) -> dict:
        kind = self.kinds.most_common(1)[0][0] if self.kinds else "unknown"
        comparisons, temporal_changes = self.temporal_stats()
        out: dict[str, Any] = {
            "source": self.source,
            "function": self.function,
            "kind": kind,
            "writes": self.hits,
            "distinct_values": len(self.distinct),
            "distinct_values_truncated": self.distinct_truncated,
            "frames_seen": self.frames_seen,
            "call_range": [self.first_call, self.last_call],
            "frame_transitions_compared": comparisons,
            "temporal_changes": temporal_changes,
            "changes_per_frame": temporal_changes > 0,
            "explicit_view": _is_explicit_view_source(self.source),
        }
        samples = []
        for s in self.samples:
            entry: dict[str, Any] = {"call_no": s.call_no, "frame": s.frame}
            if s.decode:
                entry["decode"] = s.decode.to_dict()
            if s.parameters:
                entry["parameters"] = s.parameters
            if include_matrix and len(s.matrix) == 16:
                entry["rows"] = mx.format_matrix(s.matrix)
            samples.append(entry)
        out["samples"] = samples
        return out


class _RegisterFile:
    """Shadow of a shader constant bank, so piecewise uploads still resolve."""

    def __init__(self) -> None:
        self.regs: dict[int, tuple[float, float, float, float]] = {}

    def write(self, start: int, floats: list[float]) -> tuple[int, int] | None:
        if start < 0 or start >= MAX_CONST_REGISTERS:
            return None
        count = len(floats) // 4
        if count == 0:
            return None
        count = min(count, MAX_CONST_REGISTERS - start)
        for i in range(count):
            chunk = floats[i * 4 : i * 4 + 4]
            if len(chunk) == 4:
                self.regs[start + i] = (chunk[0], chunk[1], chunk[2], chunk[3])
        return start, start + count - 1

    def windows(self, lo: int, hi: int) -> Iterable[tuple[int, list[float]]]:
        """4-register windows overlapping [lo, hi]."""
        start = max(0, lo - 3)
        for base in range(start, hi + 1):
            regs = [self.regs.get(base + k) for k in range(4)]
            if any(r is None for r in regs):
                continue
            flat: list[float] = []
            for r in regs:
                flat.extend(r)  # type: ignore[arg-type]
            yield base, flat


@dataclass
class _MatrixEvent:
    source: str
    function: str
    matrix: list[float]
    floor: float = 0.0
    decode: mx.MatrixDecode | None = None
    parameters: dict[str, float] | None = None


def _arg_ci(call: Call, *names: str, default: Any = None) -> Any:
    wanted = {name.lower() for name in names}
    for name, value in call.args:
        if name.lower() in wanted:
            return value
    return default


def _resource_token(value: Any) -> str:
    """Stable, bounded resource identity suitable for a slot name."""
    if value is None:
        return "?"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (bytes, bytearray)):
        return f"blob:{len(value)}"
    if isinstance(value, (list, tuple, dict, set)):
        return f"{type(value).__name__}:{len(value)}"
    if isinstance(value, (int, float, str)):
        text = str(value)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:96] if text else "?"


def _length3_safe(value: Iterable[float]) -> float:
    parts = list(value)[:3]
    return sum(component * component for component in parts) ** 0.5


def _mul16(a: list[float], b: list[float]) -> list[float]:
    return [
        sum(a[row * 4 + k] * b[k * 4 + col] for k in range(4))
        for row in range(4)
        for col in range(4)
    ]


def _identity16() -> list[float]:
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _gl_mul16(a: list[float], b: list[float]) -> list[float]:
    """Return ``a * b`` for OpenGL column-major flattened matrices."""
    return [
        sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
        for col in range(4)
        for row in range(4)
    ]


_GL_MATRIX_MODE_BY_ENUM = {
    0x1700: "GL_MODELVIEW",
    0x1701: "GL_PROJECTION",
    0x1702: "GL_TEXTURE",
    0x1800: "GL_COLOR",
}


def _gl_matrix_mode(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return _GL_MATRIX_MODE_BY_ENUM.get(value)
    text = str(value).strip().upper()
    for mode in _GL_MATRIX_MODE_BY_ENUM.values():
        if mode in text:
            return mode
    try:
        return _GL_MATRIX_MODE_BY_ENUM.get(int(text, 0))
    except ValueError:
        return None


def _gl_true(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().upper() in ("1", "TRUE", "GL_TRUE")
    return bool(value)


def _null_resource(value: Any) -> bool:
    if value is None or value == 0:
        return True
    return isinstance(value, str) and value.strip().upper() in (
        "",
        "0",
        "0X0",
        "NULL",
        "NONE",
    )


@dataclass
class _GLMatrixState:
    """OpenGL fixed-function matrix stacks for one context (or fallback thread).

    Buffer and program bindings are context state too, so they live here rather
    than on the extractor -- otherwise multi-context traces mislabel resources.
    """

    mode: str = "GL_MODELVIEW"
    stacks: dict[str, list[list[float]]] = field(default_factory=dict)
    program: Any = 0
    buffer_bindings: dict[str, Any] = field(default_factory=dict)
    arb_program_bindings: dict[str, Any] = field(default_factory=dict)

    def _stack(self) -> list[list[float]]:
        return self.stacks.setdefault(self.mode, [_identity16()])

    def current(self) -> list[float]:
        return list(self._stack()[-1])

    def load(self, matrix: list[float]) -> list[float]:
        self._stack()[-1] = list(matrix)
        return self.current()

    def multiply(self, matrix: list[float]) -> list[float]:
        stack = self._stack()
        stack[-1] = _gl_mul16(stack[-1], matrix)
        return self.current()

    def push(self) -> bool:
        stack = self._stack()
        if len(stack) >= MAX_GL_MATRIX_STACK_DEPTH:
            return False
        stack.append(list(stack[-1]))
        return True

    def pop(self) -> list[float] | None:
        stack = self._stack()
        if len(stack) <= 1:
            return None
        stack.pop()
        return self.current()


class _MatrixEventExtractor:
    """One stateful call decoder shared by discovery and camera tracking.

    Keeping this in one place guarantees that a source advertised by
    ``scan_matrices`` can be matched byte-for-byte during ``track_camera``.
    """

    def __init__(self, *, scan_constants: bool = True, scan_buffers: bool = True) -> None:
        self.scan_constants = scan_constants
        self.scan_buffers = scan_buffers
        self.vs_file = _RegisterFile()
        self.ps_file = _RegisterFile()
        self.gl_param_files: dict[str, _RegisterFile] = {}
        # apitrace exposes a thread id for every call and usually records the
        # platform make-current calls too. Prefer the real context identity when
        # recoverable; otherwise isolate fixed-function state by trace thread.
        self.gl_matrix_states: dict[str, _GLMatrixState] = {}
        self.gl_context_by_thread: dict[str, str] = {}
        self._overflow_gl_matrix_state = _GLMatrixState()
        self.fixed_transforms: dict[str, list[float]] = {}

    @staticmethod
    def _thread_scope(call: Call) -> str:
        thread = call.thread_id if call.thread_id is not None else "unknown"
        return f"thread={_resource_token(thread)}"

    @staticmethod
    def _context_scope(api: str, value: Any) -> str:
        return f"context={api}:{_resource_token(value)}"

    @staticmethod
    def _context_call_succeeded(call: Call, *, zero_is_success: bool = False) -> bool:
        if call.ret is None:
            return True
        if zero_is_success:
            if isinstance(call.ret, str):
                return call.ret.strip().upper() in ("0", "NO_ERROR", "K CGL NO ERROR", "KCGLNOERROR")
            return call.ret == 0
        return _gl_true(call.ret)

    def _update_context_binding(self, call: Call) -> bool:
        """Track make-current calls and discard state for destroyed contexts."""
        short = _short(call.name).lower()
        binders = {
            "wglmakecurrent": ("wgl", ("hglrc", "context", "ctx"), 1, False),
            "wglmakecontextcurrentarb": ("wgl", ("hglrc", "context", "ctx"), 2, False),
            "wglmakecontextcurrentext": ("wgl", ("hglrc", "context", "ctx"), 2, False),
            "glxmakecurrent": ("glx", ("ctx", "context"), 2, False),
            "glxmakecontextcurrent": ("glx", ("ctx", "context"), 3, False),
            "eglmakecurrent": ("egl", ("ctx", "context"), 3, False),
            "cglsetcurrentcontext": ("cgl", ("ctx", "context"), 0, True),
        }
        binder = binders.get(short)
        if binder is not None:
            api, names, index, zero_is_success = binder
            if not self._context_call_succeeded(call, zero_is_success=zero_is_success):
                return True
            context = _arg_ci(call, *names, default=call.arg_at(index))
            thread_scope = self._thread_scope(call)
            if _null_resource(context):
                self.gl_context_by_thread.pop(thread_scope, None)
            else:
                scope = self._context_scope(api, context)
                if (
                    thread_scope in self.gl_context_by_thread
                    or len(self.gl_context_by_thread) < MAX_GL_CONTEXT_BINDINGS
                ):
                    self.gl_context_by_thread[thread_scope] = scope
            return True

        destroyers = {
            "wgldeletecontext": ("wgl", ("hglrc", "context", "ctx"), 0),
            "glxdestroycontext": ("glx", ("ctx", "context"), 1),
            "egldestroycontext": ("egl", ("ctx", "context"), 1),
            "cgldestroycontext": ("cgl", ("ctx", "context"), 0),
        }
        destroyer = destroyers.get(short)
        if destroyer is None:
            return False
        api, names, index = destroyer
        context = _arg_ci(call, *names, default=call.arg_at(index))
        if not _null_resource(context):
            scope = self._context_scope(api, context)
            self.gl_matrix_states.pop(scope, None)
            for thread_scope, current_scope in list(self.gl_context_by_thread.items()):
                if current_scope == scope:
                    del self.gl_context_by_thread[thread_scope]
        return True

    def _gl_matrix_state(self, call: Call) -> tuple[_GLMatrixState, str]:
        thread_scope = self._thread_scope(call)
        scope = self.gl_context_by_thread.get(thread_scope, thread_scope)
        state = self.gl_matrix_states.get(scope)
        if state is not None:
            return state, scope
        if len(self.gl_matrix_states) >= MAX_GL_MATRIX_STATES:
            return self._overflow_gl_matrix_state, "scope=overflow"
        state = self.gl_matrix_states[scope] = _GLMatrixState()
        return state, scope

    @staticmethod
    def _gl_fixed_source(short: str, state: _GLMatrixState, scope: str) -> str:
        return f"{short}[{state.mode}]({scope})"

    def _update_state(self, call: Call, gl_state: _GLMatrixState) -> bool:
        """Update binding state; return True when the call has no matrix payload."""
        short = _short(call.name).lower()
        if short == "glmatrixmode":
            mode = _gl_matrix_mode(_arg_ci(call, "mode", default=call.arg_at(0)))
            if mode is not None:
                gl_state.mode = mode
            return True
        if short in ("gluseprogram", "gluseprogramobjectarb"):
            gl_state.program = _arg_ci(
                call, "program", "programObj", default=call.arg_at(0)
            )
            return True
        if short in ("glbindbuffer", "glbindbufferbase", "glbindbufferrange"):
            target = _arg_ci(call, "target", default=call.arg_at(0))
            buffer = _arg_ci(call, "buffer")
            if target is not None and buffer is not None:
                # BindBufferBase/Range also update the generic target binding.
                gl_state.buffer_bindings[str(target)] = buffer
            return True
        if short in ("glbindprogram", "glbindprogramarb", "glbindprogramnv"):
            target = _arg_ci(call, "target", default=call.arg_at(0))
            program = _arg_ci(call, "program", "id", default=call.arg_at(1))
            if target is not None:
                gl_state.arb_program_bindings[str(target)] = program
            return True
        return False

    def _buffer_label(self, call: Call, gl_state: _GLMatrixState) -> str:
        short = _short(call.name)
        low = short.lower()
        if low.startswith("glnamedbuffer"):
            resource = _arg_ci(call, "buffer", "bufferName", default=call.arg_at(0))
            return f"buffer={_resource_token(resource)}"
        if low.startswith("glbuffer"):
            target = _arg_ci(call, "target", default=call.arg_at(0))
            resource = gl_state.buffer_bindings.get(str(target))
            return f"target={_resource_token(target)},buffer={_resource_token(resource)}"
        if "updatesubresource" in low:
            resource = _arg_ci(
                call,
                "pDstResource",
                "DstResource",
                "destinationResource",
                "resource",
            )
            subresource = _arg_ci(call, "DstSubresource", "subresource", default=0)
            return (
                f"resource={_resource_token(resource)},"
                f"subresource={_resource_token(subresource)}"
            )
        if "memcpy" in low:
            destination = _arg_ci(
                call,
                "dest",
                "dst",
                "destination",
                "pDst",
                default=call.arg_at(0),
            )
            return f"dst={_resource_token(destination)}"
        return "resource=?"

    def feed(self, call: Call) -> list[_MatrixEvent]:
        if self._update_context_binding(call):
            return []
        gl_state, gl_scope = self._gl_matrix_state(call)
        if self._update_state(call, gl_state):
            return []

        name = call.name
        low = name.lower()
        short = _short(name)
        short_low = short.lower()
        # ARB/EXT/NV-suffixed entry points behave like the core call.
        short_norm = _GL_VENDOR_SUFFIX.sub("", short_low)
        events: list[_MatrixEvent] = []

        if matches(low, FIXED_FUNCTION_MATRIX):
            state = call.arg("State")
            if state is None:
                state = call.arg_at(0)
            floats = harvest_floats([value for _, value in call.args])
            if len(floats) >= 16:
                incoming = list(floats[-16:])
                key = str(state)
                if "multiplytransform" in low and key in self.fixed_transforms:
                    incoming = _mul16(self.fixed_transforms[key], incoming)
                self.fixed_transforms[key] = incoming
                events.append(_MatrixEvent(f"{short}({state})", name, incoming))
            return events

        if short_low == "glloadidentity":
            matrix = gl_state.load(_identity16())
            return [
                _MatrixEvent(
                    self._gl_fixed_source(short, gl_state, gl_scope), name, matrix
                )
            ]

        if short_low == "glpushmatrix":
            gl_state.push()
            return events

        if short_low == "glpopmatrix":
            matrix = gl_state.pop()
            if matrix is None:
                return events
            return [
                _MatrixEvent(
                    self._gl_fixed_source(short, gl_state, gl_scope), name, matrix
                )
            ]

        if short_norm in GL_MATRIX_CALLS:
            floats = harvest_floats([value for _, value in call.args])
            if len(floats) < 16:
                return events
            matrices = [list(floats[off : off + 16]) for off in range(0, len(floats) - 15, 16)]
            if short_norm in GL_UNIFORM_MATRIX_CALLS:
                if _gl_true(_arg_ci(call, "transpose", default=False)):
                    matrices = [mx.transpose16(matrix) for matrix in matrices]
                location = _arg_ci(call, "location", default=call.arg_at(0))
                explicit_program = _arg_ci(call, "program")
                program = gl_state.program if explicit_program is None else explicit_program
                base = f"{short}(program={_resource_token(program)},location={location})"
                for index, matrix in enumerate(matrices):
                    source = base if len(matrices) == 1 else f"{base}[element={index}]"
                    events.append(_MatrixEvent(source, name, matrix))
                return events

            incoming = matrices[0]
            if "transposematrix" in short_norm:
                incoming = mx.transpose16(incoming)
            if short_norm.startswith("glload"):
                matrix = gl_state.load(incoming)
            else:
                matrix = gl_state.multiply(incoming)
            return [
                _MatrixEvent(
                    self._gl_fixed_source(short, gl_state, gl_scope), name, matrix
                )
            ]

        if short_low in GL_FRUSTUM_CALLS:
            floats = harvest_floats([value for _, value in call.args])
            operation = _gl_projection_operation(name, floats)
            if operation is None:
                return events
            operation_matrix, parameters = operation
            matrix = gl_state.multiply(operation_matrix)
            # glFrustum/glOrtho are legal multiplications in every matrix mode,
            # but treating one on MODELVIEW/TEXTURE as a camera projection creates
            # an especially convincing false positive. Keep the state truthful and
            # only advertise projection-stack results.
            if gl_state.mode != "GL_PROJECTION":
                return events
            return [
                _MatrixEvent(
                    self._gl_fixed_source(short, gl_state, gl_scope),
                    name,
                    matrix,
                    decode=mx.classify(matrix, "gl"),
                    parameters=parameters,
                )
            ]

        if self.scan_buffers and matches(low, BUFFER_UPLOADS):
            label = self._buffer_label(call, gl_state)
            for found_label, byte_off, window in _blob_matrix_windows(call, label=label):
                events.append(
                    _MatrixEvent(
                        f"{short}[{found_label}]+{byte_off}",
                        name,
                        window,
                        floor=BLOB_MIN_CONFIDENCE,
                    )
                )
            return events

        if self.scan_constants and matches(low, VS_CONSTANT_SETTERS + PS_CONSTANT_SETTERS):
            is_ps = matches(low, PS_CONSTANT_SETTERS)
            reg_file = self.ps_file if is_ps else self.vs_file
            bank = "ps" if is_ps else "vs"
            start = _first_int(call, ("StartRegister", "index", "Register", "location"))
            floats = harvest_floats([value for _, value in call.args])
            if not floats:
                # D3D8's Set*ShaderConstant hands pConstantData over as an opaque
                # byte blob rather than a float array; recover the floats so the
                # shadow register file still fills.
                floats = _blob_constant_floats(call)
            if start is None or not floats:
                return events
            written = reg_file.write(int(start), floats)
            if written:
                for base, flat in reg_file.windows(*written):
                    events.append(_MatrixEvent(f"{bank}_c[{base}..{base + 3}]", name, flat))
            return events

        if self.scan_constants and matches(low, GL_PROGRAM_PARAM):
            start = _first_int(call, ("index", "Register", "location"))
            floats = harvest_floats([value for _, value in call.args])
            if start is None or not floats:
                return events
            target = _arg_ci(call, "target", default="?")
            is_env = "envparameter" in low
            program = "environment" if is_env else gl_state.arb_program_bindings.get(str(target))
            bank = (
                f"gl_program[target={_resource_token(target)},"
                f"program={_resource_token(program)}]"
            )
            reg_file = self.gl_param_files.setdefault(bank, _RegisterFile())
            written = reg_file.write(int(start), floats)
            if written:
                for base, flat in reg_file.windows(*written):
                    events.append(_MatrixEvent(f"{bank}_c[{base}..{base + 3}]", name, flat))
            return events

        return events


def scan_matrices(
    root: ApitraceRoot,
    trace: str | Path,
    *,
    calls: str | None = None,
    limit: int | None = None,
    min_confidence: float = 0.5,
    include_kinds: tuple[str, ...] = ("projection", "viewproj", "rigid", "ortho"),
    max_samples: int = 3,
    scan_constants: bool = True,
    scan_buffers: bool = True,
    skip_identity: bool = True,
) -> dict:
    """Find every 4x4 matrix the game hands to the driver, grouped by slot."""
    slots: dict[str, MatrixSlot] = {}
    frame = 0
    scanned = 0
    dropped_slots = 0
    extractor = _MatrixEventExtractor(
        scan_constants=scan_constants,
        scan_buffers=scan_buffers,
    )

    def record(
        source: str,
        function: str,
        m16: list[float],
        call_no: int,
        floor: float | None = None,
        decode: mx.MatrixDecode | None = None,
        parameters: dict[str, float] | None = None,
    ) -> None:
        nonlocal dropped_slots
        if skip_identity and len(m16) == 16 and mx.is_identity(m16):
            return
        decode = decode or mx.classify(m16, convention_of(function))
        if decode is None:
            return
        if decode.confidence < max(min_confidence, floor or 0.0):
            return
        if decode.kind not in include_kinds:
            return
        slot = slots.get(source)
        if slot is None:
            if len(slots) >= MAX_MATRIX_SLOTS:
                dropped_slots += 1
                return
            slot = slots[source] = MatrixSlot(source=source, function=function)
        slot.add(MatrixSample(call_no, frame, m16, decode, parameters), max_samples)

    for call in iter_calls(root, trace, calls=calls, limit=limit):
        scanned += 1
        name = call.name
        low = name.lower()

        if call.is_frame_end:
            frame += 1
            continue
        for event in extractor.feed(call):
            record(
                event.source,
                event.function,
                event.matrix,
                call.no,
                floor=event.floor,
                decode=event.decode,
                parameters=event.parameters,
            )

    ranked = sorted(
        slots.values(),
        key=lambda s: (_kind_rank(s), -s.hits),
    )
    return {
        "calls_scanned": scanned,
        "frames_scanned": frame,
        "slots_found": len(ranked),
        "slots_truncated": dropped_slots > 0,
        "dropped_slot_events": dropped_slots,
        "slots": [s.to_dict() for s in ranked],
        "hint": (
            "kind=projection gives you the game's FOV and near/far. kind=rigid with "
            "changes_per_frame=true is a camera candidate, but moving models, lights, "
            "and shadow views can look the same. kind=viewproj "
            "in a vs_c[...] slot is the register range a VR patch would rewrite."
        ),
    }


def _blob_matrix_windows(
    call: Call,
    *,
    label: str | None = None,
) -> Iterable[tuple[str, int, list[float]]]:
    """Yield (label, byte_offset, 16 floats) for every float4-aligned window in a blob.

    A uniform/constant buffer is just bytes, so the byte offset of a hit is the
    actionable result: it is what you would patch, and what you would look for in
    the code that fills the buffer.
    """
    target = _arg_ci(call, "target")
    target_label = target if isinstance(target, str) else "?"
    if target_label in NON_CONSTANT_TARGETS:
        return
    resource_label = label or target_label

    base = 0
    for arg_name, value in call.args:
        if arg_name.lower() in ("offset", "dstoffset") and isinstance(value, int):
            base = value

    for arg_name, value in call.args:
        if not isinstance(value, (bytes, bytearray)):
            continue
        raw = bytes(value)
        if len(raw) < 64 or len(raw) > MAX_BLOB_SCAN_BYTES:
            continue
        count = len(raw) // 4
        try:
            floats = struct.unpack_from(f"<{count}f", raw, 0)
        except struct.error:
            continue
        candidates: list[tuple[int, list[float], mx.MatrixDecode]] = []
        for off in range(0, count - 15, 4):
            window = list(floats[off : off + 16])
            decode = mx.classify(window, convention_of(call.name))
            if decode is None or decode.confidence < BLOB_MIN_CONFIDENCE:
                continue
            candidates.append((off * 4, window, decode))

        # A 64-byte matrix scanned every float4 produces four overlapping
        # candidates around a real hit.  Keep the strongest non-overlapping set
        # so one physical matrix does not become several fake camera slots.
        kind_order = {"projection": 0, "viewproj": 1, "rigid": 2, "ortho": 3}
        chosen: list[tuple[int, list[float], mx.MatrixDecode]] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (-item[2].confidence, kind_order.get(item[2].kind, 9), item[0]),
        ):
            byte_off = candidate[0]
            if any(abs(byte_off - existing[0]) < 64 for existing in chosen):
                continue
            chosen.append(candidate)

        arg_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", arg_name)[:48] or "blob"
        full_label = f"{resource_label},arg={arg_label}"
        for byte_off, window, decode in sorted(chosen, key=lambda item: item[0]):
            # Apply weak-source plausibility only after overlap suppression.  A
            # rejected candidate must not expose the same bytes again through a
            # shifted 16-byte window (the source of the old +192/+208 ghosts).
            if decode.eye is not None and _length3_safe(decode.eye) > MAX_BLOB_EYE_DISTANCE:
                continue
            if max(abs(item) for item in window) > 1e8:
                continue
            if decode.kind == "viewproj" and (
                decode.fov_x_deg is None
                or decode.fov_y_deg is None
                or decode.fov_x_deg >= 165.0
                or decode.fov_y_deg >= 165.0
            ):
                continue
            yield full_label, base + byte_off, window


def _gl_projection_operation(
    name: str, floats: list[float]
) -> tuple[list[float], dict[str, float]] | None:
    """Build the real column-major matrix multiplied by glFrustum/glOrtho."""
    import math

    if len(floats) < 6:
        return None
    left, right, bottom, top, near, far = floats[:6]
    if not all(math.isfinite(value) for value in (left, right, bottom, top, near, far)):
        return None
    # Inverted extents (glOrtho(0, w, h, 0, ...) for a top-left HUD) are legal
    # GL; only degenerate planes make the matrix undefined.
    if left == right or bottom == top:
        return None

    is_frustum = "frustum" in name.lower()
    if is_frustum and not (near > 0.0 and far > 0.0 and near != far):
        return None
    if not is_frustum and near == far:
        return None

    if is_frustum:
        operation = [
            2.0 * near / (right - left), 0.0, 0.0, 0.0,
            0.0, 2.0 * near / (top - bottom), 0.0, 0.0,
            (right + left) / (right - left),
            (top + bottom) / (top - bottom),
            -(far + near) / (far - near), -1.0,
            0.0, 0.0, -(2.0 * far * near) / (far - near), 0.0,
        ]
    else:
        operation = [
            2.0 / (right - left), 0.0, 0.0, 0.0,
            0.0, 2.0 / (top - bottom), 0.0, 0.0,
            0.0, 0.0, -2.0 / (far - near), 0.0,
            -(right + left) / (right - left),
            -(top + bottom) / (top - bottom),
            -(far + near) / (far - near), 1.0,
        ]
    parameters = {
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
        "near": near,
        "far": far,
    }
    return operation, parameters


def _frustum_event(name: str, floats: list[float], mode: str) -> _MatrixEvent | None:
    """Decode a valid projection-stack glFrustum/glOrtho call."""
    mode_name = _gl_matrix_mode(mode)
    if mode_name != "GL_PROJECTION":
        return None
    operation = _gl_projection_operation(name, floats)
    if operation is None:
        return None
    matrix, parameters = operation
    return _MatrixEvent(
        source=f"{_short(name)}[{mode_name}]",
        function=name,
        matrix=matrix,
        decode=mx.classify(matrix, "gl"),
        parameters=parameters,
    )


def record_frustum(
    slots: dict[str, MatrixSlot],
    name: str,
    floats: list[float],
    call_no: int,
    frame: int,
    mode: str,
) -> None:
    """Compatibility wrapper used by older callers and focused unit tests."""
    event = _frustum_event(name, floats, mode)
    if event is None:
        return
    slot = slots.setdefault(
        event.source, MatrixSlot(source=event.source, function=name)
    )
    slot.add(
        MatrixSample(call_no, frame, event.matrix, event.decode, event.parameters),
        max_samples=3,
    )


def _kind_rank(slot: MatrixSlot) -> int:
    kind = slot.kinds.most_common(1)[0][0] if slot.kinds else "unknown"
    return {"projection": 0, "viewproj": 1, "rigid": 2, "ortho": 3}.get(kind, 4)


def _is_explicit_view_source(source: str) -> bool:
    upper = source.upper()
    return (
        "D3DTS_VIEW" in upper
        or "(VIEW)" in upper
        or "[GL_MODELVIEW]" in upper
    )


def _short(name: str) -> str:
    return name.rsplit("::", 1)[-1]


def _blob_constant_floats(call: Call) -> list[float]:
    """Unpack a constant-setter's raw byte payload as little-endian float32s.

    apitrace's d3d8 spec records SetVertexShaderConstant/SetPixelShaderConstant
    pConstantData as a bytes blob instead of a float array.
    """
    for _, value in call.args:
        if not isinstance(value, (bytes, bytearray)):
            continue
        raw = bytes(value)
        if not raw or len(raw) % 4 != 0 or len(raw) > MAX_BLOB_SCAN_BYTES:
            continue
        try:
            return list(struct.unpack(f"<{len(raw) // 4}f", raw))
        except struct.error:
            continue
    return []


def _first_int(call: Call, names: tuple[str, ...]) -> int | None:
    for n in names:
        value = call.arg(n)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    value = call.arg_at(0)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


# ---------------------------------------------------------------------------
# camera tracking
# ---------------------------------------------------------------------------


def track_camera(
    root: ApitraceRoot,
    trace: str | Path,
    *,
    calls: str | None = None,
    source: str | None = None,
    limit: int | None = None,
    max_frames: int = 200,
) -> dict:
    """Per-frame camera state, so you can watch it move as the player moves.

    `source` is a slot name from find_matrices (e.g. "SetTransform(D3DTS_VIEW)"
    or "vs_c[8..11]"). Without it, the best-looking view-ish slot is chosen.
    """
    scan = scan_matrices(
        root,
        trace,
        calls=calls,
        limit=limit,
        include_kinds=("projection", "viewproj", "rigid"),
        max_samples=1,
    )
    slots = scan["slots"]
    if not slots:
        return {"error": "no matrix slots found in the scanned range", "scan": scan}

    if source:
        chosen = next((s for s in slots if s["source"] == source), None)
        if chosen is None:
            return {
                "error": f"slot {source!r} not found",
                "available": [s["source"] for s in slots],
            }
    else:
        viewish = [s for s in slots if s["kind"] in ("rigid", "viewproj")]
        explicit = [s for s in viewish if s.get("explicit_view")]
        movers = [s for s in viewish if s["changes_per_frame"]]
        pool = explicit or movers or viewish or slots
        chosen = max(
            pool,
            key=lambda item: (
                bool(item.get("explicit_view")),
                int(item.get("temporal_changes", 0)),
                int(item.get("frames_seen", 0)),
                int(item.get("writes", 0)),
            ),
        )

    proj_slot = next((s for s in slots if s["kind"] == "projection"), None)

    # Second pass, collecting the final value written in each frame.  Discovery
    # and tracking deliberately share the exact same stateful extractor.
    track: list[dict] = []
    frame = 0
    extractor = _MatrixEventExtractor(scan_constants=True, scan_buffers=True)
    target = chosen["source"]
    pending: dict[str, Any] | None = None

    def finish_frame() -> bool:
        nonlocal pending
        if max_frames <= 0:
            pending = None
            return True
        if pending is not None:
            track.append(pending)
            pending = None
        return len(track) >= max(0, max_frames)

    for call in iter_calls(root, trace, calls=calls, limit=limit):
        if call.is_frame_end:
            if finish_frame():
                break
            frame += 1
            continue
        for event in extractor.feed(call):
            if event.source != target:
                continue
            decode = event.decode
            if decode is None and len(event.matrix) == 16:
                decode = mx.classify(event.matrix, convention_of(call.name))
            if decode is None:
                continue
            entry: dict[str, Any] = {"frame": frame, "call_no": call.no}
            if decode.eye:
                entry["eye"] = list(decode.eye)
            if decode.axis_z:
                view_z = list(decode.axis_z)
                entry["view_z_axis"] = view_z
                entry["forward_candidates"] = {
                    "left_handed": view_z,
                    "right_handed": [-value for value in view_z],
                }
            if decode.fov_y_deg is not None:
                entry["fov_y_deg"] = decode.fov_y_deg
            if event.parameters:
                entry["parameters"] = event.parameters
            pending = entry

    if len(track) < max(0, max_frames):
        finish_frame()

    result: dict[str, Any] = {
        "slot": chosen["source"],
        "kind": chosen["kind"],
        "function": chosen["function"],
        "frames": track,
        "frame_count": len(track),
        "orientation_note": (
            "A rigid view matrix alone does not identify handedness. view_z_axis is "
            "reported directly; choose the matching forward_candidates vector from "
            "the engine/projection convention."
        ),
    }
    if proj_slot:
        result["projection_slot"] = {
            "source": proj_slot["source"],
            "samples": proj_slot["samples"],
        }
    eye_track = [entry for entry in track if "eye" in entry]
    if len(eye_track) >= 2:
        moved = any(
            any(abs(a - b) > 1e-3 for a, b in zip(eye_track[0]["eye"], entry["eye"]))
            for entry in eye_track[1:]
        )
        result["camera_moves"] = moved
        if not moved:
            result["note"] = (
                "eye is static across the captured frames -- either the player did not "
                "move during capture, or this slot is a model matrix rather than the view"
            )
    return result


# ---------------------------------------------------------------------------
# text dump / search
# ---------------------------------------------------------------------------


def dump_text(
    root: ApitraceRoot,
    trace: str | Path,
    *,
    calls: str | None = None,
    grep: str | None = None,
    max_bytes: int = 60_000,
    blobs: bool = False,
    timeout: int = 300,
) -> tuple[str, bool]:
    cmd = [str(root.apitrace_exe), "dump", "--color=never"]
    if calls:
        cmd.append(f"--calls={calls}")
    if grep:
        cmd.append(f"--grep={grep}")
    if blobs:
        cmd.append("--blobs")
    cmd.append(str(trace))
    stdout, _, truncated = run(cmd, timeout=timeout, max_bytes=max_bytes)
    return stdout, truncated


CALL_NO_RE = re.compile(r"^(\d+)\s")


def search_calls(
    root: ApitraceRoot,
    trace: str | Path,
    pattern: str,
    *,
    calls: str | None = None,
    max_hits: int = 200,
    timeout: int = 600,
) -> dict:
    """Regex over function names; returns counts and where they happen."""
    cmd = [str(root.apitrace_exe), "dump", "--color=never", f"--grep={pattern}"]
    if calls:
        cmd.append(f"--calls={calls}")
    cmd.append(str(trace))
    stdout, _, truncated = run(cmd, timeout=timeout, max_bytes=4_000_000)

    per_function: Counter[str] = Counter()
    hits: list[dict] = []
    for line in stdout.splitlines():
        m = CALL_NO_RE.match(line)
        if not m:
            continue
        call_no = int(m.group(1))
        rest = line[m.end() :]
        fname = rest.split("(", 1)[0].strip()
        per_function[fname] += 1
        if len(hits) < max_hits:
            hits.append({"no": call_no, "line": line[:400]})
    return {
        "pattern": pattern,
        "total_matches": sum(per_function.values()),
        "functions": [{"function": f, "count": c} for f, c in per_function.most_common(50)],
        "hits": hits,
        "hits_truncated": len(hits) >= max_hits or truncated,
    }


# ---------------------------------------------------------------------------
# shaders
# ---------------------------------------------------------------------------

SHADER_CREATORS = (
    "createvertexshader",
    "createpixelshader",
    "setvertexshaderprogram",
    "glshadersource",
    "glprogramstring",
)

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _safe_filename_part(value: Any, fallback: str) -> str:
    text = _SAFE_FILENAME.sub("_", str(value)).strip(" ._")[:72]
    if not text:
        text = fallback
    if text.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        text = "_" + text
    return text


def _write_blob_file(root: Path, filename: str, raw: bytes) -> Path:
    """Create a non-symlink blob file beneath root without overwriting."""
    root = root.resolve()
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    for index in range(1000):
        candidate_name = filename if index == 0 else f"{stem}_{index}{suffix}"
        candidate = root / candidate_name
        resolved = candidate.resolve(strict=False)
        if resolved.parent != root:
            raise ValueError("refusing blob filename that escapes the output directory")
        if candidate.is_symlink():
            continue
        try:
            with candidate.open("xb") as stream:
                stream.write(raw)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"too many filename collisions for {filename!r}")


def _string_sequence_preview(
    values: list[Any] | tuple[Any, ...], max_chars: int = 600
) -> tuple[str, int]:
    """Join shader source logically while retaining only a bounded prefix."""
    parts: list[str] = []
    total = 0
    remaining = max(0, max_chars)
    seen = False
    for value in values:
        if not isinstance(value, str):
            continue
        if seen:
            total += 1
            if remaining:
                parts.append("\n")
                remaining -= 1
        seen = True
        total += len(value)
        if remaining:
            piece = value[:remaining]
            parts.append(piece)
            remaining -= len(piece)
    return "".join(parts), total


def list_shaders(
    root: ApitraceRoot,
    trace: str | Path,
    *,
    calls: str | None = None,
    limit: int | None = None,
    max_shaders: int = 100,
) -> dict:
    if isinstance(max_shaders, bool) or not isinstance(max_shaders, int):
        raise ValueError("max_shaders must be an integer")
    if max_shaders <= 0:
        return {"shaders": [], "count": 0, "truncated": False}
    found: list[dict] = []
    frame = 0
    for call in iter_calls(root, trace, calls=calls, limit=limit):
        if call.is_frame_end:
            frame += 1
            continue
        if not matches(call.name, SHADER_CREATORS):
            continue
        entry: dict[str, Any] = {
            "no": call.no,
            "frame": frame,
            "function": call.name,
            "ret": str(call.ret) if call.ret is not None else None,
        }
        for arg_name, value in call.args:
            if isinstance(value, (bytes, bytearray)):
                entry["bytecode_bytes"] = len(value)
            elif isinstance(value, str) and len(value) > 40:
                entry["source_preview"] = value[:600]
                entry["source_chars"] = len(value)
            elif isinstance(value, (list, tuple)) and value:
                preview, source_chars = _string_sequence_preview(value)
                if source_chars:
                    entry["source_preview"] = preview
                    entry["source_chars"] = source_chars
        found.append(entry)
        if len(found) >= max_shaders:
            break
    return {"shaders": found, "count": len(found), "truncated": len(found) >= max_shaders}


def extract_blobs(
    root: ApitraceRoot,
    trace: str | Path,
    *,
    calls: str,
    out_dir: str | Path,
    limit: int = 50,
) -> dict:
    """Write blob arguments (shader bytecode, vertex/index buffers) to disk."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if limit == 0:
        return {"blobs": [], "count": 0, "truncated": False, "calls_scanned": 0}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out = out.resolve()
    if not out.is_dir():
        raise NotADirectoryError(f"blob output is not a directory: {out}")
    written: list[dict] = []
    # Blobs are sparse -- most calls in any range carry none -- so the walk has to
    # be bounded separately from the number of blobs asked for.
    walk_cap = max(limit * 500, 20000)
    scanned = 0
    for call in iter_calls(root, trace, calls=calls, limit=walk_cap):
        scanned += 1
        for arg_name, value in call.args:
            if not isinstance(value, (bytes, bytearray)):
                continue
            raw = bytes(value)
            function_part = _safe_filename_part(_short(call.name), "call")
            arg_part = _safe_filename_part(arg_name, "blob")
            filename = f"call{call.no}_{function_part}_{arg_part}.bin"
            path = _write_blob_file(out, filename, raw)
            written.append(
                {
                    "no": call.no,
                    "function": call.name,
                    "arg": arg_name,
                    "bytes": len(raw),
                    "path": str(path),
                    "head_hex": raw[:32].hex(),
                }
            )
            if len(written) >= limit:
                return {
                    "blobs": written,
                    "count": len(written),
                    "truncated": True,
                    "calls_scanned": scanned,
                }
    # Reaching the walk cap without filling the blob quota means later blobs in
    # the requested range were never even inspected -- say so.
    hit_cap = scanned >= walk_cap
    result: dict[str, Any] = {
        "blobs": written,
        "count": len(written),
        "truncated": hit_cap,
        "calls_scanned": scanned,
    }
    if hit_cap:
        result["note"] = (
            f"stopped after scanning {scanned} calls (walk cap) with only "
            f"{len(written)} of {limit} requested blobs; narrow `calls` to the "
            "range that carries the blobs, or raise `limit`"
        )
    return result
