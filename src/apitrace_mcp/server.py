"""MCP server exposing apitrace for reverse-engineering older games.

Design notes that matter when reading this file:

* Traces are enormous. Every tool that reads one is bounded, and says so when it
  truncates. `calls=` (apitrace's callset syntax) is the primary way to keep a
  query cheap: "0-5000", "1000-2000/draw", "frame", "@list.txt".
 * Tools return native structured results. Failures use MCP's protocol-level
   tool error channel with a concrete explanation wherever possible.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import json
import os
import shlex
import sys
import tempfile
import threading
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import mcp.server.fastmcp.server as fastmcp_server
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from . import analysis, matrices as mx, wrappers
from .config import ENV, VALID_APIS, WRAPPER_FOR_API, trace_dir
from .pe import PEError, graphics_apis, read_pe
from .pickleparse import call_to_dict, iter_calls
from .runner import CommandError, run
from .sessions import MANAGER


def _repair_fastmcp_settings_forward_refs() -> None:
    """Work around MCP 1.29's unresolved Settings.lifespan ForwardRef."""
    try:
        fastmcp_server.Settings.model_rebuild(
            _types_namespace=vars(fastmcp_server)
        )
    except (AttributeError, NameError, TypeError):
        # Older/newer MCP releases may not expose this model or may already have
        # resolved it. In either case there is nothing to repair.
        pass


_repair_fastmcp_settings_forward_refs()

mcp = FastMCP("apitrace")

READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
WRITES_FILES = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)
STARTS_PROCESS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)
DESTRUCTIVE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=False,
)

CALLSET_HELP = (
    "apitrace callset syntax: '42' one call, '0,2,4' a set, '100-2000' a range, "
    "'0-1000/2' every 2nd, '0-1000/draw' draw calls only, '0-1000/fbo' render-target "
    "changes, 'frame' the end-of-frame calls, '@file.txt' read from a file."
)

MAX_CONCURRENT_FOREGROUND_TOOLS = 2
MAX_TOOL_OUTPUT_BYTES = 2_000_000
MAX_STRUCTURED_CALLS = 5_000
MAX_STRUCTURED_LIST_ITEMS = 1_024
# How long a slotted tool waits for a free work slot before giving up with a
# clear busy error rather than blocking the caller indefinitely.
WORK_SLOT_TIMEOUT_SECONDS = 30.0
_FOREGROUND_WORK_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_FOREGROUND_TOOLS)

DUMP_CALLS_DESCRIPTION = (
    "Human-readable call dump, exactly as `apitrace dump` writes it. Good for "
    "eyeballing a stretch of a frame; use get_calls for numeric arguments. Calls: "
    + CALLSET_HELP
)
GET_CALLS_DESCRIPTION = (
    "Structured calls with real argument values via `apitrace pickle`. Long arrays "
    "and blobs are summarized. Calls: "
    + CALLSET_HELP
)


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _run_with_work_slot(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    if not _FOREGROUND_WORK_SLOTS.acquire(timeout=WORK_SLOT_TIMEOUT_SECONDS):
        raise RuntimeError(
            f"server busy: all {MAX_CONCURRENT_FOREGROUND_TOOLS} concurrent work "
            f"slots are in use (waited {WORK_SLOT_TIMEOUT_SECONDS:g}s); retry once "
            "a long job (trim/gltrim/sed/replay) finishes -- trace_status and "
            "list_sessions stay available meanwhile"
        )
    try:
        return fn(*args, **kwargs)
    finally:
        _FOREGROUND_WORK_SLOTS.release()


def _validate_output_limit(max_bytes: int) -> None:
    if not 0 <= max_bytes <= MAX_TOOL_OUTPUT_BYTES:
        raise ValueError(
            f"max_bytes must be between 0 and {MAX_TOOL_OUTPUT_BYTES}"
        )


def _run_without_work_slot(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    return fn(*args, **kwargs)


def guard(
    fn: Callable[..., Any] | None = None, *, work_slot: bool = True
) -> Callable[..., Any]:
    """Run blocking tools off-loop and surface failures as MCP tool errors.

    work_slot=False exempts cheap status/listing tools (no subprocess work)
    from the bounded work slots so polling stays responsive while long batch
    jobs hold every slot.
    """
    if fn is None:
        return lambda actual: guard(actual, work_slot=work_slot)
    runner = _run_with_work_slot if work_slot else _run_without_work_slot

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(*args, **kwargs)
            else:
                result = await asyncio.to_thread(runner, fn, args, kwargs)
        except CommandError as exc:
            payload = {
                "error": str(exc),
                "command": " ".join(exc.cmd),
                "exit_code": exc.returncode,
            }
            raise ToolError(_json(payload)) from exc
        except (FileNotFoundError, PEError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - converted to protocol-level tool error
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        if isinstance(result, dict):
            return result
        return {"text": str(result)}

    return wrapper


def _resolve_trace(trace: str) -> Path:
    path = Path(trace)
    if not path.is_absolute():
        candidate = trace_dir() / trace
        if candidate.is_file():
            return candidate.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"trace not found: {trace} (also looked in {trace_dir()})"
        )
    return path.resolve()


def _resolved_file(value: str, *, label: str = "file") -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _resolved_output(value: str | Path, default: Path) -> Path:
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = trace_dir() / path
    return path.resolve()


def _resolve_callset(calls: str) -> str:
    """Resolve @callset files without changing ordinary apitrace syntax."""
    if not calls.startswith("@"):
        return calls
    path = _resolved_file(calls[1:], label="callset file")
    return f"@{path}"


def _coerce_argv(value: list[str] | str | None, *, label: str) -> list[str]:
    """Accept native argv or safely parse the legacy string form without a shell."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            parsed = shlex.split(value, posix=False)
        except ValueError as exc:
            raise ValueError(f"invalid {label}: {exc}") from exc
        args = [
            token[1:-1]
            if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"')
            else token
            for token in parsed
        ]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        args = list(value)
    else:
        raise TypeError(f"{label} must be a list of strings or a command-line string")
    if any("\0" in item for item in args):
        raise ValueError(f"{label} cannot contain NUL characters")
    return args


def _coerce_string_list(
    value: list[str] | str | None, *, label: str
) -> list[str]:
    if value is None or value == "":
        return []
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise TypeError(f"{label} must be a string or list of strings")
    if any(not item or "\0" in item for item in values):
        raise ValueError(f"{label} entries must be non-empty and cannot contain NUL")
    return list(values)


def _reject_options(args: list[str], forbidden: tuple[str, ...], *, label: str) -> None:
    for arg in args:
        for option in forbidden:
            if arg == option or arg.startswith(option + "="):
                raise ValueError(f"{label} cannot override managed option {option}")
            if len(option) == 2 and option.startswith("-") and arg.startswith(option):
                raise ValueError(f"{label} cannot override managed option {option}")


def _resolved_image_prefix(value: str, *, label: str) -> Path:
    prefix = Path(value).expanduser().resolve()
    if not prefix.is_dir() and not prefix.parent.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {prefix.parent}")
    if not _images_for_prefix(prefix):
        raise FileNotFoundError(f"no PNG/BMP images match {label}: {prefix}")
    return prefix


def _images_for_prefix(prefix: Path) -> list[Path]:
    base = prefix if prefix.is_dir() else prefix.parent
    prefix_text = str(prefix)
    images: list[Path] = []
    if not base.is_dir():
        return images
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix not in (".png", ".bmp"):
            continue
        secondary_suffix = Path(path.stem).suffix
        if secondary_suffix in (".diff", ".thumb"):
            continue
        if str(path).startswith(prefix_text):
            images.append(path.resolve())
    return sorted(images)


def _image_sidecars(reference: Path, source: Path) -> list[Path]:
    """Potential files snapdiff.py creates for these two prefixes."""
    ref_text, src_text = str(reference), str(source)
    ref_suffixes = {str(path)[len(ref_text) :] for path in _images_for_prefix(reference)}
    src_suffixes = {str(path)[len(src_text) :] for path in _images_for_prefix(source)}
    targets: set[Path] = set()
    for suffix in ref_suffixes | src_suffixes:
        ref_image = Path(ref_text + suffix)
        src_image = Path(src_text + suffix)
        for image in (ref_image, src_image):
            targets.add(image.with_name(f"{image.stem}.thumb{image.suffix}"))
        delta = src_image.with_name(f"{src_image.stem}.diff{src_image.suffix}")
        targets.add(delta)
        targets.add(delta.with_name(f"{delta.stem}.thumb{delta.suffix}"))
    return sorted(target.resolve() for target in targets)


def _complete_html(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 4096))
        return b"</html>" in stream.read().lower()


_OUTPUT_RESERVATION_LOCK = threading.Lock()
_RESERVED_OUTPUT_FILES: set[Path] = set()
_RESERVED_OUTPUT_PREFIXES: set[Path] = set()


def _unique_file(path: Path) -> Path:
    """Reserve a UUID-backed output path without overwriting an existing file."""
    path = path.resolve()
    with _OUTPUT_RESERVATION_LOCK:
        for _ in range(1000):
            token = uuid.uuid4().hex[:12]
            candidate = path.with_name(f"{path.stem}-{token}{path.suffix}")
            if not candidate.exists() and candidate not in _RESERVED_OUTPUT_FILES:
                _RESERVED_OUTPUT_FILES.add(candidate)
                return candidate
    raise RuntimeError(f"could not allocate a unique output path beside {path}")


def _names_with_prefix(directory: Path, name_prefix: str) -> list[Path]:
    """Literal startswith scan; glob would misread brackets in trace names."""
    if not directory.is_dir():
        return []
    return [
        entry for entry in directory.iterdir() if entry.name.startswith(name_prefix)
    ]


def _unique_prefix(prefix: Path) -> Path:
    """Reserve a UUID-backed prefix for commands that create many files."""
    prefix = prefix.resolve()
    with _OUTPUT_RESERVATION_LOCK:
        for _ in range(1000):
            candidate = prefix.with_name(prefix.name + "-" + uuid.uuid4().hex[:12])
            if (
                candidate not in _RESERVED_OUTPUT_PREFIXES
                and not _names_with_prefix(candidate.parent, candidate.name)
            ):
                _RESERVED_OUTPUT_PREFIXES.add(candidate)
                return candidate
    raise RuntimeError(f"could not allocate a unique output prefix beside {prefix}")


def _reserve_output_files(paths: list[Path], *, label: str) -> list[Path]:
    """Atomically reserve a set of auxiliary outputs within this MCP process."""
    resolved = [path.resolve() for path in paths]
    with _OUTPUT_RESERVATION_LOCK:
        collisions = [
            path
            for path in resolved
            if path.exists() or path in _RESERVED_OUTPUT_FILES
        ]
        if collisions:
            raise FileExistsError(
                f"{label} would overwrite existing or reserved files: "
                f"{[str(path) for path in collisions[:20]]}"
            )
        _RESERVED_OUTPUT_FILES.update(resolved)
    return resolved


def _release_output_files(paths: list[Path]) -> None:
    """Drop reservations taken by _reserve_output_files.

    Safe after success too: created files exist on disk, so the exists() check
    in _reserve_output_files still refuses overwrites.
    """
    with _OUTPUT_RESERVATION_LOCK:
        _RESERVED_OUTPUT_FILES.difference_update(paths)


def _acquire_diff_images_lock(reference: Path, source: Path) -> tuple[Path, bytes]:
    """Cross-process exclusion for snapdiff's deterministic sidecar names."""
    identity = f"{reference}\0{source}".encode("utf-8", errors="surrogatepass")
    digest = hashlib.sha256(identity).hexdigest()[:20]
    lock = source.parent / f".apitrace-mcp-diff-images-{digest}.lock"
    token = f"pid={os.getpid()} uuid={uuid.uuid4()}\n".encode("ascii")
    try:
        descriptor = os.open(
            lock,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "another apitrace-mcp process is comparing these image prefixes; "
            f"if no comparison is running, remove stale lock {lock}"
        ) from exc
    try:
        os.write(descriptor, token)
    finally:
        os.close(descriptor)
    return lock, token


def _release_diff_images_lock(lock: Path, token: bytes) -> None:
    """Remove only the lock file created by this invocation."""
    try:
        if lock.read_bytes() == token:
            lock.unlink()
    except FileNotFoundError:
        pass


# Import-table order is meaningless for API choice: ddraw.dll often precedes
# d3d9.dll on D3D8/D3D9-era games because DirectDraw is only a legacy/video
# path. Rank supported APIs by how useful they are to trace instead.
_API_PRIORITY = {"d3d9": 0, "d3d8": 1, "dxgi": 2, "gl": 3, "d3d7": 4}


def _trace_candidates(apis: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Supported APIs ranked best-first, excluding DXGI on a D3D12 target."""
    has_d3d12 = any(api.get("dll") == "d3d12.dll" for api in apis)
    found = [
        api
        for api in apis
        if api.get("supported")
        and not (has_d3d12 and api.get("api") == "dxgi")
    ]
    return sorted(
        found, key=lambda api: _API_PRIORITY.get(str(api.get("api")), 99)
    )


def _detect_trace_api(pe: Any) -> str:
    apis = graphics_apis(pe)
    found = _trace_candidates(apis)
    if found:
        return str(found[0]["api"])
    if any(api.get("dll") == "d3d12.dll" for api in apis):
        raise RuntimeError(
            "This target imports D3D12, which apitrace cannot trace. Use RenderDoc."
        )
    raise RuntimeError(
        "Could not auto-detect a supported graphics API from the import table. "
        "Pass api= explicitly, or run detect_target for the details."
    )


# ===========================================================================
# environment
# ===========================================================================


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard(work_slot=False)
def apitrace_status() -> dict[str, Any]:
    """Preflight: which apitrace builds are installed, where traces go, what is running.

    Call this before anything else. Unlike the GUI-backed RE servers, apitrace
    needs no host application running -- but it does need the build whose
    bitness matches the game you intend to trace.
    """
    ENV.refresh()
    roots = ENV.roots
    out: dict[str, Any] = {
        "roots": {f"win{bits}": root.as_dict() for bits, root in sorted(roots.items())},
        "trace_dir": str(trace_dir()),
        "sessions": MANAGER.list(trace_dir() / "logs"),
    }
    if not roots:
        out["error"] = (
            "No apitrace install found. Unpack a release from "
            "https://github.com/apitrace/apitrace/releases and set "
            "APITRACE_MCP_ROOT_WIN64 / APITRACE_MCP_ROOT_WIN32."
        )
        return out

    warnings = []
    if 32 not in roots:
        warnings.append(
            "No 32-bit apitrace build installed. Most pre-2012 games are 32-bit and "
            "CANNOT be traced with the win64 build -- the wrapper has to load into "
            "the target process. Download apitrace-<ver>-win32.7z and set "
            "APITRACE_MCP_ROOT_WIN32."
        )
    if 64 not in roots:
        warnings.append("No 64-bit apitrace build installed.")
    if warnings:
        out["warnings"] = warnings

    for bits, root in roots.items():
        try:
            stdout, _, _ = run([str(root.apitrace_exe), "version"], timeout=20)
            out["roots"][f"win{bits}"]["version_reported"] = stdout.strip()
        except Exception as exc:  # noqa: BLE001
            out["roots"][f"win{bits}"]["version_reported"] = f"<failed: {exc}>"

    out["traces_on_disk"] = _list_traces()
    return out


def _list_traces() -> list[dict]:
    found = []
    for path in sorted(trace_dir().glob("*.trace")):
        stat = path.stat()
        found.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
            }
        )
    return found


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard(work_slot=False)
def list_traces() -> dict[str, Any]:
    """List trace files in the trace directory."""
    return {"trace_dir": str(trace_dir().resolve()), "traces": _list_traces()}


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def detect_target(game_exe: str) -> dict[str, Any]:
    """Inspect a game executable: bitness, graphics API, and how to trace it.

    Reads the PE import table, so it answers "is this D3D8 or D3D9, and is it
    32-bit" without launching anything. Delay-loaded imports are included --
    older games often delay-load d3d9 behind a video-detection step.
    """
    exe = _resolved_file(game_exe, label="executable")
    info = read_pe(str(exe))
    apis = graphics_apis(info)

    root_available = info.bits in ENV.roots
    result: dict[str, Any] = {
        "exe": str(exe),
        "machine": info.machine,
        "bits": info.bits,
        "graphics_imports": apis,
        "all_imports": info.all_imports,
        "apitrace_build_needed": f"win{info.bits}",
        "apitrace_build_available": root_available,
    }

    has_d3d12 = any(a["dll"] == "d3d12.dll" for a in apis)
    supported = _trace_candidates(apis)
    unsupported = [a for a in apis if not a["supported"]]

    if not apis:
        result["recommendation"] = (
            "No graphics DLL in the import table. The renderer is probably loaded at "
            "runtime with LoadLibrary (common in engines with a selectable renderer), "
            "or this exe is a launcher that starts the real game. Check for other "
            "executables in the folder, or run the game and look at its loaded modules."
        )
    else:
        best = supported[0] if supported else None
        if best:
            result["recommended_api"] = best["api"]
            if len(supported) > 1:
                result["runner_up_apis"] = [api["api"] for api in supported[1:]]
                result["api_ranking_note"] = (
                    "Multiple supported APIs imported; ranked by usefulness, not "
                    "import-table order. ddraw (d3d7) is a common legacy/video "
                    "path on D3D8/D3D9-era games -- try the recommended API first."
                )
            if best["api"] == "dxgi":
                result["recommended_wrapper"] = []
                result["recommendation"] = (
                    "Use trace_launch(api='dxgi'). D3D10/D3D11 must use injection; "
                    "there is no supported manual DLL-wrapper fallback."
                )
            else:
                result["recommended_wrapper"] = WRAPPER_FOR_API.get(best["api"], [])
                result["recommendation"] = (
                    f"trace_launch(game_exe=..., api='{best['api']}'). If that yields an "
                    "empty trace (launcher indirection, DRM re-launch), fall back to "
                    f"install_wrapper(api='{best['api']}') and start the game yourself."
                )
        elif has_d3d12:
            result["recommendation"] = (
                "D3D12 is not supported by apitrace. Use RenderDoc; do not install an "
                "apitrace DXGI wrapper for this target."
            )
    if unsupported:
        result["unsupported"] = [
            f"{a['dll']} -- apitrace does not support this API "
            "(use RenderDoc for D3D12/Vulkan)"
            for a in unsupported
        ]
    if not root_available:
        result["blocker"] = (
            f"This is a {info.bits}-bit game and no win{info.bits} apitrace build is "
            "installed. See apitrace_status for how to fix that."
        )
    return result


# ===========================================================================
# capture
# ===========================================================================


@mcp.tool(annotations=STARTS_PROCESS, structured_output=True)
@guard
def trace_launch(
    game_exe: str,
    api: str = "",
    output: str = "",
    args: list[str] | str | None = None,
    method: str = "iat",
    working_dir: str = "",
) -> dict[str, Any]:
    """Launch a game under apitrace and start capturing. Returns a session id.

    This runs the game -- it does not block. Let the user play the part they want
    captured, then call trace_stop (ideally after they quit the game normally, so
    the trace is flushed cleanly).

    api: gl, d3d7, d3d8, d3d9, or dxgi (D3D10/11). Empty auto-detects from the PE.
    method: 'iat' (default) or 'mhook'. Try mhook when a game loads the graphics
    DLL dynamically and IAT patching misses it.
    """
    exe = _resolved_file(game_exe, label="game executable")

    pe = read_pe(str(exe))
    root = ENV.root_for_bits(pe.bits)

    if not api:
        api = _detect_trace_api(pe)
    if api not in VALID_APIS:
        raise ValueError(f"api must be one of {list(VALID_APIS)}, got {api!r}")

    out_path = _resolved_output(output, trace_dir() / f"{exe.stem}.trace")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = _unique_file(out_path)

    cmd = [str(root.apitrace_exe), "trace", f"--api={api}", f"--output={out_path}"]
    if method == "mhook":
        cmd.append("--mhook")
    elif method != "iat":
        raise ValueError("method must be 'iat' or 'mhook'")
    cmd.append(str(exe))
    cmd.extend(_coerce_argv(args, label="args"))

    cwd = Path(working_dir).expanduser().resolve() if working_dir else exe.parent
    if not cwd.is_dir():
        raise FileNotFoundError(f"working directory not found: {cwd}")

    session = MANAGER.start(
        "trace",
        cmd,
        log_dir=trace_dir() / "logs",
        cwd=cwd,
        trace_path=out_path,
    )
    time.sleep(1.5)  # long enough to catch an immediate failure to launch
    status = session.status()
    status["api"] = api
    status["apitrace_build"] = f"win{root.bits}"
    status["next_steps"] = [
        "The game window should be up; play the scene you want captured.",
        "Tracing an old game costs a lot of frame time -- keep captures to a few "
        "seconds unless you need the length.",
        f"When done, quit the game normally, then call trace_stop('{session.id}').",
    ]
    return status


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard(work_slot=False)
def trace_status(session: str) -> dict[str, Any]:
    """Poll a trace or replay session: still running, trace size so far, log tail."""
    MANAGER.ensure_loaded(trace_dir() / "logs")
    return MANAGER.get(session).status()


@mcp.tool(annotations=DESTRUCTIVE, structured_output=True)
@guard(work_slot=False)
def trace_stop(session: str, force: bool = False) -> dict[str, Any]:
    """Stop a session and report the resulting trace.

    Prefer quitting the game in-game first: a forced kill can lose the tail of
    the trace because apitrace never gets to flush.
    """
    MANAGER.ensure_loaded(trace_dir() / "logs")
    return MANAGER.stop(session, force=force)


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard(work_slot=False)
def list_sessions() -> dict[str, Any]:
    """All trace/replay sessions, including any still running from a previous server run."""
    return {"sessions": MANAGER.list(trace_dir() / "logs")}


@mcp.tool(annotations=DESTRUCTIVE, structured_output=True)
@guard
def install_wrapper(
    game_exe: str, api: str = "", trace_file: str = "", force: bool = False
) -> dict[str, Any]:
    """Drop apitrace's wrapper DLL next to a game exe (the manual tracing route).

    Use this when trace_launch produces nothing -- typically because the exe you
    launched is a launcher that spawns the real game, or the game re-launches
    itself for DRM. The wrapper then loads whenever the game starts, however it
    was started (including from Steam).

    Any existing DLL of the same name -- ENB, ReShade, dgVoodoo, DXVK all use
    these filenames -- is backed up and restored by uninstall_wrapper. Always
    uninstall when you are done; a left-behind wrapper keeps tracing every run.
    """
    exe = _resolved_file(game_exe, label="game executable")
    pe = read_pe(str(exe))
    root = ENV.root_for_bits(pe.bits)

    if not api:
        api = _detect_trace_api(pe)
    if api == "dxgi":
        raise ValueError(
            "D3D10/D3D11 tracing requires trace_launch(api='dxgi'); a manual DXGI "
            "wrapper install is not supported"
        )

    requested = _resolved_output(
        trace_file, trace_dir() / f"{exe.stem}.trace"
    )
    if trace_file:
        if requested.exists():
            raise FileExistsError(
                f"refusing to overwrite existing wrapper trace: {requested}"
            )
        target = str(requested)
    else:
        target = str(_unique_file(requested))
    result = wrappers.install(root, exe, api, trace_file=target, force=force)
    result["api"] = api
    result["apitrace_build"] = f"win{root.bits}"
    result["trace_file"] = target
    return result


@mcp.tool(annotations=DESTRUCTIVE, structured_output=True)
@guard
def uninstall_wrapper(game_exe_or_dir: str) -> dict[str, Any]:
    """Remove wrapper DLLs installed by install_wrapper and restore any backups."""
    target = Path(game_exe_or_dir).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"game executable/directory not found: {target}")
    return wrappers.uninstall(target)


# ===========================================================================
# inspection
# ===========================================================================


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def trace_info(trace: str, per_frame: bool = False) -> dict[str, Any]:
    """Summary of a trace file: API, call count, frame count (JSON from apitrace info)."""
    path = _resolve_trace(trace)
    root = ENV.any_root()
    info = analysis.trace_info(root, path, dump_frames=per_frame)
    payload: dict[str, Any] = {
        "trace": str(path),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "info": info,
    }
    if per_frame and isinstance(info, dict):
        frames = info.get("Frames")
        if frames is None:
            frames = info.get("frames")
        if isinstance(frames, list):
            payload["frame_count"] = len(frames)
            payload["info"] = {
                k: v for k, v in info.items() if k not in ("Frames", "frames")
            }
            payload["frames_head"] = frames[:50]
            payload["frames_tail"] = frames[-10:] if len(frames) > 50 else []
    return payload


@mcp.tool(
    annotations=READ_ONLY,
    structured_output=True,
    description=DUMP_CALLS_DESCRIPTION,
)
@guard
def dump_calls(
    trace: str,
    calls: str = "",
    grep: str = "",
    max_bytes: int = 40000,
    timeout: int = 300,
) -> dict[str, Any]:
    """Human-readable call dump, exactly as `apitrace dump` writes it.

    Good for eyeballing a stretch of a frame. For anything you want to compute
    with, use get_calls instead -- it returns real argument values. To write
    binary arguments (shader bytecode, buffers) to files, use extract_blobs.

    calls: apitrace callset syntax, such as '42', '100-2000',
    '0-1000/draw', 'frame', or '@file.txt'.
    grep: regex on function names, e.g. 'SetTransform|SetVertexShaderConstant'.
    """
    _validate_output_limit(max_bytes)
    path = _resolve_trace(trace)
    root = ENV.any_root()
    text, truncated = analysis.dump_text(
        root,
        path,
        calls=_resolve_callset(calls) or None,
        grep=grep or None,
        max_bytes=max_bytes,
        blobs=False,
        timeout=timeout,
    )
    return {
        "trace": str(path),
        "calls": calls or None,
        "grep": grep or None,
        "text": text,
        "truncated": truncated,
        "hint": "narrow calls= or raise max_bytes" if truncated else None,
    }


@mcp.tool(
    annotations=READ_ONLY,
    structured_output=True,
    description=GET_CALLS_DESCRIPTION,
)
@guard
def get_calls(
    trace: str,
    calls: str = "",
    limit: int = 200,
    name_filter: str = "",
    max_list: int = 32,
) -> dict[str, Any]:
    """Structured calls with real argument values, via `apitrace pickle`.

    This is the tool to use when you need numbers -- matrix contents, constant
    registers, primitive counts -- rather than text. Long arrays and blobs are
    summarised rather than inlined.

    calls: apitrace callset syntax, such as '42', '100-2000',
    '0-1000/draw', 'frame', or '@file.txt'.
    name_filter: case-insensitive substring on the function name, applied after
    apitrace's own filtering.
    """
    if not 0 <= limit <= MAX_STRUCTURED_CALLS:
        raise ValueError(
            f"limit must be between 0 and {MAX_STRUCTURED_CALLS}"
        )
    if not 0 <= max_list <= MAX_STRUCTURED_LIST_ITEMS:
        raise ValueError(
            f"max_list must be between 0 and {MAX_STRUCTURED_LIST_ITEMS}"
        )
    path = _resolve_trace(trace)
    root = ENV.any_root()
    if limit == 0:
        return {
            "trace": str(path),
            "calls_scanned": 0,
            "returned": 0,
            "truncated": False,
            "calls": [],
        }
    needle = name_filter.lower()
    out: list[dict] = []
    scanned = 0
    # When filtering by name we may need to walk a lot of calls to find `limit`
    # matches; cap the walk so a bad filter cannot run away with the trace.
    walk_cap = limit if not needle else max(limit * 500, 20000)
    for call in iter_calls(
        root, path, calls=_resolve_callset(calls) or None, limit=walk_cap
    ):
        scanned += 1
        if needle and needle not in call.name.lower():
            continue
        out.append(call_to_dict(call, max_list=max_list))
        if len(out) >= limit:
            break
    return {
        "trace": str(path),
        "calls_scanned": scanned,
        "returned": len(out),
        "truncated": len(out) >= limit,
        "calls": out,
    }


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def search_calls(
    trace: str, pattern: str, calls: str = "", max_hits: int = 200, timeout: int = 600
) -> dict[str, Any]:
    """Regex search over function names across the trace; returns counts and call numbers.

    The cheapest way to find out whether a game uses fixed-function transforms or
    shader constants: search 'SetTransform|SetVertexShaderConstant'.
    """
    path = _resolve_trace(trace)
    root = ENV.any_root()
    return analysis.search_calls(
        root,
        path,
        pattern,
        calls=_resolve_callset(calls) or None,
        max_hits=max_hits,
        timeout=timeout,
    )


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def call_histogram(
    trace: str, calls: str = "", limit: int = 200000, top: int = 60
) -> dict[str, Any]:
    """Function-frequency profile over a call range -- the fastest way to see what a game does.

    Scanning the whole of a large trace is slow; `limit` caps how many calls are
    read (default 200k).
    """
    path = _resolve_trace(trace)
    root = ENV.any_root()
    return analysis.histogram(
        root,
        path,
        calls=_resolve_callset(calls) or None,
        limit=limit or None,
        top=top,
    )


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def frame_summary(trace: str, calls: str, limit: int = 100000) -> dict[str, Any]:
    """Break a frame down: draw calls, render-target switches, shader binds.

    Pass a callset covering one frame, e.g. calls='120000-125000'. Use
    trace_info(per_frame=True) or list_frames to find the boundaries.
    """
    path = _resolve_trace(trace)
    root = ENV.any_root()
    return analysis.frame_summary(
        root, path, calls=_resolve_callset(calls), limit=limit or None
    )


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def list_frames(trace: str, start: int = 0, count: int = 40) -> dict[str, Any]:
    """Frame boundaries as call numbers, so you can scope other queries to one frame."""
    path = _resolve_trace(trace)
    if start < 0 or count < 0:
        raise ValueError("start and count must be non-negative")
    info = analysis.trace_info(ENV.any_root(), path, dump_frames=True)
    raw_frames = info.get("Frames") if isinstance(info, dict) else None
    if raw_frames is None and isinstance(info, dict):
        raw_frames = info.get("frames")
    if not isinstance(raw_frames, list):
        raise RuntimeError("apitrace info did not return a Frames array")
    total = len(raw_frames)
    selected = raw_frames[start : start + count] if count else []
    frames: list[dict[str, Any]] = []
    for index, entry in enumerate(selected, start=start):
        if not isinstance(entry, dict):
            continue
        first = entry.get("FirstCallId", entry.get("first_call_id"))
        last = entry.get("LastCallId", entry.get("last_call_id"))
        if not isinstance(first, int) or not isinstance(last, int):
            continue
        frames.append(
            {
                "frame": index,
                "calls": f"{first}-{last}",
                "first_call_no": first,
                "end_call_no": last,
                "call_count": entry.get("TotalCalls", last - first + 1),
                "size_bytes": entry.get("SizeInBytes"),
            }
        )
    next_start = start + len(frames)
    has_more = next_start < total
    return {
        "trace": str(path),
        "start": start,
        "requested_count": count,
        "total_frames": total,
        "frames": frames,
        "has_more": has_more,
        "next_start": next_start if count and has_more else None,
        "note": "pass one of these calls strings to frame_summary or find_matrices",
    }


# ===========================================================================
# reverse-engineering analysis
# ===========================================================================


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def find_matrices(
    trace: str,
    calls: str = "",
    limit: int = 200000,
    min_confidence: float = 0.5,
    kinds: str = "projection,viewproj,rigid,ortho",
    scan_constants: bool = True,
    scan_buffers: bool = True,
) -> dict[str, Any]:
    """Find and classify every 4x4 matrix the game sends to the driver.

    This is the core VR/camera-hacking tool. It covers:

    * fixed-function transforms -- SetTransform(D3DTS_VIEW/PROJECTION/WORLD),
      glLoadMatrixf/glMultMatrixf, glFrustum/glOrtho;
    * shader constants -- SetVertexShaderConstantF and friends, tracked through a
      shadow register file so matrices uploaded piecemeal still resolve, and
      reported as the register range that holds them (e.g. vs_c[8..11]);
    * uniform/constant buffer uploads -- glBufferData, UpdateSubresource and the
      memcpy calls apitrace records for writes to mapped memory, reported as a
      byte offset into the buffer (set scan_buffers=false to skip);
    * both storage orders, since D3D9 shader constants usually hold the transpose.

    Results are grouped by slot. Read them as:
      kind=projection    -> the game's FOV, near and far planes
      kind=rigid + changes_per_frame -> camera candidate; confirm with track_camera
      kind=viewproj in vs_c[...]     -> the register range a VR patch rewrites
      kind=viewproj at a buffer offset -> the bytes a VR patch rewrites
      kind=ortho                     -> HUD/UI or shadow pass

    Scope with `calls` (a single frame is usually enough and far faster).
    """
    path = _resolve_trace(trace)
    root = ENV.any_root()
    kind_tuple = tuple(k.strip() for k in kinds.split(",") if k.strip())
    return analysis.scan_matrices(
        root,
        path,
        calls=_resolve_callset(calls) or None,
        limit=limit or None,
        min_confidence=min_confidence,
        include_kinds=kind_tuple,
        scan_constants=scan_constants,
        scan_buffers=scan_buffers,
    )


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def decode_matrix(values: str, label: str = "") -> dict[str, Any]:
    """Decode 16 floats as a 4x4 matrix: FOV, near/far, handedness, camera position.

    Accepts whitespace-, comma- or newline-separated numbers -- paste them
    straight out of a memory dump, a Cheat Engine watch, or a shader constant.
    Both storage orders are tried, so you do not need to know whether the source
    was row- or column-major.
    """
    raw = [
        v
        for v in values.replace(",", " ").replace("\n", " ").split()
        if v not in ("", "[", "]")
    ]
    nums: list[float] = []
    for token in raw:
        try:
            nums.append(float(token.strip("[](){}")))
        except ValueError:
            continue
    if len(nums) != 16:
        raise ValueError(f"expected 16 numbers, parsed {len(nums)}: {nums}")
    decode = mx.classify(nums)
    return {
        "label": label or None,
        "rows_as_given": mx.format_matrix(nums),
        "is_identity": mx.is_identity(nums),
        "decode": decode.to_dict() if decode else None,
        "note": None
        if decode
        else "no known matrix form matched -- it may be a world/model matrix "
        "with scaling, or not a matrix at all",
    }


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def track_camera(
    trace: str, calls: str = "", source: str = "", limit: int = 200000, max_frames: int = 200
) -> dict[str, Any]:
    """Follow the camera across frames: world position, forward vector and FOV.

    Picks the most view-like matrix slot automatically, or pass `source` with a
    slot name from find_matrices. If the eye position changes as the player moved
    during the capture, you have found the real view matrix -- that is the
    confirmation step before taking the offset into Ghidra or ReGenny.
    """
    path = _resolve_trace(trace)
    root = ENV.any_root()
    return analysis.track_camera(
        root,
        path,
        calls=_resolve_callset(calls) or None,
        source=source or None,
        limit=limit or None,
        max_frames=max_frames,
    )


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def list_shaders(
    trace: str, calls: str = "", limit: int = 200000, max_shaders: int = 100
) -> dict[str, Any]:
    """List shader creation calls with a source preview or bytecode size.

    Use extract_blobs to write D3D shader bytecode out for disassembly.
    """
    path = _resolve_trace(trace)
    root = ENV.any_root()
    return analysis.list_shaders(
        root,
        path,
        calls=_resolve_callset(calls) or None,
        limit=limit or None,
        max_shaders=max_shaders,
    )


@mcp.tool(annotations=WRITES_FILES, structured_output=True)
@guard
def extract_blobs(
    trace: str, calls: str, out_dir: str = "", limit: int = 50
) -> dict[str, Any]:
    """Write binary arguments (shader bytecode, vertex/index/constant buffers) to files.

    Scope tightly with `calls` -- buffer uploads are large and there are many.
    The scan walks at most max(limit*500, 20000) calls of the callset, so blobs
    beyond that walk cap are not found; narrow `calls` if the result looks short.
    """
    path = _resolve_trace(trace)
    root = ENV.any_root()
    target = _unique_file(
        _resolved_output(out_dir, trace_dir() / f"{path.stem}-blobs")
    )
    return analysis.extract_blobs(
        root,
        path,
        calls=_resolve_callset(calls),
        out_dir=target,
        limit=limit,
    )


# ===========================================================================
# replay
# ===========================================================================


def _api_of(trace: Path) -> str:
    """Which retracer a trace needs. apitrace info reports e.g. 'OpenGL + GLX/WGL/CGL'."""
    info = analysis.trace_info(ENV.any_root(), trace)
    api = str(info.get("API") or info.get("api") or "") if isinstance(info, dict) else ""
    low = api.lower()
    if "opengl" in low or low.startswith("gl") or "egl" in low:
        return "gl"
    if "d3d12" in low or "direct3d 12" in low or "vulkan" in low:
        raise RuntimeError(f"trace API is not supported by apitrace replay: {api!r}")
    if "d3d" in low or "direct3d" in low or "directdraw" in low or "dxgi" in low:
        return "d3d"
    raise RuntimeError(f"could not determine retracer from apitrace API metadata: {api!r}")


def _retrace_kind(api: str, trace: Path) -> str:
    if not api:
        return _api_of(trace)
    low = api.lower()
    if low in ("gl", "opengl", "egl"):
        return "gl"
    if low in ("d3d", "d3d7", "d3d8", "d3d9", "dxgi", "direct3d"):
        return "d3d"
    raise ValueError("api must identify an OpenGL or Direct3D retracer")


MAX_STATE_JSON_BYTES = 32 * 1024 * 1024
MAX_STATE_STRING_CHARS = 8_192
MAX_STATE_SEQUENCE_ITEMS = 256


def _summarize_state_payload(value: Any, *, depth: int = 0) -> Any:
    """Recursively compact framebuffer/image payloads after complete JSON parsing."""
    if depth > 24:
        return {"summary": "maximum nesting depth reached"}
    if isinstance(value, str) and len(value) > MAX_STATE_STRING_CHARS:
        raw = value.encode("utf-8", errors="replace")
        return {
            "summary": "large string payload omitted",
            "chars": len(value),
            "utf8_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "head": value[:256],
        }
    if isinstance(value, list):
        if len(value) > MAX_STATE_SEQUENCE_ITEMS:
            return {
                "summary": "large array payload omitted",
                "item_count": len(value),
                "head": [
                    _summarize_state_payload(item, depth=depth + 1)
                    for item in value[:16]
                ],
                "tail": [
                    _summarize_state_payload(item, depth=depth + 1)
                    for item in value[-4:]
                ],
            }
        return [_summarize_state_payload(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _summarize_state_payload(item, depth=depth + 1)
            for key, item in value.items()
        }
    return value


def _capture_state_json(
    path: Path, call: int, api: str, timeout: int
) -> tuple[str, Any, str]:
    """Return complete validated state JSON, parsed state, and retracer kind."""
    if call < 0:
        raise ValueError("call must be non-negative")
    root = ENV.any_root()
    kind = _retrace_kind(api, path)
    exe = root.retrace_exe(kind)
    if not exe.is_file():
        raise FileNotFoundError(f"retrace executable missing: {exe}")
    stdout, stderr, truncated = run(
        [
            str(exe),
            "--headless",
            f"--dump-state={call}",
            "--dump-format=json",
            str(path),
        ],
        timeout=timeout,
        max_bytes=MAX_STATE_JSON_BYTES,
        max_stderr_bytes=200_000,
    )
    if truncated:
        raise RuntimeError(
            f"complete state JSON exceeded the {MAX_STATE_JSON_BYTES}-byte safety limit; "
            "trim the trace or choose a lower call number"
        )
    try:
        state = json.loads(stdout, strict=False)
    except json.JSONDecodeError as exc:
        detail = stderr[-4000:].strip() or stdout[:1000]
        raise RuntimeError(
            f"retrace returned complete but invalid state JSON ({exc}); output: {detail}"
        ) from exc
    return stdout, state, kind


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def dump_state(
    trace: str, call: int, api: str = "", timeout: int = 600
) -> dict[str, Any]:
    """Replay up to a call and dump the full pipeline/device state as JSON.

    The apitrace equivalent of RenderDoc's pipeline view for OpenGL and
    D3D8/D3D9. DirectDraw and D3D7 traces are capture-only upstream and cannot
    be replayed for state inspection.

    Replay has to run the trace on this machine's GPU up to that call, so it is
    slow and it can fail on very old games whose resources the driver no longer
    likes. If it fails, trim_trace to a short range first.
    """
    path = _resolve_trace(trace)
    stdout, state, kind = _capture_state_json(path, call, api, timeout)
    return {
        "trace": str(path),
        "call": call,
        "api": kind,
        "state_json_bytes": len(stdout.encode("utf-8")),
        "state": _summarize_state_payload(state),
    }


@mcp.tool(annotations=WRITES_FILES, structured_output=True)
@guard
def dump_images(
    trace: str,
    calls: str = "*/frame",
    out_prefix: str = "",
    timeout: int = 900,
    mrt: bool = False,
) -> dict[str, Any]:
    """Replay the trace and write PNG images (per frame, or per specified call).

    Useful for finding *which* draw call produces the HUD, the world, or the
    problem you are chasing -- dump images at every draw in a frame and look.
    """
    path = _resolve_trace(trace)
    root = ENV.any_root()
    prefix = _resolved_output(
        out_prefix, trace_dir() / f"{path.stem}-images" / path.stem
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix = _unique_prefix(prefix)
    cmd = [
        str(root.apitrace_exe),
        "dump-images",
        f"--calls={_resolve_callset(calls)}",
        f"--output={prefix}",
    ]
    if mrt:
        cmd.append("--mrt")
    cmd.append(str(path))
    stdout, stderr, output_truncated = run(
        cmd, timeout=timeout, max_bytes=20_000, max_stderr_bytes=20_000
    )
    produced: list[str] = []
    produced_count = 0
    for candidate in _names_with_prefix(prefix.parent, prefix.name):
        if not candidate.is_file() or not candidate.name.endswith(".png"):
            continue
        produced_count += 1
        if len(produced) < 200:
            produced.append(str(candidate.resolve()))
    produced.sort()
    return {
        "trace": str(path),
        "calls": calls,
        "mrt": mrt,
        "output_prefix": str(prefix),
        "output_dir": str(prefix.parent),
        "images": produced,
        "image_count": produced_count,
        "images_truncated": produced_count > len(produced),
        "command_output_truncated": output_truncated,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


@mcp.tool(annotations=WRITES_FILES, structured_output=True)
@guard
def trim_trace(
    trace: str, calls: str = "", frames: str = "", output: str = ""
) -> dict[str, Any]:
    """Make a raw call/frame slice of a trace.

    A slice beginning at call/frame zero often remains replayable and makes every
    downstream tool faster. A mid-run OpenGL slice can omit setup/resources; use
    gltrim_trace when replayability matters.
    """
    path = _resolve_trace(trace)
    root = ENV.any_root()
    if not calls and not frames:
        raise ValueError("pass calls= or frames=")
    out = _unique_file(
        _resolved_output(output, trace_dir() / f"{path.stem}-trim.trace")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(root.apitrace_exe), "trim"]
    if calls:
        cmd.append(f"--calls={_resolve_callset(calls)}")
    if frames:
        cmd.append(f"--frames={frames}")
    cmd.extend(["--output", str(out), str(path)])
    stdout, stderr, _ = run(cmd, timeout=1800, max_bytes=20000)
    result = {
        "output": str(out),
        "exists": out.is_file(),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }
    if out.is_file():
        result["size_mb"] = round(out.stat().st_size / (1024 * 1024), 2)
    else:
        raise RuntimeError(
            f"apitrace trim exited successfully but did not create expected output: {out}; "
            f"stderr: {stderr[-2000:]}"
        )
    return result


@mcp.tool(annotations=WRITES_FILES, structured_output=True)
@guard
def gltrim_trace(
    trace: str,
    frames: str,
    setup_frames: str = "",
    output: str = "",
    keep_all_states: bool = False,
    swap_to_finish: bool = False,
    top_calls_per_frame: int = 0,
    timeout: int = 1800,
) -> dict[str, Any]:
    """OpenGL-specific frame trim that preserves replay setup state."""
    path = _resolve_trace(trace)
    if _api_of(path) != "gl":
        raise ValueError("gltrim_trace only accepts OpenGL traces")
    if not frames:
        raise ValueError("frames is required")
    if top_calls_per_frame < 0:
        raise ValueError("top_calls_per_frame must be non-negative")
    root = ENV.any_root()
    out = _unique_file(
        _resolved_output(output, trace_dir() / f"{path.stem}-gltrim.trace")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(root.apitrace_exe),
        "gltrim",
        f"--frames={frames}",
        f"--output={out}",
    ]
    if setup_frames:
        cmd.append(f"--setupframes={setup_frames}")
    if keep_all_states:
        cmd.append("--keep-all-states")
    if swap_to_finish:
        cmd.append("--swap-to-finish")
    if top_calls_per_frame:
        cmd.append(f"--top-calls-per-frame={top_calls_per_frame}")
    cmd.append(str(path))
    stdout, stderr, truncated = run(
        cmd, timeout=timeout, max_bytes=20_000, max_stderr_bytes=20_000
    )
    if not out.is_file():
        raise RuntimeError(
            f"apitrace gltrim did not create expected output {out}; stderr: {stderr[-2000:]}"
        )
    return {
        "trace": str(path),
        "output": str(out),
        "size_mb": round(out.stat().st_size / (1024 * 1024), 2),
        "frames": frames,
        "setup_frames": setup_frames or None,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "output_truncated": truncated,
    }


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def leak_report(
    trace: str, timeout: int = 600, max_bytes: int = 200_000
) -> dict[str, Any]:
    """Run apitrace's object-leak checker against a trace."""
    _validate_output_limit(max_bytes)
    path = _resolve_trace(trace)
    root = ENV.any_root()
    stdout, stderr, truncated = run(
        [str(root.apitrace_exe), "leaks", str(path)],
        timeout=timeout,
        max_bytes=max_bytes,
        max_stderr_bytes=40_000,
    )
    report = stdout if stdout.strip() else stderr
    truncated = truncated or stderr.startswith("<stderr truncated")
    return {
        "trace": str(path),
        "report": report,
        "truncated": truncated,
        "stream": "stdout" if stdout.strip() else "stderr",
    }


@mcp.tool(annotations=WRITES_FILES, structured_output=True)
@guard
def repack_trace(
    trace: str,
    output: str = "",
    compression: str = "snappy",
    quality: int | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Repack a trace with Snappy, Brotli, Zstandard, or zlib compression."""
    path = _resolve_trace(trace)
    root = ENV.any_root()
    compression = compression.lower()
    if compression == "snappy":
        if quality is not None:
            raise ValueError("snappy does not accept a quality setting")
        option = "--snappy"
    elif compression == "zlib":
        if quality is not None:
            raise ValueError("zlib does not accept a quality setting")
        option = "--zlib"
    elif compression == "brotli":
        if quality is not None and not 0 <= quality <= 11:
            raise ValueError("brotli quality must be between 0 and 11")
        option = "--brotli" if quality is None else f"--brotli={quality}"
    elif compression in ("zstd", "zstandard"):
        if quality is not None and not 1 <= quality <= 22:
            raise ValueError("zstd quality must be between 1 and 22")
        option = "--zstd" if quality is None else f"--zstd={quality}"
        compression = "zstd"
    else:
        raise ValueError("compression must be snappy, brotli, zstd, or zlib")
    out = _unique_file(
        _resolved_output(output, trace_dir() / f"{path.stem}-{compression}.trace")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    stdout, stderr, truncated = run(
        [str(root.apitrace_exe), "repack", option, str(path), str(out)],
        timeout=timeout,
        max_bytes=20_000,
        max_stderr_bytes=20_000,
    )
    if not out.is_file():
        raise RuntimeError(
            f"apitrace repack did not create expected output {out}; stderr: {stderr[-2000:]}"
        )
    return {
        "trace": str(path),
        "output": str(out),
        "compression": compression,
        "quality": quality,
        "input_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "output_size_mb": round(out.stat().st_size / (1024 * 1024), 2),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "output_truncated": truncated,
    }


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def diff_traces(
    reference_trace: str,
    source_trace: str,
    calls: str = "0-10000",
    reference_calls: str = "",
    source_calls: str = "",
    tool: str = "python",
    call_numbers: bool = False,
    suppress_common_lines: bool = False,
    width: int = 0,
    extra_args: list[str] | str | None = None,
    timeout: int = 600,
    max_bytes: int = 200_000,
) -> dict[str, Any]:
    """Compare two traces with apitrace's semantic call-stream differ."""
    _validate_output_limit(max_bytes)
    reference = _resolve_trace(reference_trace)
    source = _resolve_trace(source_trace)
    if tool not in ("python", "diff", "sdiff", "wdiff"):
        raise ValueError("tool must be python, diff, sdiff, or wdiff")
    if width < 0:
        raise ValueError("width must be non-negative")
    root = ENV.any_root()
    cmd = [
        str(root.apitrace_exe),
        "diff",
        f"--apitrace={root.apitrace_exe}",
        f"--tool={tool}",
    ]
    shared_calls = _resolve_callset(calls) if calls else "0-10000"
    cmd.append(f"--calls={shared_calls}")
    if reference_calls:
        cmd.append(f"--ref-calls={_resolve_callset(reference_calls)}")
    if source_calls:
        cmd.append(f"--src-calls={_resolve_callset(source_calls)}")
    if call_numbers:
        cmd.append("--call-nos")
    if suppress_common_lines:
        cmd.append("--suppress-common-lines")
    if width:
        cmd.append(f"--width={width}")
    extras = _coerce_argv(extra_args, label="extra_args")
    _reject_options(
        extras,
        (
            "-a",
            "--apitrace",
            "-t",
            "--tool",
            "-c",
            "--calls",
            "--ref-calls",
            "--src-calls",
        ),
        label="extra_args",
    )
    cmd.extend(extras)
    cmd.extend([str(reference), str(source)])
    stdout, stderr, truncated = run(
        cmd,
        timeout=timeout,
        max_bytes=max_bytes,
        max_stderr_bytes=40_000,
    )
    return {
        "reference_trace": str(reference),
        "source_trace": str(source),
        "calls": shared_calls,
        "reference_calls": reference_calls or None,
        "source_calls": source_calls or None,
        "tool": tool,
        "diff": stdout,
        "truncated": truncated,
        "stderr_tail": stderr[-4000:],
    }


@mcp.tool(annotations=READ_ONLY, structured_output=True)
@guard
def diff_state(
    trace: str,
    reference_call: int,
    source_call: int,
    source_trace: str = "",
    api: str = "",
    source_api: str = "",
    ignore_added: bool = False,
    keep_images: bool = False,
    extra_args: list[str] | str | None = None,
    timeout: int = 600,
    max_bytes: int = 200_000,
) -> dict[str, Any]:
    """Capture and compare complete retracer state at two calls."""
    _validate_output_limit(max_bytes)
    reference = _resolve_trace(trace)
    source = _resolve_trace(source_trace) if source_trace else reference
    root = ENV.any_root()
    extras = _coerce_argv(extra_args, label="extra_args")
    _reject_options(
        extras,
        ("--ignore-added", "--keep-images"),
        label="extra_args",
    )
    with tempfile.TemporaryDirectory(prefix="apitrace-mcp-state-") as temp:
        directory = Path(temp).resolve()
        ref_path = directory / "reference.json"
        src_path = directory / "source.json"
        ref_json, ref_state, ref_kind = _capture_state_json(
            reference, reference_call, api, timeout
        )
        ref_json_bytes = len(ref_json.encode("utf-8"))
        # Retracers sometimes emit literal control characters. `strict=False`
        # accepts that dialect above; jsondiff uses strict json.load(), so write
        # canonical JSON from the parsed objects instead of forwarding raw text.
        with ref_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                ref_state, stream, ensure_ascii=False, separators=(",", ":")
            )
        del ref_json, ref_state
        src_json, src_state, src_kind = _capture_state_json(
            source, source_call, source_api or api, timeout
        )
        src_json_bytes = len(src_json.encode("utf-8"))
        with src_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                src_state, stream, ensure_ascii=False, separators=(",", ":")
            )
        del src_json, src_state
        cmd = [str(root.apitrace_exe), "diff-state"]
        if ignore_added:
            cmd.append("--ignore-added")
        if keep_images:
            cmd.append("--keep-images")
        cmd.extend(extras)
        cmd.extend([str(ref_path), str(src_path)])
        stdout, stderr, truncated = run(
            cmd,
            timeout=timeout,
            max_bytes=max_bytes,
            max_stderr_bytes=40_000,
        )
    return {
        "reference": {
            "trace": str(reference),
            "call": reference_call,
            "api": ref_kind,
            "state_json_bytes": ref_json_bytes,
        },
        "source": {
            "trace": str(source),
            "call": source_call,
            "api": src_kind,
            "state_json_bytes": src_json_bytes,
        },
        "ignore_added": ignore_added,
        "keep_images": keep_images,
        "differences_found": bool(stdout.strip()),
        "diff": stdout,
        "truncated": truncated,
        "stderr_tail": stderr[-4000:],
    }


@mcp.tool(annotations=WRITES_FILES, structured_output=True)
@guard
def diff_images(
    reference_prefix: str,
    source_prefix: str,
    output: str = "",
    fuzz: float = 0.05,
    alpha: bool = False,
    show_all: bool = False,
    verbose: bool = False,
    extra_args: list[str] | str | None = None,
    timeout: int = 600,
    max_bytes: int = 100_000,
) -> dict[str, Any]:
    """Compare image sets and create an HTML report plus new diff/thumbnail sidecars.

    Upstream writes sidecars beside the source prefixes. This wrapper refuses to
    run if any potential sidecar already exists and never enables --overwrite.
    """
    _validate_output_limit(max_bytes)
    if importlib.util.find_spec("PIL") is None:
        raise RuntimeError(
            "apitrace diff-images requires Pillow (module PIL), which is not installed. "
            "Install it with `uv pip install -e \".[images]\"` and retry."
        )
    if not 0.0 <= fuzz <= 1.0:
        raise ValueError("fuzz must be between 0 and 1")
    reference = _resolved_image_prefix(reference_prefix, label="reference prefix")
    source = _resolved_image_prefix(source_prefix, label="source prefix")
    sidecars = _image_sidecars(reference, source)
    reserved_sidecars = _reserve_output_files(sidecars, label="diff-images")
    try:
        out = _unique_file(
            _resolved_output(output, trace_dir() / "image-diffs" / "index.html")
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        root = ENV.any_root()
        snapdiff = (
            root.apitrace_exe.resolve().parent.parent / "lib" / "scripts" / "snapdiff.py"
        )
        if not snapdiff.is_file():
            raise FileNotFoundError(
                f"apitrace diff-images helper missing: {snapdiff}"
            )
        cmd = [
            sys.executable,
            str(snapdiff),
            f"--output={out}",
            f"--fuzz={fuzz}",
        ]
        if alpha:
            cmd.append("--alpha")
        if show_all:
            cmd.append("--show-all")
        if verbose:
            cmd.append("--verbose")
        extras = _coerce_argv(extra_args, label="extra_args")
        _reject_options(
            extras,
            ("-o", "--output", "--overwrite", "-f", "--fuzz"),
            label="extra_args",
        )
        cmd.extend(extras)
        cmd.extend([str(reference), str(source)])
        comparison_failed = False
        lock, lock_token = _acquire_diff_images_lock(reference, source)
        try:
            try:
                stdout, stderr, truncated = run(
                    cmd,
                    timeout=timeout,
                    max_bytes=max_bytes,
                    max_stderr_bytes=40_000,
                )
            except CommandError as exc:
                if exc.returncode != 1 or not _complete_html(out):
                    if "No module named 'PIL'" in exc.stderr:
                        raise RuntimeError(
                            "apitrace diff-images could not import Pillow; install the optional "
                            "extra with `uv pip install -e \".[images]\"` and retry"
                        ) from exc
                    raise
                # snapdiff.py deliberately exits 1 for a valid MISMATCH/MISSING result.
                comparison_failed = True
                stdout, stderr, truncated = "", exc.stderr, False
        finally:
            _release_diff_images_lock(lock, lock_token)
        if not _complete_html(out):
            raise RuntimeError(
                f"apitrace diff-images did not produce a complete HTML report: {out}; "
                f"stderr: {stderr[-2000:]}"
            )
        created_sidecars = [str(path) for path in sidecars if path.is_file()]
        return {
            "reference_prefix": str(reference),
            "source_prefix": str(source),
            "command": cmd,
            "report": str(out),
            "result": "different_or_missing" if comparison_failed else "match",
            "fuzz": fuzz,
            "alpha": alpha,
            "sidecars": created_sidecars[:200],
            "sidecar_count": len(created_sidecars),
            "sidecars_truncated": len(created_sidecars) > 200,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "output_truncated": truncated,
        }
    finally:
        # Release the sidecar reservations even on failure (timeout, incomplete
        # report), so a retry is not permanently refused with FileExistsError.
        # Created sidecars exist on disk, so exists() still blocks overwrites.
        _release_output_files(reserved_sidecars)


@mcp.tool(annotations=WRITES_FILES, structured_output=True)
@guard
def sed_trace(
    trace: str,
    expressions: list[str] | str | None = None,
    properties: list[str] | str | None = None,
    calls: str = "",
    output: str = "",
    extra_args: list[str] | str | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Stream-edit trace enum/string symbols or trace properties into a new trace."""
    path = _resolve_trace(trace)
    edits = _coerce_string_list(expressions, label="expressions")
    props = _coerce_string_list(properties, label="properties")
    if not edits and not props:
        raise ValueError("pass at least one search/replace expression or property")
    if any("=" not in prop for prop in props):
        raise ValueError("each property must use NAME=VALUE syntax")
    out = _unique_file(
        _resolved_output(output, trace_dir() / f"{path.stem}-edited.trace")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    root = ENV.any_root()
    cmd = [str(root.apitrace_exe), "sed", f"--output={out}"]
    for expression in edits:
        cmd.extend(["-e", expression])
    for prop in props:
        cmd.append(f"--property={prop}")
    if calls:
        cmd.append(f"--calls={_resolve_callset(calls)}")
    extras = _coerce_argv(extra_args, label="extra_args")
    _reject_options(
        extras,
        ("-o", "--output", "-e", "--property", "--calls"),
        label="extra_args",
    )
    cmd.extend(extras)
    cmd.append(str(path))
    stdout, stderr, truncated = run(
        cmd,
        timeout=timeout,
        max_bytes=20_000,
        max_stderr_bytes=20_000,
    )
    if not out.is_file():
        raise RuntimeError(
            f"apitrace sed did not create expected output {out}; stderr: {stderr[-2000:]}"
        )
    return {
        "trace": str(path),
        "output": str(out),
        "size_mb": round(out.stat().st_size / (1024 * 1024), 2),
        "expressions": edits,
        "properties": props,
        "calls": calls or None,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "output_truncated": truncated,
    }


@mcp.tool(annotations=STARTS_PROCESS, structured_output=True)
@guard
def replay_trace(
    trace: str,
    api: str = "",
    profile: str = "",
    headless: bool = False,
    extra_args: list[str] | str | None = None,
) -> dict[str, Any]:
    """Replay a trace in the background (session-based). Returns a session id.

    profile: '', 'cpu', 'gpu', 'frames', 'pixels', 'memory', 'calls',
    'frame_metrics', or 'draw_metrics'. Profiling output
    lands in the session log -- poll it with trace_status.
    """
    path = _resolve_trace(trace)
    root = ENV.any_root()
    kind = _retrace_kind(api, path)
    exe = root.retrace_exe(kind)
    if not exe.is_file():
        raise FileNotFoundError(f"retrace executable missing: {exe}")
    cmd = [str(exe)]
    flag = {
        "cpu": "--pcpu",
        "gpu": "--pgpu",
        "frames": "--pframe-times",
        "pixels": "--ppd",
        "memory": "--pmem",
        "calls": "--pcalls",
        "frame_metrics": "--pframes",
        "draw_metrics": "--pdrawcalls",
    }.get(profile)
    if profile and not flag:
        raise ValueError(
            "profile must be one of: cpu, gpu, frames, pixels, memory, calls, "
            "frame_metrics, draw_metrics"
        )
    if flag:
        cmd.append(flag)
    if headless:
        cmd.append("--headless")
    cmd.extend(_coerce_argv(extra_args, label="extra_args"))
    cmd.append(str(path))
    session = MANAGER.start("replay", cmd, log_dir=trace_dir() / "logs")
    time.sleep(1.0)
    status = session.status()
    status["api"] = kind
    status["profile"] = profile or None
    return status


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
