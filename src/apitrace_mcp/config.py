"""Locating apitrace installs and deciding which one to use.

A win64 apitrace can only trace 64-bit processes and a win32 apitrace can only
trace 32-bit ones -- the wrapper DLL has to load into the target's address
space. Most pre-2012 games are 32-bit, so both builds are worth having and the
server picks between them from the target's PE header.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .pe import PEError, read_pe

ENV_ROOT_WIN64 = "APITRACE_MCP_ROOT_WIN64"
ENV_ROOT_WIN32 = "APITRACE_MCP_ROOT_WIN32"
ENV_TRACE_DIR = "APITRACE_MCP_TRACE_DIR"
ENV_EXTRA_ROOTS = "APITRACE_MCP_ROOTS"  # os.pathsep-separated, arch auto-detected

def _nearby_globs() -> list[str]:
    """Patterns for apitrace builds unpacked near this project.

    Covers the two natural working layouts without hardcoding any machine
    path: a build unpacked inside the clone, and one in a sibling directory
    (``X\\apitrace-mcp`` next to ``X\\apitrace\\apitrace-14.0-win64``). For a
    site-packages install these directories simply match nothing.
    """
    here = Path(__file__).resolve()
    patterns: list[str] = []
    for level in (2, 3):  # the project root, and its parent directory
        if level >= len(here.parents):
            break
        base = here.parents[level]
        patterns.append(str(base / "apitrace-*-win*"))
        patterns.append(str(base / "apitrace*" / "apitrace-*-win*"))
    return patterns


SEARCH_GLOBS = _nearby_globs() + [
    r"C:\apitrace\apitrace-*-win*",
    r"C:\Program Files\apitrace*",
    r"C:\Program Files (x86)\apitrace*",
    os.path.expanduser(r"~\apitrace\apitrace-*-win*"),
]

DEFAULT_TRACE_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser(r"~\AppData\Local")),
    "apitrace-mcp",
    "traces",
)

WRAPPER_FOR_API = {
    "gl": ["opengl32.dll"],
    "d3d7": ["ddraw.dll"],
    "d3d8": ["d3d8.dll"],
    "d3d9": ["d3d9.dll"],
    # D3D10/11 genuinely need `apitrace trace -a dxgi`; the DLL-drop route is
    # unreliable there because the D3D10/11 DLLs load each other by name.
    "dxgi": ["dxgitrace.dll", "dxgi.dll", "d3d11.dll"],
}

VALID_APIS = tuple(WRAPPER_FOR_API)

_SEMVER_RE = re.compile(
    r"(?<!\d)(\d+\.\d+(?:\.\d+){0,2}"
    r"(?:-(?!(?:win(?:32|64)|x86|x64|amd64)\b)"
    r"[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?)"
    r"(?=(?:-win(?:32|64)|-(?:x86|x64|amd64)|$))",
    re.IGNORECASE,
)


@dataclass
class ApitraceRoot:
    path: Path
    bits: int
    version: str | None = None

    @property
    def bin(self) -> Path:
        return self.path / "bin"

    @property
    def apitrace_exe(self) -> Path:
        return self.bin / "apitrace.exe"

    @property
    def wrappers(self) -> Path:
        return self.path / "lib" / "wrappers"

    def retrace_exe(self, api: str) -> Path:
        return self.bin / ("glretrace.exe" if api == "gl" else "d3dretrace.exe")

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "bits": self.bits,
            "version": self.version,
            "apitrace_exe": str(self.apitrace_exe),
            "wrappers_dir": str(self.wrappers),
            "wrappers_present": sorted(
                p.name for p in self.wrappers.glob("*.dll")
            )
            if self.wrappers.is_dir()
            else [],
        }


def _root_bits(root: Path, *, allow_name_fallback: bool = True) -> int | None:
    exe = root / "bin" / "apitrace.exe"
    if not exe.is_file():
        return None
    try:
        return read_pe(str(exe)).bits
    except (PEError, OSError):
        if not allow_name_fallback:
            return None
        # Fall back to the directory naming convention used by upstream zips.
        name = root.name.lower()
        if "win64" in name or "x64" in name:
            return 64
        if "win32" in name or "x86" in name:
            return 32
        return None


def _version_of(root: Path) -> str | None:
    """Extract a semantic version without confusing it with win32/win64."""
    name = re.sub(
        r"-(?:win(?:32|64)|x86|x64|amd64)$", "", root.name, flags=re.IGNORECASE
    )
    match = _SEMVER_RE.search(name)
    return match.group(1) if match else None


def _version_key(version: str | None) -> tuple[int, int, int, int, int, str]:
    """Comparable key for release discovery; final releases beat prereleases."""
    if version is None:
        return (-1, -1, -1, -1, -1, "")
    main, separator, prerelease = version.partition("-")
    numbers = [int(part) for part in main.split(".")]
    numbers.extend([0] * (4 - len(numbers)))
    return (*numbers[:4], 0 if separator else 1, prerelease.casefold())


def _configured_path(path_str: str, source: str) -> Path:
    raw = path_str.strip()
    if not raw:
        raise RuntimeError(f"{source} is set but empty")
    if raw.startswith('"') != raw.endswith('"'):
        raise RuntimeError(f"{source} contains an unmatched quote")
    if raw.startswith('"'):
        raw = raw[1:-1]
    path = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
    if path.name.casefold() == "bin" and (path.parent / "lib").is_dir():
        path = path.parent
    return path


def _validated_root(
    path_str: str,
    *,
    source: str,
    forced_bits: int | None = None,
    strict: bool,
) -> ApitraceRoot | None:
    """Validate both package layout and PE architecture.

    Explicit environment settings are strict so a typo or a win32/win64 mix-up
    is reported instead of silently falling through to an unrelated install.
    Automatic search hits remain best-effort.
    """
    try:
        path = _configured_path(path_str, source)
    except (OSError, RuntimeError) as exc:
        if strict:
            raise RuntimeError(f"invalid apitrace root from {source}: {exc}") from exc
        return None
    if not path.is_dir():
        if strict:
            raise RuntimeError(f"{source} is not an apitrace directory: {path}")
        return None
    exe = path / "bin" / "apitrace.exe"
    wrappers = path / "lib" / "wrappers"
    if not exe.is_file() or not wrappers.is_dir():
        if strict:
            raise RuntimeError(
                f"{source} does not contain bin/apitrace.exe and lib/wrappers: {path}"
            )
        return None
    try:
        actual_bits = read_pe(str(exe)).bits
    except (PEError, OSError) as exc:
        if strict:
            raise RuntimeError(f"cannot validate apitrace.exe from {source}: {exc}") from exc
        return None
    if actual_bits not in {32, 64}:
        if strict:
            raise RuntimeError(f"unsupported apitrace architecture in {source}: {actual_bits}")
        return None
    if forced_bits is not None and actual_bits != forced_bits:
        raise RuntimeError(
            f"{source} is configured as win{forced_bits}, but its apitrace.exe is "
            f"{actual_bits}-bit: {exe}"
        )
    return ApitraceRoot(path=path, bits=actual_bits, version=_version_of(path))


def discover_roots() -> dict[int, ApitraceRoot]:
    """Return {bits: root}, honouring env overrides before falling back to globs."""
    roots: dict[int, ApitraceRoot] = {}

    def consider(
        path_str: str,
        *,
        source: str,
        forced_bits: int | None = None,
        strict: bool = False,
    ) -> None:
        root = _validated_root(
            path_str, source=source, forced_bits=forced_bits, strict=strict
        )
        if root is not None and root.bits not in roots:
            roots[root.bits] = root

    if os.environ.get(ENV_ROOT_WIN64):
        consider(
            os.environ[ENV_ROOT_WIN64],
            source=ENV_ROOT_WIN64,
            forced_bits=64,
            strict=True,
        )
    if os.environ.get(ENV_ROOT_WIN32):
        consider(
            os.environ[ENV_ROOT_WIN32],
            source=ENV_ROOT_WIN32,
            forced_bits=32,
            strict=True,
        )
    for index, extra in enumerate(os.environ.get(ENV_EXTRA_ROOTS, "").split(os.pathsep)):
        if extra:
            consider(
                extra,
                source=f"{ENV_EXTRA_ROOTS}[{index}]",
                strict=True,
            )

    # Rank automatic candidates globally, not once per search pattern.  An old
    # install in an earlier location must not mask a newer install found by a
    # later glob.  Explicit roots above still retain unconditional precedence.
    automatic: dict[int, ApitraceRoot] = {}
    for pattern in SEARCH_GLOBS:
        for hit in glob.glob(pattern):
            candidate = _validated_root(
                hit, source=f"automatic search {pattern}", strict=False
            )
            if candidate is None:
                continue
            current = automatic.get(candidate.bits)
            if current is None or _version_key(candidate.version) > _version_key(
                current.version
            ):
                automatic[candidate.bits] = candidate
    for bits, root in automatic.items():
        roots.setdefault(bits, root)

    return roots


def trace_dir() -> Path:
    path = Path(os.environ.get(ENV_TRACE_DIR, DEFAULT_TRACE_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


class Environment:
    """Cached view of the apitrace installs on this machine."""

    def __init__(self) -> None:
        self._roots: dict[int, ApitraceRoot] | None = None

    @property
    def roots(self) -> dict[int, ApitraceRoot]:
        if self._roots is None:
            self._roots = discover_roots()
        return self._roots

    def refresh(self) -> None:
        self._roots = None

    def root_for_bits(self, bits: int) -> ApitraceRoot:
        root = self.roots.get(bits)
        if root is not None:
            return root
        have = ", ".join(f"win{b}" for b in sorted(self.roots)) or "none"
        raise RuntimeError(
            f"No {bits}-bit apitrace build found (installed: {have}). "
            f"A win{bits} apitrace is required to trace {bits}-bit processes -- "
            f"download apitrace-{bits == 64 and 'X.Y-win64' or 'X.Y-win32'}.7z from "
            "https://github.com/apitrace/apitrace/releases, unpack it next to the "
            f"existing build, and set {ENV_ROOT_WIN32 if bits == 32 else ENV_ROOT_WIN64}."
        )

    def any_root(self) -> ApitraceRoot:
        """For trace-file analysis, where the host arch does not matter."""
        if not self.roots:
            raise RuntimeError(
                "No apitrace install found. Set "
                f"{ENV_ROOT_WIN64}/{ENV_ROOT_WIN32} to an unpacked apitrace directory."
            )
        # Prefer 64-bit for parsing: it handles large traces without address space limits.
        return self.roots.get(64) or next(iter(self.roots.values()))

    def root_for_target(self, exe_path: str) -> tuple[ApitraceRoot, int]:
        bits = read_pe(exe_path).bits
        return self.root_for_bits(bits), bits


ENV = Environment()
