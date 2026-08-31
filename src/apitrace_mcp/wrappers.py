"""Safe manual wrapper-DLL installation and recovery.

The wrapper route necessarily replaces DLLs in a game directory.  Treat that
as a small transaction: record recovery information before the first rename,
hash everything we may later remove or restore, and never trust paths read from
the manifest without constraining them to the game directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import WRAPPER_FOR_API, ApitraceRoot

MANIFEST_NAME = ".apitrace-mcp-install.json"
MANIFEST_SCHEMA = 2
LAUNCHER_NAME = "apitrace-run.bat"
LOCK_NAME = ".apitrace-mcp.lock"

KNOWN_SHIMS = {
    "d3d9.dll": "ENB / ReShade / dgVoodoo / DXVK / a game-specific wrapper",
    "d3d8.dll": "d3d8to9 / dgVoodoo / ENB",
    "ddraw.dll": "dgVoodoo / DDrawCompat / cnc-ddraw",
    "opengl32.dll": "ReShade / a GL wrapper",
    "dxgi.dll": "ReShade / DXVK / special-k",
    "d3d11.dll": "ReShade / DXVK",
}

PROTECTED_DIRS = [
    Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve(),
]

_HASH_CHARS = frozenset("0123456789abcdef")
_MANIFEST_STATES = {
    "installing",
    "installed",
    "failed",
    "recovery_required",
    "uninstalling",
    "cleanup_required",
    "uninstalled",
}


def _check_target_dir(game_dir: Path) -> None:
    resolved = game_dir.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"not a directory: {resolved}")
    for protected in PROTECTED_DIRS:
        if resolved == protected or protected in resolved.parents:
            raise RuntimeError(
                f"refusing to install wrapper DLLs inside a system directory: {resolved}"
            )


def manifest_path(game_dir: Path) -> Path:
    return game_dir / MANIFEST_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, operation: str) -> None:
    """Revalidate a regular file at the last boundary before mutation."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{operation}: expected regular file is missing or unsafe: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{operation}: {path} changed during the transaction "
            f"(expected {expected}, found {actual}); refusing to continue"
        )


@contextmanager
def _transaction_lock(game_dir: Path) -> Iterator[None]:
    """Hold one cross-process wrapper transaction lock per game directory."""
    lock_path = game_dir / LOCK_NAME
    if lock_path.is_symlink():
        raise RuntimeError(f"refusing unsafe wrapper transaction lock symlink: {lock_path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY  # type: ignore[attr-defined]
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(f"cannot open wrapper transaction lock {lock_path}: {exc}") from exc
    handle = os.fdopen(descriptor, "r+b", closefd=True)
    locked = False
    try:
        opened = os.fstat(handle.fileno())
        current = lock_path.stat(follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_ino and current.st_ino and opened.st_ino != current.st_ino
        ):
            raise RuntimeError(f"unsafe wrapper transaction lock file: {lock_path}")
        if opened.st_size == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError(
                f"another wrapper install/uninstall is already active in {game_dir}"
            ) from exc
        yield
    finally:
        if locked:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()
        try:
            lock_path.unlink()
        except OSError:
            # A concurrently-held lock cannot be unlinked on Windows (sharing
            # violation), so mutual exclusion survives; the uncontended case
            # leaves the game directory clean.
            pass


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Move a file without a check/replace race at the destination."""
    if os.name == "nt":
        # Unlike os.replace, MoveFile-backed os.rename fails when destination
        # already exists on Windows (the only supported host platform here).
        os.rename(source, destination)
        return
    # Keep tests and library use conservative on other hosts too.
    os.link(source, destination)
    try:
        source.unlink()
    except OSError:
        destination.unlink(missing_ok=True)
        raise


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HASH_CHARS for char in value)
    )


def _child_path(game_dir: Path, value: object, expected: str | None = None) -> Path:
    """Return a manifest path only when it names one direct child.

    Absolute paths, traversal, alternate separators and symlinks are rejected.
    The latter matters because an otherwise innocent-looking ``d3d9.dll`` link
    could point at a system or shared game file.
    """
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise RuntimeError("wrapper manifest contains an invalid relative path")
    if Path(value).name != value or "/" in value or "\\" in value:
        raise RuntimeError(f"wrapper manifest path must be a filename: {value!r}")
    if expected is not None and value.casefold() != expected.casefold():
        raise RuntimeError(
            f"wrapper manifest path {value!r} does not match expected {expected!r}"
        )
    path = game_dir / value
    if path.is_symlink():
        raise RuntimeError(f"refusing to follow a wrapper manifest symlink: {path}")
    return path


def _validate_manifest(game_dir: Path, data: object) -> dict:
    if not isinstance(data, dict):
        raise RuntimeError("wrapper manifest root must be a JSON object")
    required_keys = {
        "schema",
        "state",
        "transaction_id",
        "api",
        "apitrace_root",
        "apitrace_bits",
        "game_exe",
        "files",
        "trace_file",
        "launcher",
        "recovery_files",
    }
    allowed_keys = required_keys | {
        "install_error",
        "uninstall_error",
        "rollback_problems",
    }
    if not required_keys.issubset(data) or not set(data).issubset(allowed_keys):
        raise RuntimeError("wrapper manifest has missing or unknown top-level fields")
    if data.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError(
            "unsupported or legacy wrapper manifest; recover it manually rather than "
            "letting apitrace-mcp guess which files are safe to delete"
        )
    api = data.get("api")
    if not isinstance(api, str) or api not in WRAPPER_FOR_API:
        raise RuntimeError("wrapper manifest contains an invalid api")
    if data.get("state") not in _MANIFEST_STATES:
        raise RuntimeError("wrapper manifest contains an invalid transaction state")
    transaction_id = data.get("transaction_id")
    if (
        not isinstance(transaction_id, str)
        or len(transaction_id) != 32
        or any(char not in _HASH_CHARS for char in transaction_id)
    ):
        raise RuntimeError("wrapper manifest contains an invalid transaction id")
    root_value = data.get("apitrace_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise RuntimeError("wrapper manifest contains an invalid apitrace root")
    apitrace_root = Path(root_value).resolve(strict=False)
    if type(data.get("apitrace_bits")) is not int or data["apitrace_bits"] not in {
        32,
        64,
    }:
        raise RuntimeError("wrapper manifest contains invalid apitrace bitness")
    _child_path(game_dir, data.get("game_exe"))
    trace_file = data.get("trace_file")
    if trace_file is not None and (
        not isinstance(trace_file, str) or not Path(trace_file).is_absolute()
    ):
        raise RuntimeError("wrapper manifest contains an invalid trace path")

    files = data.get("files")
    expected_dlls = WRAPPER_FOR_API[api]
    if not isinstance(files, list) or len(files) != len(expected_dlls):
        raise RuntimeError("wrapper manifest has the wrong wrapper file set")
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise RuntimeError("wrapper manifest file entries must be objects")
        if set(entry) != {
            "dll",
            "source",
            "installed_to",
            "backup",
            "installed_sha256",
            "original_sha256",
        }:
            raise RuntimeError("wrapper manifest file entry has missing or unknown fields")
        dll = entry.get("dll")
        if not isinstance(dll, str) or dll not in expected_dlls or dll in seen:
            raise RuntimeError("wrapper manifest contains an unexpected or duplicate DLL")
        seen.add(dll)
        _child_path(game_dir, entry.get("installed_to"), dll)
        backup = entry.get("backup")
        if backup is not None:
            _child_path(game_dir, backup, dll + ".apitrace-backup")
            if not _valid_hash(entry.get("original_sha256")):
                raise RuntimeError("wrapper manifest has no valid original-file hash")
        elif entry.get("original_sha256") is not None:
            raise RuntimeError("wrapper manifest hashes an original without a backup")
        if not _valid_hash(entry.get("installed_sha256")):
            raise RuntimeError("wrapper manifest has no valid installed-file hash")
        source = entry.get("source")
        if not isinstance(source, str) or not Path(source).is_absolute():
            raise RuntimeError("wrapper manifest contains an invalid source path")
        expected_source = (apitrace_root / "lib" / "wrappers" / dll).resolve(
            strict=False
        )
        if Path(source).resolve(strict=False) != expected_source:
            raise RuntimeError("wrapper manifest source does not match its apitrace root")
    if seen != set(expected_dlls):
        raise RuntimeError("wrapper manifest is missing an expected wrapper DLL")

    launcher = data.get("launcher")
    if launcher is not None:
        if not isinstance(launcher, dict):
            raise RuntimeError("wrapper manifest launcher must be an object")
        if set(launcher) != {"path", "sha256"}:
            raise RuntimeError("wrapper manifest launcher has missing or unknown fields")
        _child_path(game_dir, launcher.get("path"), LAUNCHER_NAME)
        if not _valid_hash(launcher.get("sha256")):
            raise RuntimeError("wrapper manifest has no valid launcher hash")

    recovery_files = data.get("recovery_files", [])
    if not isinstance(recovery_files, list):
        raise RuntimeError("wrapper manifest recovery_files must be a list")
    allowed_hashes = {entry["installed_sha256"] for entry in files}
    if launcher:
        allowed_hashes.add(launcher["sha256"])
    for entry in recovery_files:
        if not isinstance(entry, dict):
            raise RuntimeError("invalid recovery file record")
        if set(entry) != {"path", "sha256"}:
            raise RuntimeError("invalid recovery file fields")
        name = entry.get("path")
        if not isinstance(name, str) or not name.startswith(".apitrace-mcp-remove-"):
            raise RuntimeError("invalid recovery file path")
        _child_path(game_dir, name)
        if entry.get("sha256") not in allowed_hashes:
            raise RuntimeError("invalid recovery file hash")
    for key in ("install_error", "uninstall_error"):
        if key in data and not isinstance(data[key], str):
            raise RuntimeError(f"wrapper manifest {key} must be a string")
    if "rollback_problems" in data and (
        not isinstance(data["rollback_problems"], list)
        or not all(isinstance(problem, str) for problem in data["rollback_problems"])
    ):
        raise RuntimeError("wrapper manifest rollback_problems must be a string list")
    return data


def read_manifest(game_dir: Path) -> dict | None:
    game_dir = game_dir.resolve()
    path = manifest_path(game_dir)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"unsafe wrapper manifest path: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"wrapper manifest is unreadable or corrupt; refusing to overwrite it: {path}"
        ) from exc
    return _validate_manifest(game_dir, data)


def _require_manifest_identity(
    game_dir: Path, transaction_id: str, expected_state: str
) -> None:
    """Refuse to unlink a manifest replaced after the transaction update."""
    manifest = read_manifest(game_dir)
    if manifest is None:
        raise RuntimeError("wrapper recovery manifest disappeared before cleanup")
    if (
        manifest["transaction_id"] != transaction_id
        or manifest["state"] != expected_state
    ):
        raise RuntimeError(
            "wrapper recovery manifest changed during cleanup; refusing to unlink it"
        )


def _create_manifest(path: Path, manifest: dict) -> None:
    """Create, but never replace, the recovery manifest."""
    payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def _update_manifest(path: Path, manifest: dict) -> None:
    """Atomically update an existing transaction record."""
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _batch_escape(value: str) -> str:
    if any(char in value for char in "\r\n\0"):
        raise ValueError("batch launcher paths may not contain newlines or NULs")
    # Delayed expansion is disabled in the launcher, so only percent expansion
    # remains active inside a quoted value.
    return value.replace("%", "%%")


def _launcher_bytes(exe_name: str, trace_file: Path) -> bytes:
    lines = [
        "@echo off",
        "setlocal DisableDelayedExpansion",
        "chcp 65001 >nul",
        'pushd "%~dp0" || exit /b 1',
        f'set "TRACE_FILE={_batch_escape(str(trace_file))}"',
        'if exist "%TRACE_FILE%" (',
        '  echo Refusing to overwrite existing apitrace capture: "%TRACE_FILE%" 1>&2',
        "  popd",
        "  exit /b 73",
        ")",
        f'"{_batch_escape(exe_name)}" %*',
        'set "_APITRACE_EXIT=%ERRORLEVEL%"',
        "popd",
        "exit /b %_APITRACE_EXIT%",
        "",
    ]
    # No BOM. cmd.exe does *not* skip one at the start of a batch file -- the
    # bytes are read as part of the first command, so "@echo off" becomes an
    # unrecognised command and the launcher misbehaves. Verified against a real
    # Morrowind capture: the .bat existed and its text was correct, but it would
    # not run. `chcp 65001` earns its keep instead -- it switches the console to
    # UTF-8 before any path-bearing line is read, which is what non-ASCII needs.
    return "\r\n".join(lines).encode("utf-8")


def _install(
    root: ApitraceRoot,
    game_exe: str | Path,
    api: str,
    *,
    trace_file: str | Path | None = None,
    force: bool = False,
) -> dict:
    exe = Path(game_exe).expanduser().resolve()
    if not exe.is_file():
        raise FileNotFoundError(f"game executable not found: {exe}")
    game_dir = exe.parent.resolve()
    _check_target_dir(game_dir)

    if api not in WRAPPER_FOR_API:
        raise ValueError(f"unknown api {api!r}; expected one of {list(WRAPPER_FOR_API)}")
    if api == "dxgi":
        raise RuntimeError(
            "D3D10/11 cannot be traced reliably by dropping DLLs -- the D3D DLLs load "
            "each other by name. Use trace_launch (apitrace trace -a dxgi) instead."
        )

    manifest_file = manifest_path(game_dir)
    if manifest_file.exists():
        existing = read_manifest(game_dir)
        raise RuntimeError(
            f"an apitrace wrapper transaction is already present in {game_dir} "
            f"(api={existing.get('api') if existing else 'unknown'}, "
            f"state={existing.get('state') if existing else 'unknown'}). Run "
            "uninstall_wrapper first; force never overwrites recovery manifests."
        )

    plans: list[dict] = []
    staged: list[Path] = []
    warnings: list[str] = []
    launcher_path: Path | None = None
    launcher_stage: Path | None = None
    launcher_record: dict | None = None
    token = uuid.uuid4().hex

    try:
        for dll_name in WRAPPER_FOR_API[api]:
            src = (root.wrappers / dll_name).resolve()
            if not src.is_file():
                raise FileNotFoundError(f"wrapper not found in apitrace install: {src}")
            dst = _child_path(game_dir, dll_name, dll_name)
            if dst.exists() and not dst.is_file():
                raise RuntimeError(f"wrapper destination is not a regular file: {dst}")
            backup = _child_path(game_dir, dll_name + ".apitrace-backup")
            if backup.exists():
                raise RuntimeError(
                    f"{backup} already exists -- refusing to overwrite an earlier backup; "
                    "force does not bypass recovery safety"
                )
            stage = game_dir / f".apitrace-mcp-stage-{token}-{dll_name}"
            shutil.copy2(src, stage)
            staged.append(stage)
            installed_hash = _sha256(stage)
            if installed_hash != _sha256(src):
                raise RuntimeError(f"wrapper changed while staging: {src}")
            had_original = dst.exists()
            original_hash = _sha256(dst) if had_original else None
            if had_original:
                shim = KNOWN_SHIMS.get(dll_name.lower())
                warnings.append(
                    f"{dll_name} already existed in the game folder"
                    + (f" (likely {shim})" if shim else "")
                    + f"; backed up to {backup.name}. It will be restored on uninstall, "
                    "but that mod is inactive while apitrace is installed."
                )
            plans.append(
                {
                    "dll": dll_name,
                    "source": str(src),
                    "installed_to": dll_name,
                    "backup": backup.name if had_original else None,
                    "installed_sha256": installed_hash,
                    "original_sha256": original_hash,
                    "_stage": stage,
                    "_dst": dst,
                    "_backup": backup,
                    "_backup_moved": False,
                    "_installed_activated": False,
                }
            )

        trace_path = None
        if trace_file is not None:
            trace_path = Path(trace_file).expanduser().resolve(strict=False)
            launcher_path = _child_path(game_dir, LAUNCHER_NAME, LAUNCHER_NAME)
            if launcher_path.exists():
                raise RuntimeError(
                    f"refusing to overwrite existing launcher without a recovery manifest: "
                    f"{launcher_path}"
                )
            launcher_stage = game_dir / f".apitrace-mcp-stage-{token}-{LAUNCHER_NAME}"
            launcher_stage.write_bytes(_launcher_bytes(exe.name, trace_path))
            staged.append(launcher_stage)
            launcher_record = {
                "path": LAUNCHER_NAME,
                "sha256": _sha256(launcher_stage),
            }

        public_files = [
            {key: value for key, value in plan.items() if not key.startswith("_")}
            for plan in plans
        ]
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "state": "installing",
            "transaction_id": token,
            "api": api,
            "apitrace_root": str(root.path.resolve()),
            "apitrace_bits": root.bits,
            "game_exe": exe.name,
            "files": public_files,
            "trace_file": str(trace_path) if trace_path else None,
            "launcher": launcher_record,
            "recovery_files": [],
        }
        _create_manifest(manifest_file, manifest)

        touched: list[dict] = []
        launcher_installed = False
        try:
            for plan in plans:
                touched.append(plan)
                if plan["backup"] is not None:
                    _require_hash(
                        plan["_dst"], plan["original_sha256"], "backing up original DLL"
                    )
                    _rename_no_replace(plan["_dst"], plan["_backup"])
                    plan["_backup_moved"] = True
                    _require_hash(
                        plan["_backup"],
                        plan["original_sha256"],
                        "verifying original DLL backup",
                    )
                _require_hash(
                    plan["_stage"], plan["installed_sha256"], "activating staged wrapper"
                )
                _rename_no_replace(plan["_stage"], plan["_dst"])
                plan["_installed_activated"] = True
                _require_hash(
                    plan["_dst"], plan["installed_sha256"], "verifying active wrapper"
                )
            if launcher_stage is not None and launcher_path is not None:
                _require_hash(
                    launcher_stage, launcher_record["sha256"], "activating launcher"  # type: ignore[index]
                )
                _rename_no_replace(launcher_stage, launcher_path)
                launcher_installed = True
                _require_hash(
                    launcher_path, launcher_record["sha256"], "verifying launcher"  # type: ignore[index]
                )
            manifest["state"] = "installed"
            _update_manifest(manifest_file, manifest)
        except Exception as exc:
            rollback_problems: list[str] = []
            if launcher_installed and launcher_path is not None:
                try:
                    if _sha256(launcher_path) != launcher_record["sha256"]:  # type: ignore[index]
                        raise RuntimeError("launcher changed during rollback")
                    launcher_path.unlink()
                except (OSError, RuntimeError) as rollback_exc:
                    rollback_problems.append(f"{launcher_path}: {rollback_exc}")
            for plan in reversed(touched):
                dst, backup = plan["_dst"], plan["_backup"]
                try:
                    if plan["_installed_activated"] and dst.exists():
                        if _sha256(dst) != plan["installed_sha256"]:
                            raise RuntimeError("installed DLL changed during rollback")
                        dst.unlink()
                    if plan["_backup_moved"] and backup.exists():
                        _require_hash(
                            backup,
                            plan["original_sha256"],
                            "restoring original DLL after failed install",
                        )
                        _rename_no_replace(backup, dst)
                except (OSError, RuntimeError) as rollback_exc:
                    rollback_problems.append(f"{dst}: {rollback_exc}")
            manifest["state"] = "recovery_required" if rollback_problems else "failed"
            manifest["install_error"] = str(exc)
            manifest["rollback_problems"] = rollback_problems
            try:
                _update_manifest(manifest_file, manifest)
            except OSError:
                pass
            raise RuntimeError(
                f"wrapper installation failed; recovery manifest retained at {manifest_file}: "
                f"{exc}"
            ) from exc
    finally:
        for path in staged:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    installed = [
        {
            "dll": plan["dll"],
            "source": plan["source"],
            "installed_to": str(plan["_dst"]),
            "backup": str(plan["_backup"]) if plan["backup"] else None,
            "sha256": plan["installed_sha256"],
        }
        for plan in plans
    ]
    if force:
        warnings.append(
            "force was requested, but recovery manifests and existing backups are never "
            "overwritten"
        )
    return {
        "installed": installed,
        "manifest": str(manifest_file),
        "launcher": str(launcher_path) if launcher_path else None,
        "warnings": warnings,
        "next_steps": [
            f"Launch the game normally from {game_dir}"
            + (f" (or run {launcher_path.name} to set TRACE_FILE)" if launcher_path else ""),
            "Play the part you want captured, then quit the game cleanly so the trace is flushed",
            "Call uninstall_wrapper afterwards -- leaving the wrapper in place will keep "
            "tracing (and slowing) every future run",
        ],
    }


def _uninstall(game_exe_or_dir: str | Path) -> dict:
    path = Path(game_exe_or_dir).expanduser()
    game_dir = (path if path.is_dir() else path.parent).resolve()
    _check_target_dir(game_dir)
    manifest = read_manifest(game_dir)
    if manifest is None:
        return {
            "removed": [],
            "restored": [],
            "note": f"no apitrace-mcp install manifest found in {game_dir}; nothing to undo",
        }

    removed: list[str] = []
    restored: list[str] = []
    problems: list[str] = []
    actions: list[dict] = []

    # Validate every destructive target and every hash before changing anything.
    for entry in manifest["files"]:
        dst = _child_path(game_dir, entry["installed_to"], entry["dll"])
        backup = (
            _child_path(game_dir, entry["backup"], entry["dll"] + ".apitrace-backup")
            if entry["backup"]
            else None
        )
        dst_hash = _sha256(dst) if dst.is_file() else None
        backup_hash = _sha256(backup) if backup is not None and backup.is_file() else None
        already_restored = (
            backup is not None
            and backup_hash is None
            and dst_hash == entry["original_sha256"]
        )
        # The user copy-restored the original by hand and left the backup in
        # place; renaming the backup over dst would collide, so treat the
        # entry as restored and drop the redundant backup instead.
        copy_restored = (
            backup is not None
            and backup_hash == entry["original_sha256"]
            and dst_hash == entry["original_sha256"]
        )
        if dst.exists() and not dst.is_file():
            problems.append(f"{dst}: destination is not a regular file")
        elif dst_hash not in {None, entry["installed_sha256"], entry.get("original_sha256")}:
            problems.append(f"{dst}: file changed since installation; refusing to delete it")
        if backup is not None and not already_restored:
            if backup_hash is None:
                problems.append(f"{backup}: original backup is missing")
            elif backup_hash != entry["original_sha256"]:
                problems.append(f"{backup}: original backup hash does not match the manifest")
        actions.append(
            {
                "entry": entry,
                "dst": dst,
                "backup": backup,
                "installed_present": dst_hash == entry["installed_sha256"],
                "already_restored": already_restored,
                "copy_restored": copy_restored,
            }
        )

    launcher = manifest.get("launcher")
    launcher_path = None
    if launcher:
        launcher_path = _child_path(game_dir, launcher["path"], LAUNCHER_NAME)
        if launcher_path.exists():
            if not launcher_path.is_file() or _sha256(launcher_path) != launcher["sha256"]:
                problems.append(
                    f"{launcher_path}: launcher changed since installation; refusing to delete it"
                )

    for recovery in manifest.get("recovery_files", []):
        recovery_path = _child_path(game_dir, recovery["path"])
        if recovery_path.exists() and (
            not recovery_path.is_file() or _sha256(recovery_path) != recovery["sha256"]
        ):
            problems.append(f"{recovery_path}: recovery file hash mismatch")

    if problems:
        return {
            "removed": removed,
            "restored": restored,
            "problems": problems,
            "manifest": str(manifest_path(game_dir)),
            "manifest_retained": True,
        }

    manifest["state"] = "uninstalling"
    manifest.pop("install_error", None)
    manifest.pop("rollback_problems", None)
    _update_manifest(manifest_path(game_dir), manifest)

    moved: list[dict] = []
    quarantine: list[tuple[Path, str]] = []
    token = uuid.uuid4().hex
    try:
        for action in actions:
            dst, backup = action["dst"], action["backup"]
            record = {"dst": dst, "backup": backup, "quarantine": None, "restored": False}
            moved.append(record)
            if action["installed_present"]:
                temporary = game_dir / f".apitrace-mcp-remove-{token}-{action['entry']['dll']}"
                _require_hash(
                    dst,
                    action["entry"]["installed_sha256"],
                    "quarantining installed wrapper",
                )
                _rename_no_replace(dst, temporary)
                record["quarantine"] = temporary
                quarantine.append((temporary, action["entry"]["installed_sha256"]))
                _require_hash(
                    temporary,
                    action["entry"]["installed_sha256"],
                    "verifying quarantined wrapper",
                )
                removed.append(str(dst))
            if action["copy_restored"]:
                _require_hash(
                    dst,
                    action["entry"]["original_sha256"],
                    "verifying copy-restored original DLL",
                )
                _require_hash(
                    backup,
                    action["entry"]["original_sha256"],
                    "removing redundant original backup",
                )
                backup.unlink()
                restored.append(str(dst))
            elif backup is not None and backup.exists():
                _require_hash(
                    backup,
                    action["entry"]["original_sha256"],
                    "restoring original DLL",
                )
                _rename_no_replace(backup, dst)
                record["restored"] = True
                _require_hash(
                    dst,
                    action["entry"]["original_sha256"],
                    "verifying restored original DLL",
                )
                restored.append(str(dst))

        if launcher_path is not None and launcher_path.exists():
            temporary = game_dir / f".apitrace-mcp-remove-{token}-{LAUNCHER_NAME}"
            _require_hash(launcher_path, launcher["sha256"], "quarantining launcher")
            _rename_no_replace(launcher_path, temporary)
            quarantine.append((temporary, launcher["sha256"]))
            _require_hash(temporary, launcher["sha256"], "verifying quarantined launcher")
            removed.append(str(launcher_path))

        for recovery in manifest.get("recovery_files", []):
            recovery_path = _child_path(game_dir, recovery["path"])
            if recovery_path.exists():
                _require_hash(
                    recovery_path, recovery["sha256"], "removing prior recovery file"
                )
                recovery_path.unlink()
                removed.append(str(recovery_path))

        cleanup_problems: list[str] = []
        retained: list[dict] = []
        for temporary, expected_hash in quarantine:
            try:
                _require_hash(temporary, expected_hash, "removing quarantined wrapper")
                temporary.unlink()
            except (OSError, RuntimeError) as exc:
                cleanup_problems.append(f"{temporary}: {exc}")
                retained.append({"path": temporary.name, "sha256": expected_hash})

        if cleanup_problems:
            manifest["state"] = "cleanup_required"
            manifest["recovery_files"] = retained
            _update_manifest(manifest_path(game_dir), manifest)
            return {
                "removed": removed,
                "restored": restored,
                "problems": cleanup_problems,
                "manifest": str(manifest_path(game_dir)),
                "manifest_retained": True,
            }

        manifest["state"] = "uninstalled"
        manifest["recovery_files"] = []
        _update_manifest(manifest_path(game_dir), manifest)
        try:
            _require_manifest_identity(
                game_dir, manifest["transaction_id"], "uninstalled"
            )
            manifest_path(game_dir).unlink()
        except (OSError, RuntimeError) as exc:
            return {
                "removed": removed,
                "restored": restored,
                "problems": [f"{manifest_path(game_dir)}: {exc}"],
                "manifest": str(manifest_path(game_dir)),
                "manifest_retained": True,
            }
    except Exception as exc:
        rollback_problems: list[str] = []
        for record in reversed(moved):
            try:
                if record["restored"] and record["dst"].exists():
                    original_hash = next(
                        action["entry"]["original_sha256"]
                        for action in actions
                        if action["dst"] == record["dst"]
                    )
                    _require_hash(
                        record["dst"], original_hash, "rolling back restored original DLL"
                    )
                    _rename_no_replace(record["dst"], record["backup"])
                if record["quarantine"] is not None and record["quarantine"].exists():
                    installed_hash = next(
                        action["entry"]["installed_sha256"]
                        for action in actions
                        if action["dst"] == record["dst"]
                    )
                    _require_hash(
                        record["quarantine"],
                        installed_hash,
                        "rolling back quarantined wrapper",
                    )
                    _rename_no_replace(record["quarantine"], record["dst"])
            except (OSError, RuntimeError) as rollback_exc:
                rollback_problems.append(f"{record['dst']}: {rollback_exc}")
        manifest["state"] = "recovery_required"
        manifest["uninstall_error"] = str(exc)
        manifest["rollback_problems"] = rollback_problems
        prior_recovery = [
            recovery
            for recovery in manifest.get("recovery_files", [])
            if os.path.lexists(game_dir / recovery["path"])
        ]
        new_recovery = [
            {"path": path.name, "sha256": expected_hash}
            for path, expected_hash in quarantine
            if path.exists()
        ]
        by_path = {
            recovery["path"]: recovery for recovery in [*prior_recovery, *new_recovery]
        }
        manifest["recovery_files"] = list(by_path.values())
        try:
            _update_manifest(manifest_path(game_dir), manifest)
        except OSError:
            pass
        return {
            "removed": removed,
            "restored": restored,
            "problems": [str(exc), *rollback_problems],
            "manifest": str(manifest_path(game_dir)),
            "manifest_retained": True,
        }

    return {
        "removed": removed,
        "restored": restored,
        "problems": [],
        "manifest_retained": False,
    }


def install(
    root: ApitraceRoot,
    game_exe: str | Path,
    api: str,
    *,
    trace_file: str | Path | None = None,
    force: bool = False,
) -> dict:
    exe = Path(game_exe).expanduser().resolve()
    game_dir = exe.parent
    _check_target_dir(game_dir)
    with _transaction_lock(game_dir):
        return _install(
            root, exe, api, trace_file=trace_file, force=force
        )


def uninstall(game_exe_or_dir: str | Path) -> dict:
    path = Path(game_exe_or_dir).expanduser()
    game_dir = (path if path.is_dir() else path.parent).resolve()
    _check_target_dir(game_dir)
    # Probe before taking the lock so a nothing-to-undo answer never creates
    # files in a directory we were merely pointed at; _uninstall re-checks the
    # manifest under the lock before doing real work.
    if read_manifest(game_dir) is None:
        return {
            "removed": [],
            "restored": [],
            "note": f"no apitrace-mcp install manifest found in {game_dir}; nothing to undo",
        }
    with _transaction_lock(game_dir):
        return _uninstall(game_dir)
