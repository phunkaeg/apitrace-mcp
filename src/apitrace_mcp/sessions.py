"""Background trace/replay processes with restart-safe identity checks."""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import subprocess
import threading
import time
import uuid
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:  # pragma: no cover - this module only manages Windows processes
    msvcrt = None

from .runner import clamp

CREATE_NO_WINDOW = 0x08000000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


@dataclass(frozen=True)
class ProcessIdentity:
    """Values that survive serialization and disambiguate recycled PIDs."""

    created_filetime: int
    executable: str


def _normalise_executable(path: str) -> str:
    if path.startswith("\\\\?\\"):
        path = path[4:]
    return os.path.normcase(os.path.abspath(path))


def _query_process_identity(pid: int) -> ProcessIdentity | None:
    """Return identity only for a live, queryable Windows process."""
    if os.name != "nt" or not isinstance(pid, int) or pid <= 0:
        return None
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return None
        if code.value != _STILL_ACTIVE:
            return None

        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None

        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None
        created_value = (created.dwHighDateTime << 32) | created.dwLowDateTime
        return ProcessIdentity(created_value, _normalise_executable(buffer.value[: size.value]))
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive(pid: int) -> bool:
    return _query_process_identity(pid) is not None


def _identity_matches(
    current: ProcessIdentity,
    created_filetime: int | None,
    executable: str | None,
) -> bool:
    return (
        isinstance(created_filetime, int)
        and created_filetime > 0
        and isinstance(executable, str)
        and bool(executable)
        and current.created_filetime == created_filetime
        and current.executable == _normalise_executable(executable)
    )


@dataclass
class Session:
    id: str
    kind: str  # trace | replay
    cmd: list[str]
    pid: int
    log_path: Path
    proc: subprocess.Popen | None = None
    log_handle: object | None = None
    trace_path: Path | None = None
    started_at: float = field(default_factory=time.time)
    stopped_at: float | None = None
    stop_note: str | None = None
    detached: bool = False
    process_created_filetime: int | None = None
    process_executable: str | None = None
    final_exit_code: int | None = None
    stopping: bool = False
    persist_warning: str | None = None
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )

    def _close_log_handle(self) -> None:
        with self._lock:
            if self.log_handle is None:
                return
            try:
                self.log_handle.close()  # type: ignore[union-attr]
            except OSError:
                pass
            finally:
                self.log_handle = None

    def _finalize_if_exited(self) -> int | None:
        with self._lock:
            if self.proc is None:
                return self.final_exit_code
            code = self.proc.poll()
            if code is None:
                return None
            try:
                self.proc.wait(timeout=0)
            except (OSError, subprocess.TimeoutExpired):
                pass
            self.final_exit_code = code
            # Dropping the Popen object releases its Windows process/thread handles.
            self.proc = None
            self._close_log_handle()
            return code

    @property
    def running(self) -> bool:
        with self._lock:
            if self.final_exit_code is not None:
                return False
            if self.proc is not None:
                return self._finalize_if_exited() is None
            if not self.detached:
                return False
            current = _query_process_identity(self.pid)
            return current is not None and _identity_matches(
                current, self.process_created_filetime, self.process_executable
            )

    @property
    def exit_code(self) -> int | None:
        with self._lock:
            self._finalize_if_exited()
            return self.final_exit_code

    def log_tail(self, max_bytes: int = 4000) -> str:
        with self._lock:
            max_bytes = max(0, int(max_bytes))
            if max_bytes == 0:
                return ""
            if self.log_handle is not None:
                try:
                    self.log_handle.flush()  # type: ignore[union-attr]
                except OSError:
                    pass
            try:
                with self.log_path.open("rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - max_bytes), os.SEEK_SET)
                    data = handle.read(max_bytes)
            except OSError:
                return ""
            return data.decode("utf-8", errors="replace")

    def status(self, tail_bytes: int = 4000) -> dict:
        with self._lock:
            running = self.running
            info: dict = {
                "session": self.id,
                "kind": self.kind,
                "running": running,
                "pid": self.pid,
                "exit_code": self.exit_code,
                "elapsed_s": round(
                    (self.stopped_at or time.time()) - self.started_at, 1
                ),
                "command": subprocess.list2cmdline(self.cmd),
                "log": str(self.log_path),
                "log_tail": clamp(self.log_tail(tail_bytes), tail_bytes)[0],
            }
            if self.detached:
                info["detached"] = (
                    "reloaded from disk after a server restart; PID, creation time, and "
                    "executable identity are verified before it can be stopped"
                )
            if self.stopping:
                info["stopping"] = True
            if self.stop_note:
                info["stop_note"] = self.stop_note
            if self.persist_warning:
                info["persist_warning"] = self.persist_warning
            if self.trace_path:
                info["trace"] = str(self.trace_path)
                if self.trace_path.is_file():
                    size = self.trace_path.stat().st_size
                    info["trace_bytes"] = size
                    info["trace_mb"] = round(size / (1024 * 1024), 2)
                else:
                    info["trace_exists"] = False
                    if not running:
                        info["warning"] = (
                            "process exited but no trace file was produced -- the wrapper "
                            "probably never loaded. Check the log, confirm the API with "
                            "detect_target, and try method='mhook' or the DLL-drop route "
                            "(install_wrapper) if the game uses a launcher."
                        )
            return info

    def to_record(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "cmd": self.cmd,
                "pid": self.pid,
                "log_path": str(self.log_path),
                "trace_path": str(self.trace_path) if self.trace_path else None,
                "started_at": self.started_at,
                "process_created_filetime": self.process_created_filetime,
                "process_executable": self.process_executable,
            }

    @classmethod
    def from_record(cls, rec: dict) -> "Session":
        required = {
            "id": str,
            "kind": str,
            "cmd": list,
            "pid": int,
            "log_path": str,
            "started_at": (int, float),
            "process_created_filetime": int,
            "process_executable": str,
        }
        for key, expected_type in required.items():
            if not isinstance(rec.get(key), expected_type):
                raise TypeError(f"invalid session record field {key}")
        if not rec["cmd"] or not all(isinstance(arg, str) for arg in rec["cmd"]):
            raise TypeError("invalid session command")
        if rec["pid"] <= 0 or rec["process_created_filetime"] <= 0:
            raise TypeError("invalid process identity")
        trace_value = rec.get("trace_path")
        if trace_value is not None and not isinstance(trace_value, str):
            raise TypeError("invalid trace path")
        return cls(
            id=rec["id"],
            kind=rec["kind"],
            cmd=list(rec["cmd"]),
            pid=rec["pid"],
            log_path=Path(rec["log_path"]),
            trace_path=Path(trace_value) if trace_value else None,
            started_at=float(rec["started_at"]),
            detached=True,
            process_created_filetime=rec["process_created_filetime"],
            process_executable=_normalise_executable(rec["process_executable"]),
        )


class SessionManager:
    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self._store: Path | None = None
        self._lock = threading.RLock()

    # -- persistence -------------------------------------------------------

    def _load(self, log_dir: Path) -> None:
        with self._lock:
            self._load_locked(log_dir)

    def _load_locked(self, log_dir: Path) -> None:
        self._store = log_dir / "sessions.json"
        if not self._store.is_file():
            return
        try:
            records = json.loads(self._store.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        if not isinstance(records, list):
            return
        for rec in records:
            if not isinstance(rec, dict) or rec.get("id") in self.sessions:
                continue
            try:
                session = Session.from_record(rec)
            except (KeyError, TypeError, ValueError):
                continue
            if session.running:
                self.sessions[session.id] = session

    def _save(self) -> None:
        with self._lock:
            self._save_locked()

    @contextlib.contextmanager
    def _store_lock(self):
        """Serialize load-merge-save cycles across server processes.

        Four hosts run this server concurrently against the same store, so a
        pure snapshot write from one instance would erase another's live
        sessions. A ``sessions.json.lock`` byte lock makes each
        read-merge-write cycle atomic between processes.
        """
        if self._store is None or msvcrt is None:
            yield
            return
        lock_path = self._store.with_name(self._store.name + ".lock")
        try:
            handle = lock_path.open("a+b")
        except OSError:
            # A missing lock file must not block persistence entirely; the
            # merge in _save_locked still limits the damage.
            yield
            return
        try:
            deadline = time.monotonic() + 10.0
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"timed out waiting for session store lock {lock_path}"
                        ) from exc
                    time.sleep(0.05)
            try:
                yield
            finally:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        finally:
            handle.close()

    def _merged_records(self) -> list[dict]:
        """Local live sessions plus other instances' still-alive records.

        Never write a bare snapshot of local memory: records owned by another
        server instance (id unknown here, identity check still passing) must
        be carried forward or that instance's running game becomes unlisted
        and unstoppable after a restart.
        """
        live = [s.to_record() for s in self.sessions.values() if s.running]
        # Local ids -- including stopped ones we intend to drop from disk.
        known = set(self.sessions)
        try:
            on_disk = json.loads(self._store.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            on_disk = []
        if isinstance(on_disk, list):
            for rec in on_disk:
                if not isinstance(rec, dict) or rec.get("id") in known:
                    continue
                try:
                    foreign = Session.from_record(rec)
                except (KeyError, TypeError, ValueError):
                    continue
                if foreign.running:
                    live.append(rec)
        return live

    def _replace_with_retry(self, temporary: Path) -> None:
        """os.replace, retrying transient Windows sharing violations.

        Another instance reading sessions.json holds it without
        FILE_SHARE_DELETE, so the rename can fail with winerror 5/32 for a
        few milliseconds; that must not be treated as a hard failure.
        """
        last_error: OSError | None = None
        for attempt in range(10):
            if attempt:
                time.sleep(0.05)
            try:
                os.replace(temporary, self._store)
                return
            except OSError as exc:
                if getattr(exc, "winerror", None) not in (5, 32):
                    raise
                last_error = exc
        raise last_error  # type: ignore[misc]

    def _save_locked(self) -> None:
        if self._store is None:
            return
        try:
            self._store.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"cannot create session store directory: {exc}") from exc

        with self._store_lock():
            live = self._merged_records()

            temporary: Path | None = None
            handle = None
            for _ in range(100):
                candidate = self._store.with_name(
                    f".{self._store.name}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    handle = candidate.open("x", encoding="utf-8", newline="\n")
                except FileExistsError:
                    continue
                except OSError as exc:
                    raise RuntimeError(
                        f"cannot create temporary session store beside {self._store}: {exc}"
                    ) from exc
                temporary = candidate
                break
            if temporary is None or handle is None:
                raise RuntimeError(
                    f"cannot allocate a unique session store beside {self._store}"
                )
            try:
                with handle:
                    json.dump(live, handle, indent=2)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self._replace_with_retry(temporary)
            except OSError as exc:
                raise RuntimeError(
                    f"cannot durably save sessions to {self._store}: {exc}"
                ) from exc
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

    # -- lifecycle ---------------------------------------------------------

    def start(
        self,
        kind: str,
        cmd: list[str],
        *,
        log_dir: Path,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        trace_path: Path | None = None,
    ) -> Session:
        if not kind or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in kind):
            raise ValueError("session kind must contain only lowercase letters, digits, '_' or '-'")
        if not cmd or not all(isinstance(arg, str) and "\0" not in arg for arg in cmd):
            raise ValueError("session command must be a non-empty list of strings")
        with self._lock:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(f"cannot create session log directory {log_dir}: {exc}") from exc
            self._load_locked(log_dir)
            while True:
                sid = f"{kind}-{uuid.uuid4().hex[:12]}"
                log_path = log_dir / f"{sid}.log"
                if sid not in self.sessions and not log_path.exists():
                    break
            try:
                handle = log_path.open("xb")
            except OSError as exc:
                raise RuntimeError(f"cannot create session log {log_path}: {exc}") from exc
            full_env = {**os.environ, **(env or {})}
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    cwd=str(cwd) if cwd else None,
                    env=full_env,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            except OSError as exc:
                handle.close()
                if isinstance(exc, FileNotFoundError):
                    raise RuntimeError(f"executable not found: {cmd[0]}") from exc
                raise RuntimeError(f"could not start {cmd[0]}: {exc}") from exc

            identity = self._identity_after_start(proc)
            if identity is None:
                self._terminate_owned_process(proc, None)
                handle.close()
                raise RuntimeError(
                    f"started {cmd[0]} but could not obtain a durable process identity; "
                    "the new process was terminated and no session was created"
                )

            session = Session(
                id=sid,
                kind=kind,
                cmd=list(cmd),
                pid=proc.pid,
                log_path=log_path,
                proc=proc,
                log_handle=handle,
                trace_path=trace_path,
                process_created_filetime=identity.created_filetime,
                process_executable=identity.executable,
            )
            self.sessions[sid] = session
            try:
                self._save_locked()
            except RuntimeError as exc:
                # A transient persistence failure must not kill a healthy
                # launch: keep the session in memory and surface the gap.
                session.persist_warning = (
                    "session live but not persisted; it will not survive a "
                    f"server restart: {exc}"
                )
            return session

    @staticmethod
    def _identity_after_start(proc: subprocess.Popen) -> ProcessIdentity | None:
        for attempt in range(20):
            identity = _query_process_identity(proc.pid)
            if identity is not None:
                return identity
            if proc.poll() is not None:
                return None
            if attempt != 19:
                time.sleep(0.025)
        return None

    @staticmethod
    def _terminate_owned_process(
        proc: subprocess.Popen, identity: ProcessIdentity | None
    ) -> None:
        """Clean up a failed start, including children when identity is known."""
        if identity is not None:
            current = _query_process_identity(proc.pid)
            if current is not None and _identity_matches(
                current, identity.created_filetime, identity.executable
            ):
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        capture_output=True,
                        creationflags=CREATE_NO_WINDOW,
                        check=False,
                    )
                except OSError:
                    pass
                try:
                    proc.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass

        # The retained Popen handle is immune to PID reuse and is the safe
        # fallback when identity was unavailable or tree termination failed.
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        except OSError:
            # The process may have exited between poll and TerminateProcess.
            pass

    def ensure_loaded(self, log_dir: Path) -> None:
        """Pick up sessions from a previous server run before touching them."""
        with self._lock:
            if self._store is None:
                self._load_locked(log_dir)

    def get(self, sid: str) -> Session:
        with self._lock:
            session = self.sessions.get(sid)
            if session is None and self._store is not None:
                self._load_locked(self._store.parent)
                session = self.sessions.get(sid)
            if session is None:
                known = ", ".join(self.sessions) or "none"
                raise KeyError(f"unknown session {sid!r} (active: {known})")
            return session

    @staticmethod
    def _verify_for_kill(session: Session) -> bool:
        """Verify PID identity immediately before any taskkill invocation.

        This applies to sessions with a live Popen handle too: the process can
        exit after ``poll()``, and Windows can recycle its PID before taskkill.
        """
        current = _query_process_identity(session.pid)
        if current is None:
            if session.proc is not None and session.proc.poll() is None:
                raise RuntimeError(
                    f"refusing to stop session {session.id}: live process identity "
                    "could not be verified"
                )
            return False
        if not _identity_matches(
            current, session.process_created_filetime, session.process_executable
        ):
            raise RuntimeError(
                f"refusing to stop session {session.id}: PID {session.pid} no "
                "longer has the recorded creation time and executable identity"
            )
        return True

    def stop(self, sid: str, force: bool = False) -> dict:
        with self._lock:
            session = self.get(sid)
            with session._lock:
                if session.stopping:
                    raise RuntimeError(f"session {sid} is already being stopped")
                session.stopping = True
        persist_error: RuntimeError | None = None
        try:
            result = self._stop_session(session, force)
        finally:
            with session._lock:
                session.stopping = False
            # Surface persistence failures without replacing the stop outcome:
            # a successful stop must stay reported as such.
            try:
                self._save()
            except RuntimeError as exc:
                persist_error = exc
        if persist_error is not None:
            result["persist_warning"] = (
                f"stop completed but the session store could not be updated: {persist_error}"
            )
        return result

    def _stop_session(self, session: Session, force: bool) -> dict:
        if not session.running:
            # Detached ``running`` is false for both a gone process and a
            # recycled PID; surface the latter instead of silently treating it
            # as a clean exit.
            if session.detached:
                self._verify_for_kill(session)
            return session.status()
        if not self._verify_for_kill(session):
            return session.status()

        # apitrace spawns the game as a child, so the whole tree has to go.
        base = ["taskkill", "/PID", str(session.pid), "/T"]
        subprocess.run(
            base + (["/F"] if force else []),
            capture_output=True,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )

        for _ in range(50):
            if not session.running:
                break
            time.sleep(0.1)

        if session.running and not force:
            if not self._verify_for_kill(session):
                return session.status()
            subprocess.run(
                base + ["/F"],
                capture_output=True,
                creationflags=CREATE_NO_WINDOW,
                check=False,
            )
            session.stop_note = (
                "graceful close did not take, process tree was force-killed; the "
                "final frames of the trace may be missing"
            )
            for _ in range(50):
                if not session.running:
                    break
                time.sleep(0.1)
        elif force:
            session.stop_note = (
                "force-killed; apitrace may not have flushed the tail of the trace"
            )

        if session.running:
            session.stop_note = "taskkill returned but the verified process is still running"
        else:
            with session._lock:
                session.stopped_at = time.time()
            session._finalize_if_exited()
            session._close_log_handle()
        return session.status()

    def list(self, log_dir: Path | None = None) -> list[dict]:
        with self._lock:
            if log_dir is not None:
                self._load_locked(log_dir)
            return [s.status(tail_bytes=400) for s in self.sessions.values()]


MANAGER = SessionManager()
