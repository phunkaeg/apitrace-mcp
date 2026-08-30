"""Subprocess plumbing.

Two hard rules live here:

1. Nothing blocks the MCP server for long. Anything that runs a game (trace,
   replay) goes through `sessions.py` instead of `run()`.
2. Nothing dumps an unbounded amount of a multi-gigabyte trace into the reply.
   Every helper takes a byte cap and says so when it truncates.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import BinaryIO

DEFAULT_TIMEOUT = 120
DEFAULT_MAX_BYTES = 60_000
DEFAULT_MAX_STDERR_BYTES = 60_000

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
PROCESS_CREATION_FLAGS = (
    CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
)


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    class _JobBasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobBasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


class _WindowsJob:
    """A kill-on-close Job Object containing one foreground command tree."""

    def __init__(self) -> None:
        self._handle = None
        if os.name != "nt":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.UINT,
        ]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            kernel32.CloseHandle(handle)
            return
        self._kernel32 = kernel32
        self._handle = handle

    def assign(self, pid: int) -> bool:
        if self._handle is None:
            return False
        access = 0x0001 | 0x0100 | 0x1000  # TERMINATE | SET_QUOTA | QUERY_LIMITED
        process = self._kernel32.OpenProcess(access, False, pid)
        if not process:
            return False
        try:
            return bool(
                self._kernel32.AssignProcessToJobObject(self._handle, process)
            )
        finally:
            self._kernel32.CloseHandle(process)

    def terminate(self) -> None:
        if self._handle is not None:
            self._kernel32.TerminateJobObject(self._handle, 1)
            self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


class CommandError(RuntimeError):
    def __init__(self, cmd: list[str], returncode: int, stderr: str):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"{Path(cmd[0]).name} {' '.join(cmd[1:3])} failed (exit {returncode}): "
            f"{stderr.strip()[:2000] or '<no stderr>'}"
        )


def _terminate_process_tree(
    proc: subprocess.Popen, job: _WindowsJob | None = None
) -> None:
    """Best-effort hard kill of a command and all descendants."""
    if job is not None:
        job.terminate()
    if os.name == "nt":
        try:
            # /T is essential: apitrace's Python dispatchers and retracers can
            # otherwise survive their immediate apitrace.exe parent.
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                creationflags=CREATE_NO_WINDOW,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _join_readers(readers: list[threading.Thread], timeout: float = 5.0) -> None:
    """Join pipe drainers for a bounded time, then give up.

    A leaked descendant can hold a pipe's write end open, leaving a reader
    blocked inside a kernel read.  Closing the stream from this thread would
    block on the same BufferedReader lock the reader holds, so a stuck reader
    is simply abandoned: the threads are daemons and their bytearrays remain
    safe to snapshot (``bytes(...)`` is atomic under the GIL).
    """
    deadline = time.monotonic() + timeout
    for reader in readers:
        reader.join(timeout=max(0.0, deadline - time.monotonic()))


def clamp(text: str, max_bytes: int = DEFAULT_MAX_BYTES) -> tuple[str, bool]:
    """Trim to a byte budget on a line boundary. Returns (text, truncated)."""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text, False
    cut = raw[:max_bytes].decode("utf-8", errors="ignore")
    nl = cut.rfind("\n")
    if nl > max_bytes // 2:
        cut = cut[:nl]
    return cut, True


def _read_bounded(
    stream: BinaryIO,
    limit: int,
    output: bytearray,
    state: dict[str, bool],
    *,
    keep_tail: bool = False,
) -> None:
    """Drain a pipe while retaining at most ``limit`` bytes.

    Draining stdout and stderr concurrently prevents the child from blocking on
    a full OS pipe.  Stdout keeps its prefix (needed for JSON/text parsing),
    while stderr keeps its tail (where command-line tools put the useful error).
    """
    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                break
            if keep_tail:
                output.extend(chunk)
                if len(output) > limit:
                    del output[: len(output) - limit]
                    state["truncated"] = True
            else:
                remaining = limit - len(output)
                if remaining > 0:
                    output.extend(chunk[:remaining])
                if len(chunk) > max(remaining, 0):
                    state["truncated"] = True
    except (OSError, ValueError):
        # Closing/killing a process can invalidate a reader on Windows.  The
        # process result remains authoritative.
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def run(
    cmd: list[str],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES,
) -> tuple[str, str, bool]:
    """Run to completion with bounded streaming capture.

    Returns ``(stdout, stderr, stdout_truncated)``.  Output is drained while the
    process runs, so memory use never scales with command output volume.
    """
    if max_bytes < 0 or max_stderr_bytes < 0:
        raise ValueError("output byte limits must be non-negative")
    job = _WindowsJob() if os.name == "nt" else None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd) if cwd else None,
            env=env,
            creationflags=PROCESS_CREATION_FLAGS,
            start_new_session=os.name != "nt",
        )
    except FileNotFoundError as exc:
        if job is not None:
            job.close()
        raise RuntimeError(f"executable not found: {cmd[0]}") from exc
    except OSError:
        if job is not None:
            job.close()
        raise
    if job is not None and not job.assign(proc.pid):
        job.close()
        job = None

    assert proc.stdout is not None and proc.stderr is not None
    stdout_raw = bytearray()
    stderr_raw = bytearray()
    stdout_state = {"truncated": False}
    stderr_state = {"truncated": False}
    readers = [
        threading.Thread(
            target=_read_bounded,
            args=(proc.stdout, max_bytes, stdout_raw, stdout_state),
            daemon=True,
        ),
        threading.Thread(
            target=_read_bounded,
            args=(proc.stderr, max_stderr_bytes, stderr_raw, stderr_state),
            kwargs={"keep_tail": True},
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(proc, job)
        job = None
        _join_readers(readers)
        raise RuntimeError(
            f"{Path(cmd[0]).name} timed out after {timeout}s. Large traces are slow -- "
            "narrow the call range (calls=) or raise timeout."
        ) from exc
    if job is not None:
        # A dispatcher that exited without waiting for descendants must not
        # leak them or leave their inherited pipe handles open.
        job.close()
    _join_readers(readers)

    stdout = bytes(stdout_raw).decode("utf-8", errors="replace")
    stderr = bytes(stderr_raw).decode("utf-8", errors="replace")
    if stderr_state["truncated"]:
        stderr = "<stderr truncated; showing tail>\n" + stderr
    if check and proc.returncode != 0:
        raise CommandError(cmd, proc.returncode, stderr)
    text, line_truncated = clamp(stdout, max_bytes)
    return text, stderr, stdout_state["truncated"] or line_truncated


def popen_pipe(
    cmd: list[str], *, cwd: str | Path | None = None
) -> subprocess.Popen:
    """Start a process with a binary stdout pipe (used for `apitrace pickle`)."""
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        # pickleparse drains this concurrently and uses it for truthful errors.
        stderr=subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
        creationflags=PROCESS_CREATION_FLAGS,
        start_new_session=os.name != "nt",
    )
