"""Structured access to a trace via `apitrace pickle`.

`apitrace dump` gives text you have to scrape; `apitrace pickle` gives the real
call model -- every argument with its actual numeric value. That is what makes
matrix and constant-register analysis possible, so it is the backbone of this
server.

Wire format (see apitrace's own lib/scripts/unpickle.py): a concatenation of
pickled tuples, one per call:

    (callNo, threadId, functionName, [(argName, value), ...], retValue, flags)

Pointers arrive as a `unpickle.Pointer` global. Traces are untrusted binary
input, so unpickling is restricted to exactly that one class -- a stock
`pickle.load` on a hostile trace would be an arbitrary-code-execution hole.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import signal
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from .config import ApitraceRoot
from .runner import CREATE_NO_WINDOW, CommandError

# Mirrors trace_model.hpp / unpickle.py.
CALL_FLAG_FAKE = 1 << 0
CALL_FLAG_NON_REPRODUCIBLE = 1 << 1
CALL_FLAG_NO_SIDE_EFFECTS = 1 << 2
CALL_FLAG_RENDER = 1 << 3
CALL_FLAG_SWAP_RENDERTARGET = 1 << 4
CALL_FLAG_END_FRAME = 1 << 5
CALL_FLAG_INCOMPLETE = 1 << 6
CALL_FLAG_VERBOSE = 1 << 7
CALL_FLAG_MARKER = 1 << 8
CALL_FLAG_MARKER_PUSH = 1 << 9
CALL_FLAG_MARKER_POP = 1 << 10

FLAG_NAMES = [
    (CALL_FLAG_FAKE, "fake"),
    (CALL_FLAG_NON_REPRODUCIBLE, "non_reproducible"),
    (CALL_FLAG_NO_SIDE_EFFECTS, "no_side_effects"),
    (CALL_FLAG_RENDER, "render"),
    (CALL_FLAG_SWAP_RENDERTARGET, "swap_rendertarget"),
    (CALL_FLAG_END_FRAME, "end_frame"),
    (CALL_FLAG_INCOMPLETE, "incomplete"),
    (CALL_FLAG_VERBOSE, "verbose"),
    (CALL_FLAG_MARKER, "marker"),
    (CALL_FLAG_MARKER_PUSH, "marker_push"),
    (CALL_FLAG_MARKER_POP, "marker_pop"),
]

DEFAULT_PICKLE_TIMEOUT = 600
MAX_PICKLE_STDERR = 64 * 1024
DEFAULT_WORKER_MEMORY_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_FRAME_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_RECORD_NODES = 1_000_000
DEFAULT_MAX_RECORD_DEPTH = 64
DEFAULT_MAX_RECORD_BLOB_BYTES = 48 * 1024 * 1024
DEFAULT_MAX_RECORD_STRING_CHARS = 4 * 1024 * 1024
DEFAULT_MAX_RECORD_FLOATS = 4_000_000
CREATE_NEW_PROCESS_GROUP = 0x00000200
PROCESS_FLAGS = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0


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


class _DecoderJob:
    """Kill-on-close worker Job Object with hard process/job memory limits."""

    def __init__(self, memory_bytes: int) -> None:
        self._lock = threading.Lock()
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
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise RuntimeError("could not create memory-limited pickle decoder job")
        limits = _ExtendedLimits()
        # PROCESS_MEMORY | JOB_MEMORY | KILL_ON_JOB_CLOSE
        limits.BasicLimitInformation.LimitFlags = 0x00000100 | 0x00000200 | 0x00002000
        limits.ProcessMemoryLimit = memory_bytes
        limits.JobMemoryLimit = memory_bytes
        if not kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise RuntimeError(f"could not set pickle decoder job limits (WinError {error})")
        self._kernel32 = kernel32
        self._handle = handle

    def assign(self, pid: int) -> None:
        if os.name != "nt":
            return
        access = 0x0001 | 0x0100 | 0x1000
        self._kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        process = self._kernel32.OpenProcess(access, False, pid)
        if not process:
            raise RuntimeError("could not open pickle decoder worker for job assignment")
        try:
            self._kernel32.AssignProcessToJobObject.argtypes = [
                wintypes.HANDLE,
                wintypes.HANDLE,
            ]
            self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
            if not self._kernel32.AssignProcessToJobObject(self._handle, process):
                error = ctypes.get_last_error()
                raise RuntimeError(
                    f"could not memory-limit pickle decoder worker (WinError {error})"
                )
        finally:
            self._kernel32.CloseHandle(process)

    def terminate(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._kernel32.TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._kernel32.CloseHandle(self._handle)
                self._handle = None


class Pointer(int):
    def __str__(self) -> str:
        return "NULL" if self == 0 else hex(self)

    __repr__ = __str__


class _RestrictedUnpickler(pickle.Unpickler):
    """Allows only apitrace's Pointer marker class through."""

    def find_class(self, module: str, name: str) -> Any:
        if name == "Pointer" and module.rsplit(".", 1)[-1] in ("unpickle", "apitrace"):
            return Pointer
        raise pickle.UnpicklingError(
            f"refusing to unpickle {module}.{name} from trace data"
        )


@dataclass(frozen=True)
class _RecordLimits:
    max_nodes: int = DEFAULT_MAX_RECORD_NODES
    max_depth: int = DEFAULT_MAX_RECORD_DEPTH
    max_blob_bytes: int = DEFAULT_MAX_RECORD_BLOB_BYTES
    max_string_chars: int = DEFAULT_MAX_RECORD_STRING_CHARS
    max_floats: int = DEFAULT_MAX_RECORD_FLOATS


def _children(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield item
    else:
        yield from value


def _validate_record_graph(root: Any, limits: _RecordLimits) -> None:
    """Validate without copying containers; reject cycles and resource bombs."""
    nodes = blob_bytes = string_chars = floats = 0
    active: set[int] = set()
    stack: list[tuple[str, Any, int]] = [("value", root, 0)]
    while stack:
        kind, value, depth = stack.pop()
        if kind == "exit":
            active.discard(value)
            continue
        if kind == "iterator":
            try:
                child = next(value)
            except StopIteration:
                continue
            stack.append(("iterator", value, depth))
            stack.append(("value", child, depth))
            continue

        nodes += 1
        if nodes > limits.max_nodes:
            raise ValueError(f"record exceeds node budget ({limits.max_nodes})")
        if depth > limits.max_depth:
            raise ValueError(f"record exceeds nesting depth ({limits.max_depth})")
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, Pointer):
            continue
        if isinstance(value, int):
            if value.bit_length() > 4096:
                raise ValueError("record contains an integer wider than 4096 bits")
            continue
        if isinstance(value, float):
            floats += 1
            if floats > limits.max_floats:
                raise ValueError(f"record exceeds float budget ({limits.max_floats})")
            continue
        if isinstance(value, str):
            string_chars += len(value)
            if string_chars > limits.max_string_chars:
                raise ValueError(
                    f"record exceeds string budget ({limits.max_string_chars} characters)"
                )
            continue
        if isinstance(value, (bytes, bytearray)):
            blob_bytes += len(value)
            if blob_bytes > limits.max_blob_bytes:
                raise ValueError(
                    f"record exceeds blob budget ({limits.max_blob_bytes} bytes)"
                )
            continue
        if not isinstance(value, (list, tuple, dict)):
            raise ValueError(f"record contains unsupported {type(value).__name__}")
        identity = id(value)
        if identity in active:
            raise ValueError("record contains a container cycle")
        active.add(identity)
        stack.append(("exit", identity, depth))
        stack.append(("iterator", iter(_children(value)), depth + 1))


def _write_frame(stream: BinaryIO, kind: bytes, payload: bytes = b"") -> None:
    size = 1 + len(payload)
    stream.write(struct.pack(">I", size))
    stream.write(kind)
    if payload:
        stream.write(payload)
    stream.flush()


def _worker_entry() -> None:
    """Decode raw apitrace pickle records in an expendable constrained process."""
    try:
        config = json.loads(sys.argv[1])
        memory_bytes = int(config["memory_bytes"])
        max_frame_bytes = int(config["max_frame_bytes"])
        limits = _RecordLimits(
            max_nodes=int(config["max_nodes"]),
            max_depth=int(config["max_depth"]),
            max_blob_bytes=int(config["max_blob_bytes"]),
            max_string_chars=int(config["max_string_chars"]),
            max_floats=int(config["max_floats"]),
        )
        if os.name != "nt":
            try:
                import resource

                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            except (ImportError, OSError, ValueError):
                pass
        source = sys.stdin.buffer
        destination = sys.stdout.buffer
        unpickler = _RestrictedUnpickler(source)
        while True:
            try:
                record = unpickler.load()
            except EOFError:
                _write_frame(destination, b"D")
                return
            _validate_record_graph(record, limits)
            _record_to_call(record)
            payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
            if len(payload) + 1 > max_frame_bytes:
                raise ValueError(
                    f"decoded record frame exceeds {max_frame_bytes} bytes"
                )
            _write_frame(destination, b"R", payload)
            # Do not retain both the decoded blob graph and serialized frame
            # while blocking for the next input record.
            del payload
            del record
    except BaseException as exc:  # worker must turn malformed streams into a frame
        try:
            message = f"{type(exc).__name__}: {exc}".encode("utf-8", errors="replace")
            _write_frame(sys.stdout.buffer, b"E", message[:64 * 1024])
        except BaseException:
            pass
        raise SystemExit(2)


class _StderrDrain:
    """Drain stderr concurrently while retaining only a bounded diagnostic tail."""

    def __init__(self, pipe: Any, max_bytes: int = MAX_PICKLE_STDERR) -> None:
        self._pipe = pipe
        self._max_bytes = max(1024, max_bytes)
        self._tail = bytearray()
        self._thread = threading.Thread(
            target=self._read,
            name="apitrace-pickle-stderr",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _read(self) -> None:
        try:
            while True:
                chunk = self._pipe.read(8192)
                if not chunk:
                    break
                self._tail.extend(chunk)
                extra = len(self._tail) - self._max_bytes
                if extra > 0:
                    del self._tail[:extra]
        except (OSError, ValueError):
            # The main thread may close pipes while cancelling a partial walk.
            pass
        finally:
            try:
                self._pipe.close()
            except OSError:
                pass

    def finish(self, timeout: float = 5.0) -> str:
        self._thread.join(timeout=timeout)
        return bytes(self._tail).decode("utf-8", errors="replace")


@dataclass
class Call:
    no: int
    thread_id: int | None
    name: str
    args: list[tuple[str, Any]]
    ret: Any
    flags: int

    @property
    def flag_names(self) -> list[str]:
        return [label for bit, label in FLAG_NAMES if self.flags & bit]

    @property
    def is_draw(self) -> bool:
        return bool(self.flags & CALL_FLAG_RENDER)

    @property
    def is_frame_end(self) -> bool:
        return bool(self.flags & CALL_FLAG_END_FRAME)

    def arg(self, name: str, default: Any = None) -> Any:
        for key, value in self.args:
            if key == name:
                return value
        return default

    def arg_at(self, index: int, default: Any = None) -> Any:
        if 0 <= index < len(self.args):
            return self.args[index][1]
        return default


def _record_to_call(record: Any) -> Call:
    """Validate the apitrace pickle schema before exposing a record to callers."""
    if not isinstance(record, (tuple, list)) or len(record) != 6:
        raise ValueError(
            "unexpected pickle record: expected a 6-item tuple/list, got "
            f"{type(record).__name__}"
        )
    no, thread_id, name, args, ret, flags = record
    if isinstance(no, bool) or not isinstance(no, int) or no < 0:
        raise ValueError(f"invalid call number in pickle record: {no!r}")
    if thread_id is not None and (
        isinstance(thread_id, bool) or not isinstance(thread_id, int) or thread_id < 0
    ):
        raise ValueError(f"invalid thread id in call {no}: {thread_id!r}")
    if not isinstance(name, str) or not name:
        raise ValueError(f"invalid function name in call {no}: {name!r}")
    if args is None:
        clean_args: list[tuple[str, Any]] = []
    elif isinstance(args, (tuple, list)):
        for index, arg in enumerate(args):
            if not isinstance(arg, (tuple, list)) or len(arg) != 2:
                raise ValueError(f"invalid argument {index} in call {no}: expected name/value pair")
            arg_name, _ = arg
            if not isinstance(arg_name, str):
                raise ValueError(f"invalid argument name {arg_name!r} in call {no}")
        # Lists are apitrace's normal representation. Reuse their outer and
        # inner containers after validation; for tuple outers make only the one
        # shallow list Call requires, never a new pair per argument.
        clean_args = args if isinstance(args, list) else list(args)  # type: ignore[assignment]
    else:
        raise ValueError(f"invalid argument collection in call {no}: {type(args).__name__}")
    if flags is None:
        clean_flags = 0
    elif isinstance(flags, bool) or not isinstance(flags, int) or flags < 0:
        raise ValueError(f"invalid flags in call {no}: {flags!r}")
    else:
        clean_flags = flags
    return Call(
        no=no,
        thread_id=thread_id,
        name=name,
        args=clean_args,
        ret=ret,
        flags=clean_flags,
    )


def pickle_cmd(
    root: ApitraceRoot,
    trace: str | Path,
    *,
    calls: str | None = None,
    symbolic: bool = True,
) -> list[str]:
    cmd = [str(root.apitrace_exe), "pickle"]
    if symbolic:
        cmd.append("--symbolic")
    if calls:
        cmd.append(f"--calls={calls}")
    cmd.append(str(trace))
    return cmd


def _kill_process_tree(proc: subprocess.Popen, job: _DecoderJob | None = None) -> None:
    if job is not None:
        try:
            job.terminate()
        except OSError:
            pass
    if os.name == "nt":
        try:
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


def _pump_stream(source: BinaryIO, destination: BinaryIO) -> None:
    """Copy raw pickle bytes without retaining output-sized buffers in parent."""
    try:
        while True:
            chunk = source.read(64 * 1024)
            if not chunk:
                break
            destination.write(chunk)
            destination.flush()
    except (BrokenPipeError, OSError, ValueError):
        pass
    finally:
        try:
            destination.close()
        except OSError:
            pass
        try:
            source.close()
        except OSError:
            pass


def _read_exact(stream: BinaryIO, size: int, *, eof_ok: bool = False) -> bytearray | None:
    buffer = bytearray(size)
    view = memoryview(buffer)
    offset = 0
    while offset < size:
        count = stream.readinto(view[offset:])
        if not count:
            if offset == 0 and eof_ok:
                return None
            raise ValueError(f"truncated decoder frame ({offset}/{size} bytes)")
        offset += count
    return buffer


def _read_frame(stream: BinaryIO, max_frame_bytes: int) -> tuple[bytes, memoryview] | None:
    header = _read_exact(stream, 4, eof_ok=True)
    if header is None:
        return None
    (size,) = struct.unpack(">I", header)
    if size < 1 or size > max_frame_bytes:
        raise ValueError(
            f"pickle decoder emitted invalid frame size {size} (cap {max_frame_bytes})"
        )
    body = _read_exact(stream, size)
    assert body is not None
    return bytes(body[:1]), memoryview(body)[1:]


def _worker_command(config: dict[str, int]) -> list[str]:
    bootstrap = "from apitrace_mcp.pickleparse import _worker_entry; _worker_entry()"
    return [sys.executable, "-c", bootstrap, json.dumps(config, separators=(",", ":"))]


def _process_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": PROCESS_FLAGS}
    return {"start_new_session": True}


def iter_calls(
    root: ApitraceRoot,
    trace: str | Path,
    *,
    calls: str | None = None,
    symbolic: bool = True,
    limit: int | None = None,
    timeout: int | float = DEFAULT_PICKLE_TIMEOUT,
    worker_memory_bytes: int = DEFAULT_WORKER_MEMORY_BYTES,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    max_record_nodes: int = DEFAULT_MAX_RECORD_NODES,
    max_record_depth: int = DEFAULT_MAX_RECORD_DEPTH,
    max_record_blob_bytes: int = DEFAULT_MAX_RECORD_BLOB_BYTES,
    max_record_string_chars: int = DEFAULT_MAX_RECORD_STRING_CHARS,
    max_record_floats: int = DEFAULT_MAX_RECORD_FLOATS,
) -> Iterator[Call]:
    """Decode in a disposable memory-capped worker and stream bounded records."""
    trace_path = Path(trace)
    if not trace_path.is_file():
        raise FileNotFoundError(f"trace not found: {trace_path}")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError("limit must be a non-negative integer or None")
    if limit == 0:
        return
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("timeout must be a positive number")
    budgets = {
        "memory_bytes": worker_memory_bytes,
        "max_frame_bytes": max_frame_bytes,
        "max_nodes": max_record_nodes,
        "max_depth": max_record_depth,
        "max_blob_bytes": max_record_blob_bytes,
        "max_string_chars": max_record_string_chars,
        "max_floats": max_record_floats,
    }
    for name, value in budgets.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if worker_memory_bytes < 32 * 1024 * 1024:
        raise ValueError("worker_memory_bytes must be at least 32 MiB")
    if max_frame_bytes > worker_memory_bytes // 2:
        raise ValueError("max_frame_bytes must not exceed half the worker memory limit")

    cmd = pickle_cmd(root, trace_path, calls=calls, symbolic=symbolic)
    proc: subprocess.Popen | None = None
    worker: subprocess.Popen | None = None
    job: _DecoderJob | None = None
    pump: threading.Thread | None = None
    stderr_drain: _StderrDrain | None = None
    worker_stderr_drain: _StderrDrain | None = None
    timer: threading.Timer | None = None
    timed_out = threading.Event()
    stopped_early = False
    completed = False
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_process_kwargs(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"executable not found: {cmd[0]}") from exc
    try:
        assert proc.stdout is not None
        assert proc.stderr is not None
        worker = subprocess.Popen(
            _worker_command(budgets),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_process_kwargs(),
        )
        assert worker.stdin is not None
        assert worker.stdout is not None
        assert worker.stderr is not None
        job = _DecoderJob(worker_memory_bytes)
        job.assign(worker.pid)

        stderr_drain = _StderrDrain(proc.stderr)
        worker_stderr_drain = _StderrDrain(worker.stderr)
        stderr_drain.start()
        worker_stderr_drain.start()
        pump = threading.Thread(
            target=_pump_stream,
            args=(proc.stdout, worker.stdin),
            name="apitrace-pickle-pump",
            daemon=True,
        )
        pump.start()

        def expire() -> None:
            timed_out.set()
            if worker is not None:
                _kill_process_tree(worker, job)
            if proc is not None:
                _kill_process_tree(proc)

        timer = threading.Timer(float(timeout), expire)
        timer.daemon = True
        timer.start()
        count = 0
        while True:
            frame = _read_frame(worker.stdout, max_frame_bytes)
            if frame is None:
                if timed_out.is_set():
                    raise RuntimeError(
                        f"apitrace pickle timed out after {timeout}s. Narrow the call "
                        "range (calls=) or raise the timeout."
                    )
                raise RuntimeError(
                    "pickle decoder worker exited before a completion frame; it may "
                    f"have exceeded its {worker_memory_bytes}-byte memory limit"
                )
            kind, payload = frame
            if kind == b"D":
                completed = True
                break
            if kind == b"E":
                detail = bytes(payload[:64 * 1024]).decode("utf-8", errors="replace")
                raise ValueError(f"invalid apitrace pickle stream: {detail}")
            if kind != b"R":
                raise ValueError(f"unknown pickle decoder frame type {kind!r}")
            try:
                # This pickle is emitted only by our restricted worker after a
                # graph/type validation pass, and its byte length is framed.
                record = pickle.loads(payload)
            except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as exc:
                raise ValueError(f"invalid bounded decoder record: {exc}") from exc
            call = _record_to_call(record)
            # The yielded Call owns the decoded values; release the serialized
            # frame before yielding so large blobs are not duplicated in parent
            # memory for the entire caller processing interval.
            del record
            del payload
            del frame
            yield call
            count += 1
            if limit is not None and count >= limit:
                stopped_early = True
                break

        if timed_out.is_set():
            raise RuntimeError(
                f"apitrace pickle timed out after {timeout}s. Narrow the call range "
                "(calls=) or raise the timeout."
            )

        if completed and not stopped_early:
            worker_code = worker.wait(timeout=10)
            returncode = proc.wait(timeout=10)
            stderr_text = stderr_drain.finish()
            worker_stderr = worker_stderr_drain.finish()
            if timed_out.is_set():
                raise RuntimeError(
                    f"apitrace pickle timed out after {timeout}s. Narrow the call range "
                    "(calls=) or raise the timeout."
                )
            if worker_code != 0:
                raise RuntimeError(
                    f"pickle decoder worker failed (exit {worker_code}): "
                    f"{worker_stderr.strip()[-2000:] or '<no stderr>'}"
                )
            if returncode != 0:
                raise CommandError(cmd, returncode, stderr_text)
    finally:
        if timer is not None:
            timer.cancel()
        if worker is not None and worker.poll() is None:
            _kill_process_tree(worker, job)
        if proc is not None and proc.poll() is None:
            _kill_process_tree(proc)
        if pump is not None:
            pump.join(timeout=5)
        for process in (worker, proc):
            if process is None:
                continue
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if stderr_drain is not None:
            stderr_drain.finish()
        if worker_stderr_drain is not None:
            worker_stderr_drain.finish()
        if job is not None:
            job.close()


# --------------------------------------------------------------------------
# Value rendering
# --------------------------------------------------------------------------

MAX_INLINE_LIST = 32
DEFAULT_CONVERSION_NODES = 20_000
DEFAULT_CONVERSION_FLOATS = 100_000
DEFAULT_CONVERSION_BLOB_BYTES = 16 * 1024 * 1024
DEFAULT_HARVEST_NODES = 1_000_000
DEFAULT_HARVEST_FLOATS = 1_000_000


@dataclass
class _ConversionBudget:
    nodes_left: int
    floats_left: int
    blob_bytes_left: int


def _budget_marker(kind: str) -> str:
    return f"<{kind}-budget-exceeded>"


def to_jsonable(
    value: Any,
    *,
    max_list: int = MAX_INLINE_LIST,
    depth: int = 0,
    max_nodes: int = DEFAULT_CONVERSION_NODES,
    max_floats: int = DEFAULT_CONVERSION_FLOATS,
    max_blob_bytes: int = DEFAULT_CONVERSION_BLOB_BYTES,
    _budget: _ConversionBudget | None = None,
    _active: set[int] | None = None,
) -> Any:
    """Render an argument value compactly, never inlining a whole vertex buffer."""
    if _budget is None:
        _budget = _ConversionBudget(max_nodes, max_floats, max_blob_bytes)
    if _active is None:
        _active = set()
    _budget.nodes_left -= 1
    if _budget.nodes_left < 0:
        return _budget_marker("node")
    if depth > 6:
        return "<deep>"
    if isinstance(value, Pointer):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        size = len(value)
        _budget.blob_bytes_left -= size
        if _budget.blob_bytes_left < 0:
            return {"blob_bytes": size, "summary": _budget_marker("blob")}
        return {
            "blob_bytes": size,
            "sha1": hashlib.sha1(value).hexdigest(),
            "head_hex": memoryview(value)[:32].hex(),
        }
    if isinstance(value, float):
        _budget.floats_left -= 1
        return value if _budget.floats_left >= 0 else _budget_marker("float")
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, dict):
        identity = id(value)
        if identity in _active:
            return "<cycle>"
        _active.add(identity)
        shown: dict[str, Any] = {}
        try:
            for index, (key, item) in enumerate(value.items()):
                if index >= max_list:
                    shown["<truncated>"] = f"...+{len(value) - max_list} more"
                    break
                shown[str(key)] = to_jsonable(
                    item,
                    max_list=max_list,
                    depth=depth + 1,
                    _budget=_budget,
                    _active=_active,
                )
            return shown
        finally:
            _active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in _active:
            return "<cycle>"
        _active.add(identity)
        # apitrace models "pointer to one value" as a single-element list.
        try:
            if len(value) == 1:
                return to_jsonable(
                    value[0],
                    max_list=max_list,
                    depth=depth + 1,
                    _budget=_budget,
                    _active=_active,
                )
            shown = []
            for index, item in enumerate(value):
                if index >= max_list:
                    shown.append(f"...+{len(value) - max_list} more")
                    break
                shown.append(
                    to_jsonable(
                        item,
                        max_list=max_list,
                        depth=depth + 1,
                        _budget=_budget,
                        _active=_active,
                    )
                )
            return shown
        finally:
            _active.remove(identity)
    rendered = repr(value)
    return rendered if len(rendered) <= 1000 else rendered[:997] + "..."


def call_to_dict(call: Call, *, max_list: int = MAX_INLINE_LIST) -> dict:
    budget = _ConversionBudget(
        DEFAULT_CONVERSION_NODES,
        DEFAULT_CONVERSION_FLOATS,
        DEFAULT_CONVERSION_BLOB_BYTES,
    )
    active: set[int] = set()
    rendered_args: dict[str, Any] = {}
    for name, value in call.args:
        rendered_args[name] = to_jsonable(
            value, max_list=max_list, _budget=budget, _active=active
        )
    out: dict[str, Any] = {
        "no": call.no,
        "name": call.name,
        "args": rendered_args,
    }
    if call.ret is not None:
        out["ret"] = to_jsonable(
            call.ret, max_list=max_list, _budget=budget, _active=active
        )
    flags = call.flag_names
    if flags:
        out["flags"] = flags
    if call.thread_id is not None:
        out["thread"] = call.thread_id
    return out


def harvest_floats(
    value: Any,
    out: list[float] | None = None,
    *,
    max_depth: int = DEFAULT_MAX_RECORD_DEPTH,
    max_nodes: int = DEFAULT_HARVEST_NODES,
    max_floats: int = DEFAULT_HARVEST_FLOATS,
) -> list[float]:
    """Flatten every float in a nested value, in order.

    apitrace models D3DMATRIX as a struct-of-array, GL matrices as a flat array
    and shader constants as an array of float4 -- flattening sidesteps having to
    special-case each representation.
    """
    if out is None:
        out = []
    if len(out) > max_floats:
        raise ValueError(f"float harvest exceeds float budget ({max_floats})")
    nodes = 0
    active: set[int] = set()
    stack: list[tuple[str, Any, int]] = [("value", value, 0)]
    while stack:
        kind, current, depth = stack.pop()
        if kind == "exit":
            active.discard(current)
            continue
        if kind == "iterator":
            try:
                child = next(current)
            except StopIteration:
                continue
            stack.append(("iterator", current, depth))
            stack.append(("value", child, depth))
            continue
        nodes += 1
        if nodes > max_nodes:
            raise ValueError(f"float harvest exceeds node budget ({max_nodes})")
        if depth > max_depth:
            raise ValueError(f"float harvest exceeds nesting depth ({max_depth})")
        if isinstance(current, bool):
            continue
        if isinstance(current, float):
            if len(out) >= max_floats:
                raise ValueError(f"float harvest exceeds float budget ({max_floats})")
            out.append(current)
            continue
        if not isinstance(current, (dict, list, tuple)):
            continue
        identity = id(current)
        if identity in active:
            raise ValueError("float harvest encountered a container cycle")
        active.add(identity)
        stack.append(("exit", identity, depth))
        children = current.values() if isinstance(current, dict) else current
        stack.append(("iterator", iter(children), depth + 1))
    return out
