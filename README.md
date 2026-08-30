# apitrace-mcp

An MCP server wrapping [apitrace](https://github.com/apitrace/apitrace), aimed at
**reverse-engineering older games**: DirectDraw/D3D7, D3D8, D3D9 and legacy
OpenGL — the APIs RenderDoc will not open.

It is not a thin CLI wrapper. On top of capture and replay it adds an analysis
layer that answers the questions that actually come up when you are trying to
understand an old renderer:

- **Where is the camera?** `find_matrices` scans candidate 4×4 matrices the game
  hands the driver — fixed-function `SetTransform`, `glLoadMatrixf`, shader
  constants, and resource-qualified uniform/constant-buffer uploads — and ranks
  likely projection, view, and combined view-projection slots. Legacy OpenGL
  matrix stacks are reconstructed per context when make-current calls identify
  it, with a conservative per-trace-thread fallback when they do not.
- **What is the FOV, near and far?** `decode_matrix` / `find_matrices` decode
  them, in both D3D and OpenGL conventions, and in both storage orders (D3D9
  shader constants usually hold the transpose).
- **Is that really the view matrix?** `track_camera` extracts the world-space eye
  and view-Z axis per frame. It reports both handedness-dependent forward-vector
  candidates, so the trace is not made to claim an orientation it cannot prove.
- **Which register would a VR patch rewrite?** Matrix slots are reported as
  register ranges — `vs_c[8..11]` — not just call numbers. Both float4x4
  (4-register) and float4x3 (3-register) uploads are recognised, and windows are
  only taken at whole-matrix offsets from the upload that wrote them, so a large
  packed bone-matrix block does not bury the camera in overlapping candidates.

Validated against a real Direct3D 9 / Unreal Engine 3 capture: from a 786-frame
Dishonored trace it recovers the float4x3 view matrix (world-space eye,
cross-checked against the camera-position constant the engine uploads
separately) and the projection scales giving 58.08° vertical / 89.25° horizontal
FOV at 16:9.

## Why apitrace and not RenderDoc

RenderDoc captures D3D11/12, OpenGL (modern) and Vulkan. It cannot capture D3D9,
D3D8, DirectDraw, or the legacy GL that pre-2010 games use. apitrace can, and it
records the full API call stream with argument values rather than a single frame.

| | apitrace | RenderDoc |
|---|---|---|
| DirectDraw / D3D7 | trace only | no |
| D3D8 / D3D9 | trace + replay/inspect | no |
| Legacy OpenGL (1.x–2.x, ARB programs) | yes | no |
| D3D11, modern GL, Vulkan | yes (D3D11/GL) | yes, and better |
| D3D12 | no | yes |
| Captures | whole run, all frames | one frame |
| Per-draw pipeline state | via replay (`dump_state`) | first-class |

Use both: apitrace for old APIs and whole-run call streams, RenderDoc for modern
APIs and deep single-frame inspection.

## Requirements

- An unpacked apitrace release. The server auto-discovers builds unpacked next
  to this clone (inside it, or in a sibling directory such as `..\apitrace\`)
  plus the standard locations (`C:\apitrace`, Program Files, `~\apitrace`), or
  set `APITRACE_MCP_ROOT_WIN64` / `APITRACE_MCP_ROOT_WIN32` explicitly.
- **Both the win64 and the win32 build if you trace 32-bit games.** A win64
  apitrace cannot trace a 32-bit process — the wrapper DLL has to load into the
  target's address space. Most pre-2012 games are 32-bit.
- Windows and Python 3.11+. Core tools need only the MCP SDK; `diff_images`
  additionally needs Pillow because upstream `snapdiff.py` imports `PIL`:
  `uv pip install -e ".[images]"`.

`APITRACE_MCP_TRACE_DIR` sets where traces, logs and extracted files go
(default `%LOCALAPPDATA%\apitrace-mcp\traces`). Traces are large — point this
at a drive with room.

## Install

```bash
uv venv --python 3.13
uv pip install -e .
```

Register it (Claude Code, from the clone directory):

```bash
claude mcp add apitrace -e APITRACE_MCP_TRACE_DIR=<where-traces-should-go> -- "<path-to-clone>/.venv/Scripts/apitrace-mcp.exe"
```

Or add to `claude_desktop_config.json` / `.claude.json`:

```json
"apitrace": {
  "command": "<path-to-clone>\\.venv\\Scripts\\apitrace-mcp.exe",
  "args": [],
  "env": { "APITRACE_MCP_TRACE_DIR": "<where-traces-should-go>" }
}
```

(The `env` block is optional — without it traces land in
`%LOCALAPPDATA%\apitrace-mcp\traces`.)

`python register.py` can register the editable install with Claude Desktop and
Codex after backing up their configuration files.

## Tools

**Environment** — `apitrace_status` (preflight), `list_traces`, `detect_target`

**Capture** — `trace_launch`, `trace_status`, `trace_stop`, `list_sessions`,
`install_wrapper`, `uninstall_wrapper`

**Inspect** — `trace_info`, `list_frames`, `dump_calls`, `get_calls`,
`search_calls`, `call_histogram`, `frame_summary`

**Analyse** — `find_matrices`, `decode_matrix`, `track_camera`, `list_shaders`,
`extract_blobs`

**Replay** — `dump_state`, `dump_images`, `replay_trace`

**Maintain** — `trim_trace` (raw slice), `gltrim_trace` (replayable GL subset),
`leak_report`, `repack_trace`, `sed_trace` (stream edit into a new trace)

**Compare** — `diff_traces` (semantic call streams), `diff_state` (complete
retracer state at two calls), `diff_images` (HTML image report and sidecars)

Most read tools take apitrace's callset syntax in `calls=`: `42`, `0,2,4`,
`100-2000`, `0-1000/2`, `0-1000/draw`, `0-1000/fbo`, `frame`, `@file.txt`.

## Typical session

```
apitrace_status()                                  # which builds are installed
detect_target("D:/Games/Foo/Foo.exe")              # 32-bit? D3D9? 
trace_launch(game_exe=..., api="d3d9")             # play the scene
trace_stop("trace-1")
list_frames(trace)                                 # find a frame's call range
find_matrices(trace, calls="120000-125000")        # projection / view / viewproj
track_camera(trace, calls="100000-200000")         # confirm the eye moves
small = gltrim_trace(trace, frames="40-42")        # replayable GL subset
dump_images(small["output"], calls="*/draw")       # see what each draw contributes
```

Then take the finding into Ghidra (find the code that writes that matrix) or
ReGenny (rebuild the struct it lives in).

## Notes and limits

- **Tracing is invasive and slow.** Old games can drop to single-digit FPS under
  a wrapper. Capture a few seconds, not a level.
- **Quit the game normally** before `trace_stop`. A forced kill can lose the tail
  of the trace.
- **Empty trace?** Usually launcher indirection — the exe you launched spawned
  the real game, so the injection landed in the wrong process. Use
  `install_wrapper` and start the game yourself (this also works for Steam).
- **`install_wrapper` backs up existing DLLs.** ENB, ReShade, dgVoodoo and DXVK
  all install as `d3d9.dll`/`dxgi.dll`/`opengl32.dll`. Existing files are backed
  up and restored on `uninstall_wrapper` — but those mods are inactive while the
  apitrace wrapper is in place. Its generated launcher refuses to overwrite an
  earlier capture; uninstall and reinstall to allocate a fresh default trace.
  Always uninstall when done.
- **D3D10/11 must use `trace_launch`,** not `install_wrapper`: those DLLs
  load each other by name, so a DLL drop traces the wrong calls. apitrace's
  injector handles it.
- **D3D12 and Vulkan are not supported** by apitrace at all — use RenderDoc.
- **Traces are untrusted binary input.** The pickle reader is restricted to
  apitrace's own `Pointer` class and runs in a disposable decoder worker with a
  512 MiB memory limit, wall-clock timeout, and length-framed output. Per-call
  decoding is capped at 1,000,000 value nodes, depth 64, 4,000,000 floats,
  48 MiB of blobs, and a 64 MiB return frame. An individual call whose payload
  exceeds those limits is rejected rather than returned or extracted; narrow
  `calls=` when a trace contains unusually large resource uploads. A plain
  `pickle.load` on a hostile trace would be an arbitrary-code-execution hole.
- **`trim_trace` is a raw slice.** For a replayable mid-run OpenGL subset, use
  `gltrim_trace`, which preserves setup state. DirectDraw/D3D7 captures cannot be
  replayed upstream.
- **`diff_images` writes thumbnails and diff PNGs beside its input prefixes.**
  The MCP wrapper never enables upstream overwrite mode and rejects existing or
  concurrently reserved sidecars; use fresh image prefixes for each comparison.
- **Generated file outputs are UUID-backed.** The actual path is returned in the
  structured result, and an existing requested path is never overwritten. This
  remains safe when Claude and Codex use separate MCP server processes.
- **`diff_state` accepts a trace and two call numbers.** It captures complete
  retracer JSON at both calls, canonicalises apitrace's JSON dialect, then runs
  upstream `jsondiff`; it does not require hand-created state files.
- Tools return native structured MCP data, advertise read/write/destructive
  annotations, and report operational failures through the MCP error channel.
- Cancelling an MCP request cannot yet interrupt a synchronous apitrace command
  already running in a worker thread. Foreground work is limited to two commands,
  remains bounded by each tool's timeout, and timeout cleanup terminates the
  complete child-process tree.

## Tests

The suite uses the standard library and needs no test dependency:

```bash
.venv/Scripts/python -m unittest discover -s tests -v
```

For pytest-based development tooling, install the optional extra with
`uv pip install -e ".[dev]"`.
