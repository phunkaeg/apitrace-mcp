# apitrace MCP — agent guide

## Preflight

`apitrace_status`. No host application needs to be running — unlike Ghidra, ReGenny or
Cheat Engine, apitrace is a command-line tool, so the server works whenever it is called.
What it *does* need is the build matching the target's bitness. Check the `warnings` field.

**A win64 apitrace cannot trace a 32-bit game.** The wrapper DLL has to load into the target's
address space. Most pre-2012 games are 32-bit, so a missing win32 build blocks Bioshock HD,
Dishonored, SWAT4, Far Cry 2 and Cliffs of Dover outright.

## When to reach for this instead of RenderDoc

| Target API | Tool |
|---|---|
| DirectDraw / D3D7 / D3D8 / D3D9 | **apitrace** (RenderDoc cannot open these) |
| Legacy OpenGL 1.x–2.x, ARB programs | **apitrace** |
| Modern OpenGL, D3D11, Vulkan | RenderDoc first; apitrace if you want the whole run |
| D3D12 | RenderDoc only |

apitrace records **every frame of a run** with argument values; RenderDoc captures **one frame**
in depth. For "what does this engine do over time", apitrace. For "why is this pixel wrong",
RenderDoc.

## Capture

1. `detect_target(game_exe)` — bitness and API from the PE import table. Do this first; it costs
   nothing and it decides everything else. API auto-selection is ranked (d3d9 > d3d8 > dxgi > gl >
   d3d7), so a legacy `ddraw.dll` import cannot shadow the real renderer; runners-up are reported.
2. `trace_launch(game_exe, api=...)` → session id. **This runs the game.** It returns immediately;
   the user plays.
3. `trace_stop(session)` — ask the user to quit the game normally first. A forced kill can lose
   the tail of the trace.

**Empty trace?** Almost always launcher indirection: the exe you launched spawned the real game,
so the injection landed in the wrong process. Fall back to `install_wrapper`, which drops the
wrapper DLL next to the exe so it loads however the game is started (including from Steam).
Then `uninstall_wrapper` — a left-behind wrapper keeps tracing and slowing every future run.

`install_wrapper` backs up any existing `d3d9.dll` / `dxgi.dll` / `opengl32.dll`. In the
old-game modding scene those names are usually already taken by ENB, ReShade, dgVoodoo or DXVK.
Those mods are inactive while apitrace is installed; the backup is restored on uninstall.

D3D10/11 must use `trace_launch -a dxgi`, never the DLL drop — the D3D DLLs load each other by name.

## Reading a trace

Traces are large. Scope everything with `calls=`: `100-2000`, `0-1000/draw`, `0-1000/fbo`,
`frame`, `@file.txt`.

- `list_frames` → per-frame call ranges. Get these before anything else and reuse them.
- `call_histogram` → what the game does, in one call.
- `search_calls("SetTransform|SetVertexShaderConstant")` → fixed-function or shader-era? This
  single query decides how you look for the camera.
- `get_calls` → structured args with real numbers. `dump_calls` → text, for eyeballing.
- Structured decoding happens in a killable 512 MiB worker. One call is rejected
  if it exceeds 1,000,000 nested values, depth 64, 4,000,000 floats, 48 MiB of
  blobs, or its 64 MiB framed result. In particular, `extract_blobs` will not
  extract a single upload above the 48 MiB per-call blob budget; narrow `calls=`
  or use upstream tools directly for intentionally huge resource payloads.
- `extract_blobs` also caps its walk at max(limit*500, 20000) calls and reports
  `truncated: true` with `calls_scanned` when the cap cut the search short —
  narrow `calls=` rather than trusting a count=0 from a huge range.
- Slice from call/frame zero with `trim_trace` when a raw prefix is sufficient.
  For a replayable mid-run OpenGL subset, use `gltrim_trace` so setup state and
  resources are retained.

## Finding the camera

`find_matrices(trace, calls=<one frame>)` is the main event. Read the slots:

| slot kind | meaning |
|---|---|
| `projection` | the game's FOV, near and far — read `fov_y_deg` |
| `rigid` + `changes_per_frame` | camera candidate; confirm it with `track_camera` |
| `viewproj` in `vs_c[8..11]` | the constant registers a VR patch rewrites |
| `rigid` in a 3-register slot like `vs_c[9..11]` | a float4x3 view/world matrix — D3D9 engines upload these to save constant space |
| `viewproj` at `glBufferData[target=GL_UNIFORM_BUFFER,buffer=N,...]+96` | resource-qualified bytes a VR patch may rewrite |
| `ortho` | HUD/UI or shadow pass |
| `SetTransform(D3DTS_PROJECTION)` | fixed-function engines (D3D8/D3D9) hand you FOV/near/far directly and decode at confidence 1.0 |

Then `track_camera` to confirm: if the eye position changes when the player moved during capture,
the slot is a strong view candidate. When a slot is written many times per frame it is a per-draw
world*view*projection, not a camera constant: the result then carries `writes_per_frame`,
`sample_ordinal` and `multi_write_note`, and the trajectory follows whichever write moves on the
most frames (not the one covering the most distance -- a single discontinuity would win that).
Do not read a single write's eye as the camera in that case — each expresses the eye in
its own object's space. Moving models, lights, reflections, and shadow cameras can
also move, so corroborate it with the explicit transform state/resource binding and scene timing.
The tool reports `view_z_axis` plus left-/right-handed forward candidates because a rigid matrix
alone does not prove handedness.

`camera_moves` requires *sustained* travel -- at least a quarter of frame transitions -- and the
result carries `moving_transitions` / `frame_transitions` so you can judge it yourself. A single
large jump between two otherwise-static values is a pass or mode switch, not a camera; the note
says so. In a fixed-function engine the camera is often folded into `D3DTS_WORLD` rather than
`D3DTS_VIEW`, so check both.

**Find the gameplay before you analyse.** A capture is mostly menu far more often than you expect:
a Morrowind trace ran 64 million calls of menu at ~11 draws/frame before ~440 draws/frame of actual
world rendering. Probe `call_histogram` at a few offsets and follow the draws-per-frame.

`decode_matrix` takes 16 pasted floats from anywhere — a Cheat Engine watch, a memory dump, a
ReGenny read — and reports FOV, near/far, handedness and camera position.

### What the decoders can and cannot tell you

- **Storage order is inferred, not known.** Both row- and column-major are tried and the result
  is labelled `layout: as_stored | transposed`. D3D9 shader constants normally hold the transpose.
- **Depth convention is ambiguous from the matrix alone.** A D3D projection (NDC 0..1) and a GL
  one (NDC -1..1) have identical layouts and differ only in how near/far decode. The convention
  is taken from the function name; where it cannot be, both readings are reported in
  `alt_near_far`. **FOV and aspect are unaffected** — those are always trustworthy.
- **Constant windows are upload-aligned.** Candidate register windows are only
  taken at whole-matrix offsets from the upload that wrote them, in 4-register
  (float4x4) and 3-register (float4x3) sizes. Without that, one 144-register
  bone-matrix upload yields ~140 overlapping pseudo-matrices. A consequence
  worth knowing: a matrix written by a *partial* upload that starts mid-matrix
  will not be enumerated at that offset.
- **A slot can decode as more than one kind over a run.** When it does, the
  result carries `kind_breakdown` alongside the majority `kind` — read the
  per-sample `decode.kind`, not just the slot header. Slots are ranked by kind,
  then decode confidence, then write count.
- **Buffer scanning is heuristic.** Windows found inside `glBufferData`/`UpdateSubresource`/
  `memcpy` blobs must clear a higher bar (the w-column has to be a unit view direction), and
  vertex/index buffer targets are skipped entirely. Treat a lone buffer hit as a lead, not a
  fact — sanity-check the FOV and eye before acting on it.
- A `rigid` matrix from `SetTransform(D3DTS_WORLD)` is a *model* matrix; its "eye" is meaningless.
  The `State` argument in the slot name disambiguates.

## Replay

`dump_state(trace, call)` is the apitrace answer to RenderDoc's pipeline view for OpenGL and
D3D8/D3D9. It parses the complete retracer state and hashes/summarises embedded image payloads
before returning a bounded result. It is slow and can fail on very old games; DirectDraw/D3D7
are capture-only upstream.

`dump_images(trim, calls="*/draw")` writes a PNG per draw: the fastest way to find which draw
call produces the HUD, the world, or the artefact you are chasing.

Use `leak_report` for OpenGL object-lifetime leaks and `repack_trace` to change trace compression.

Use `diff_traces(reference, source)` for a semantic API-call diff. Use
`diff_state(trace, reference_call, source_call)` when the interesting change is
pipeline/device state; it safely captures both complete state dumps before invoking
upstream `jsondiff`. Use `sed_trace` to stream-edit enum/string symbols or properties
into a new, non-overwriting trace.

`diff_images(reference_prefix, source_prefix)` produces an HTML comparison report.
It requires the optional Pillow extra (`uv pip install -e ".[images]"`) and upstream
also creates `.thumb.png` / `.diff.png` sidecars
beside the supplied image prefixes. The MCP refuses existing or concurrently reserved
sidecars and never enables overwrite, so generate fresh prefixes for each run.
All generated output files/prefixes receive a UUID suffix; consume the actual path
returned by the tool rather than assuming the requested base name.

## Pairings

- **apitrace → Ghidra** — find the call that sets a matrix, then find the code that produced it.
- **apitrace → ReGenny** — a view matrix in a trace tells you what to look for in memory.
- **apitrace → Cheat Engine** — matrix values from a trace are exact search targets.
