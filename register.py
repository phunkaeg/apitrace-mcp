"""Register (or unregister) apitrace-mcp with local MCP hosts.

Only the apitrace entry is changed.  Every real change gets a unique backup and
is committed with an atomic same-directory replace so a failed write cannot
leave either host's configuration half-written.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import os
import shutil
import sys
import tomllib
import uuid
from pathlib import Path
from typing import Any, Callable

NAME = "apitrace"
SERVER_EXE = Path(__file__).resolve().parent / ".venv" / "Scripts" / "apitrace-mcp.exe"

CLAUDE_DESKTOP = Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
CODEX = Path.home() / ".codex" / "config.toml"


class RegistrationError(RuntimeError):
    """A configuration could not be changed safely."""


def _newline_of(text: str) -> str:
    return "\r\n" if text.count("\r\n") > text.count("\n") - text.count("\r\n") else "\n"


def _read_utf8(path: Path, *, allow_bom: bool) -> tuple[str, bool]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RegistrationError(f"cannot read {path}: {exc}") from exc
    has_bom = payload.startswith(b"\xef\xbb\xbf")
    if has_bom:
        if not allow_bom:
            raise RegistrationError(f"{path} has a UTF-8 BOM, which TOML does not permit")
        payload = payload[3:]
    try:
        return payload.decode("utf-8"), has_bom
    except UnicodeDecodeError as exc:
        raise RegistrationError(f"{path} is not valid UTF-8: {exc}") from exc


def backup(path: Path) -> Path:
    """Create a timestamped, collision-proof backup without replacing anything."""
    destination: Path | None = None
    target = None
    for _ in range(100):
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        candidate = path.with_name(
            f"{path.name}.bak.{stamp}-{uuid.uuid4().hex[:8]}"
        )
        try:
            target = candidate.open("xb")
        except FileExistsError:
            continue
        except OSError as exc:
            raise RegistrationError(f"cannot create backup for {path}: {exc}") from exc
        destination = candidate
        break
    if destination is None or target is None:
        raise RegistrationError(f"cannot allocate a unique backup name for {path}")
    try:
        with path.open("rb") as source, target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        shutil.copystat(path, destination)
    except OSError as exc:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise RegistrationError(f"cannot back up {path} to {destination}: {exc}") from exc
    return destination


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    handle = None
    for _ in range(100):
        candidate = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            handle = candidate.open("xb")
        except FileExistsError:
            continue
        except OSError as exc:
            raise RegistrationError(f"cannot create temporary file beside {path}: {exc}") from exc
        temporary = candidate
        break
    if temporary is None or handle is None:
        raise RegistrationError(f"cannot allocate a unique temporary file beside {path}")
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            shutil.copystat(path, temporary)
        except OSError as exc:
            raise RegistrationError(f"cannot preserve permissions for {path}: {exc}") from exc
        os.replace(temporary, path)
    except RegistrationError:
        raise
    except OSError as exc:
        raise RegistrationError(f"cannot atomically replace {path}: {exc}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _commit(path: Path, payload: bytes) -> Path:
    saved = backup(path)
    try:
        _atomic_write(path, payload)
    except RegistrationError as exc:
        raise RegistrationError(
            f"{exc}; original remains in place and backup is available at {saved}"
        ) from exc
    return saved


def _parse_json(text: str, path: Path) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RegistrationError(
            f"invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise RegistrationError(f"{path} must contain a JSON object at the top level")
    servers = data.get("mcpServers")
    if servers is not None and not isinstance(servers, dict):
        raise RegistrationError(f"mcpServers in {path} must be a JSON object")
    return data


def claude_desktop(remove: bool) -> str:
    if not CLAUDE_DESKTOP.is_file():
        return f"claude desktop: skipped (not found): {CLAUDE_DESKTOP}"
    text, had_bom = _read_utf8(CLAUDE_DESKTOP, allow_bom=True)
    data = _parse_json(text, CLAUDE_DESKTOP)
    servers = data.get("mcpServers")

    if remove:
        if not isinstance(servers, dict) or NAME not in servers:
            return "claude desktop: nothing to remove"
        del servers[NAME]
    else:
        if isinstance(servers, dict) and NAME in servers:
            existing = servers[NAME]
            if isinstance(existing, dict) and existing.get("command") == str(SERVER_EXE):
                return "claude desktop: already registered"
            raise RegistrationError(
                f"claude desktop already has mcpServers.{NAME} with a different "
                "configuration; refusing to overwrite it"
            )
        if servers is None:
            servers = {}
            data["mcpServers"] = servers
        servers[NAME] = {"command": str(SERVER_EXE), "args": []}

    newline = _newline_of(text)
    rendered = json.dumps(data, indent=2, ensure_ascii=False)
    rendered = rendered.replace("\n", newline) + newline
    # Validate our serialization before creating a backup or touching the file.
    _parse_json(rendered, CLAUDE_DESKTOP)
    payload = rendered.encode("utf-8")
    if had_bom:
        payload = b"\xef\xbb\xbf" + payload
    saved = _commit(CLAUDE_DESKTOP, payload)
    action = "removed" if remove else "registered"
    return f"claude desktop: {action} (backup: {saved})"


def _parse_toml(text: str, path: Path) -> dict[str, Any]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RegistrationError(f"invalid TOML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistrationError(f"{path} must contain a TOML document")
    servers = data.get("mcp_servers")
    if servers is not None and not isinstance(servers, dict):
        raise RegistrationError(f"mcp_servers in {path} must be a table")
    return data


def _header_document(line: str) -> dict[str, Any] | None:
    """Parse a physical line as a standalone TOML table header, if it is one."""
    if not line.lstrip().startswith("["):
        return None
    try:
        return tomllib.loads(line.rstrip("\r\n") + "\n")
    except tomllib.TOMLDecodeError:
        return None


def _is_blank_or_comment(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _without_server(data: dict[str, Any]) -> dict[str, Any]:
    expected = copy.deepcopy(data)
    servers = expected.get("mcp_servers")
    if isinstance(servers, dict):
        servers.pop(NAME, None)
        if not servers:
            expected.pop("mcp_servers", None)
    return expected


def _normalise_empty_servers(data: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    if result.get("mcp_servers") == {}:
        result.pop("mcp_servers")
    return result


def _remove_codex_table(text: str, parsed: dict[str, Any], path: Path) -> str:
    """Find a target table span by validating candidate removals semantically.

    This avoids a regex accidentally consuming an unrelated table, and also
    handles quoted table names and target subtables such as ``.env``.
    """
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line)
    headers: list[tuple[int, dict[str, Any]]] = []
    for index, line in enumerate(lines):
        header = _header_document(line)
        if header is not None:
            headers.append((index, header))

    target_shape = {"mcp_servers": {NAME: {}}}
    target_starts = [index for index, header in headers if header == target_shape]
    expected = _without_server(parsed)
    matches: list[tuple[int, int, str]] = []
    header_indexes = [index for index, _ in headers]
    for start_index in target_starts:
        start = offsets[start_index]
        possible_ends: list[int] = []
        for index in header_indexes:
            if index <= start_index:
                continue
            # Comment lines directly above the next header document that
            # header, not the removed table, so end the span before them.
            # Blank separator lines at the top of that run would leave
            # doubled spacing behind; let those go with the removed span.
            end_index = index
            while end_index - 1 > start_index and _is_blank_or_comment(
                lines[end_index - 1]
            ):
                end_index -= 1
            while end_index < index and not lines[end_index].strip():
                end_index += 1
            possible_ends.append(offsets[end_index])
        possible_ends.append(len(text))
        for end in possible_ends:
            candidate = text[:start] + text[end:]
            try:
                candidate_data = tomllib.loads(candidate)
            except tomllib.TOMLDecodeError:
                continue
            if _normalise_empty_servers(candidate_data) == _normalise_empty_servers(expected):
                matches.append((start, end, candidate))
                break

    unique = {(start, end): candidate for start, end, candidate in matches}
    if len(unique) != 1:
        detail = "not represented by one standalone table" if not unique else "ambiguous"
        raise RegistrationError(
            f"cannot safely remove mcp_servers.{NAME} from {path}: target table is {detail}; "
            "no changes were made"
        )
    return next(iter(unique.values()))


def _toml_string(value: str) -> str:
    # JSON basic strings are also valid TOML basic strings for Windows paths.
    return json.dumps(value, ensure_ascii=False)


def codex(remove: bool) -> str:
    if not CODEX.is_file():
        return f"codex: skipped (not found): {CODEX}"
    text, _ = _read_utf8(CODEX, allow_bom=False)
    parsed = _parse_toml(text, CODEX)
    servers = parsed.get("mcp_servers")
    existing = servers.get(NAME) if isinstance(servers, dict) else None

    if remove:
        if not isinstance(servers, dict) or NAME not in servers:
            return "codex: nothing to remove"
        rendered = _remove_codex_table(text, parsed, CODEX)
    else:
        if isinstance(servers, dict) and NAME in servers:
            if isinstance(existing, dict) and existing.get("command") == str(SERVER_EXE):
                return "codex: already registered"
            raise RegistrationError(
                f"codex already has mcp_servers.{NAME} with a different configuration; "
                "refusing to overwrite it"
            )
        newline = _newline_of(text)
        block = (
            f"[mcp_servers.{NAME}]{newline}"
            f"command = {_toml_string(str(SERVER_EXE))}{newline}"
            f"args = []{newline}"
        )
        rendered = text
        if rendered and not rendered.endswith(("\n", "\r")):
            rendered += newline
        if rendered and not rendered.endswith(newline * 2):
            rendered += newline
        rendered += block

    result = _parse_toml(rendered, CODEX)
    result_servers = result.get("mcp_servers", {})
    if remove and isinstance(result_servers, dict) and NAME in result_servers:
        raise RegistrationError(f"internal error: {NAME} remained after TOML removal")
    if not remove and (
        not isinstance(result_servers, dict)
        or not isinstance(result_servers.get(NAME), dict)
        or result_servers[NAME].get("command") != str(SERVER_EXE)
    ):
        raise RegistrationError(f"internal error: generated TOML did not register {NAME}")

    saved = _commit(CODEX, rendered.encode("utf-8"))
    action = "removed" if remove else "registered"
    return f"codex: {action} (backup: {saved})"


def main() -> None:
    remove = "--remove" in sys.argv
    if not remove and not SERVER_EXE.is_file():
        sys.exit(
            f"server not built: {SERVER_EXE}\nRun: uv venv --python 3.13 && uv pip install -e ."
        )

    errors: list[str] = []
    operations: list[tuple[str, Callable[[bool], str]]] = [
        ("claude desktop", claude_desktop),
        ("codex", codex),
    ]
    for label, operation in operations:
        try:
            print(operation(remove))
        except RegistrationError as exc:
            message = f"{label}: ERROR: {exc}"
            print(message, file=sys.stderr)
            errors.append(message)

    print(
        "\nClaude Code is registered separately:\n"
        f'  claude mcp {"remove " + NAME if remove else f"add {NAME} --scope user -- {SERVER_EXE}"}'
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
