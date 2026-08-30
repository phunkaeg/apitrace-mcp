"""Minimal, dependency-free PE reader.

Used for two things the rest of the server depends on:

* deciding whether a target game is 32- or 64-bit, so we pick the matching
  apitrace build (a win64 apitrace cannot trace a 32-bit process);
* listing imported DLLs, so we can guess which graphics API a game uses
  without asking the user to go spelunking in Process Explorer.

Old games delay-load surprisingly often (d3d9 behind a "detect video card"
step), so the delay-load directory is walked as well as the normal one.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import BinaryIO

MACHINE_NAMES = {
    0x014C: "i386",
    0x0166: "mips",
    0x01C0: "arm",
    0x01C4: "armnt",
    0x0200: "ia64",
    0x8664: "amd64",
    0xAA64: "arm64",
}

# Graphics DLLs worth reporting, mapped to the apitrace `-a` API name and the
# wrapper DLL that has to be dropped next to the exe for manual tracing.
GRAPHICS_DLLS = {
    "opengl32.dll": ("gl", "opengl32.dll"),
    "ddraw.dll": ("d3d7", "ddraw.dll"),
    "d3d8.dll": ("d3d8", "d3d8.dll"),
    "d3d9.dll": ("d3d9", "d3d9.dll"),
    "d3d10.dll": ("dxgi", "d3d10.dll"),
    "d3d10_1.dll": ("dxgi", "d3d10_1.dll"),
    "d3d11.dll": ("dxgi", "d3d11.dll"),
    "dxgi.dll": ("dxgi", "dxgi.dll"),
    "d3d12.dll": (None, None),  # apitrace does not support D3D12
    "vulkan-1.dll": (None, None),  # nor Vulkan
}


class PEError(Exception):
    pass


@dataclass
class Section:
    name: str
    virtual_address: int
    virtual_size: int
    raw_address: int
    raw_size: int


@dataclass
class PEInfo:
    path: str
    machine: str
    bits: int
    is_dll: bool
    imports: list[str] = field(default_factory=list)
    delay_imports: list[str] = field(default_factory=list)

    @property
    def all_imports(self) -> list[str]:
        seen: dict[str, None] = {}
        for name in self.imports + self.delay_imports:
            seen.setdefault(name.lower(), None)
        return list(seen)


def _read_at(fh: BinaryIO, offset: int, size: int) -> bytes:
    if offset < 0 or size < 0:
        raise PEError(f"invalid file range: offset={offset}, size={size}")
    fh.seek(offset)
    data = fh.read(size)
    if len(data) != size:
        raise PEError(f"truncated file: wanted {size} bytes at {offset:#x}")
    return data


def _rva_to_offset(
    rva: int,
    sections: list[Section],
    size: int = 1,
    *,
    headers_size: int = 0,
    file_size: int | None = None,
) -> int | None:
    """Map only bytes that are actually backed by the file.

    ``VirtualSize`` can exceed ``SizeOfRawData``; mapping that zero-filled tail
    to a file offset can accidentally read the next section or past EOF.
    """
    if rva < 0 or size < 0:
        return None
    requested = max(size, 1)
    if headers_size and rva < headers_size and rva + requested <= headers_size:
        if file_size is None or rva + requested <= file_size:
            return rva
    for sec in sections:
        mapped_size = max(sec.virtual_size, sec.raw_size)
        if sec.virtual_address <= rva < sec.virtual_address + mapped_size:
            delta = rva - sec.virtual_address
            if delta + requested > sec.raw_size:
                return None
            offset = sec.raw_address + delta
            if file_size is not None and offset + requested > file_size:
                return None
            return offset
    return None


def _read_cstring(fh: BinaryIO, offset: int, limit: int = 256) -> str:
    if limit <= 0:
        raise PEError("invalid string bound")
    fh.seek(offset)
    raw = fh.read(limit)
    end = raw.find(b"\0")
    if end < 0:
        raise PEError(f"unterminated PE string at {offset:#x}")
    return raw[:end].decode("ascii", errors="replace")


def _rva_span(
    rva: int,
    sections: list[Section],
    *,
    headers_size: int,
    file_size: int,
) -> tuple[int, int] | None:
    """Return file offset and remaining contiguous mapped bytes for an RVA."""
    offset = _rva_to_offset(
        rva, sections, headers_size=headers_size, file_size=file_size
    )
    if offset is None:
        return None
    if headers_size and rva < headers_size:
        return offset, min(headers_size - rva, file_size - offset)
    for sec in sections:
        mapped_size = max(sec.virtual_size, sec.raw_size)
        if sec.virtual_address <= rva < sec.virtual_address + mapped_size:
            delta = rva - sec.virtual_address
            return offset, min(sec.raw_size - delta, file_size - offset)
    return None


def read_pe(path: str) -> PEInfo:
    """Parse enough of a PE file to answer 'how wide' and 'what does it import'."""
    with open(path, "rb") as fh:
        file_size = os.fstat(fh.fileno()).st_size
        if _read_at(fh, 0, 2) != b"MZ":
            raise PEError(f"not a PE image (no MZ header): {path}")
        (e_lfanew,) = struct.unpack("<I", _read_at(fh, 0x3C, 4))
        if _read_at(fh, e_lfanew, 4) != b"PE\0\0":
            raise PEError(f"not a PE image (no PE signature): {path}")

        coff = _read_at(fh, e_lfanew + 4, 20)
        machine, num_sections, _, _, _, opt_size, characteristics = struct.unpack(
            "<HHIIIHH", coff
        )
        if num_sections > 96:
            raise PEError(f"invalid PE section count: {num_sections}")
        opt_off = e_lfanew + 24
        (magic,) = struct.unpack("<H", _read_at(fh, opt_off, 2))
        if magic == 0x10B:
            bits, dir_relative, image_base_format, image_base_relative = 32, 96, "<I", 28
        elif magic == 0x20B:
            bits, dir_relative, image_base_format, image_base_relative = 64, 112, "<Q", 24
        else:
            raise PEError(f"unknown optional header magic {magic:#x}")
        if opt_size < dir_relative:
            raise PEError(
                f"optional header is too small for PE{bits}: {opt_size} bytes"
            )
        dir_off = opt_off + dir_relative
        (image_base,) = struct.unpack(
            image_base_format,
            _read_at(fh, opt_off + image_base_relative, struct.calcsize(image_base_format)),
        )
        (headers_size,) = struct.unpack("<I", _read_at(fh, opt_off + 60, 4))

        (num_dirs,) = struct.unpack(
            "<I", _read_at(fh, opt_off + (92 if bits == 32 else 108), 4)
        )
        available_dirs = (opt_size - dir_relative) // 8
        num_dirs = min(num_dirs, available_dirs)

        sec_off = opt_off + opt_size
        sections: list[Section] = []
        for i in range(num_sections):
            raw = _read_at(fh, sec_off + i * 40, 40)
            name = raw[:8].rstrip(b"\0").decode("ascii", errors="replace")
            vsize, vaddr, rsize, raddr = struct.unpack("<IIII", raw[8:24])
            if rsize and (raddr > file_size or rsize > file_size - raddr):
                raise PEError(f"section {name!r} extends past end of file")
            sections.append(Section(name, vaddr, vsize, raddr, rsize))

        info = PEInfo(
            path=path,
            machine=MACHINE_NAMES.get(machine, f"{machine:#06x}"),
            bits=bits,
            is_dll=bool(characteristics & 0x2000),
        )

        def directory(index: int) -> tuple[int, int]:
            if index >= num_dirs:
                return 0, 0
            return struct.unpack("<II", _read_at(fh, dir_off + index * 8, 8))

        def descriptor_range(
            rva: int, directory_size: int, descriptor_size: int, cap: int
        ) -> tuple[int, int] | None:
            if not rva or directory_size < descriptor_size:
                return None
            span = _rva_span(
                rva,
                sections,
                headers_size=headers_size,
                file_size=file_size,
            )
            if span is None:
                return None
            offset, available = span
            count = min(directory_size, available) // descriptor_size
            return offset, min(count, cap)

        def dll_name(name_rva: int) -> str | None:
            span = _rva_span(
                name_rva,
                sections,
                headers_size=headers_size,
                file_size=file_size,
            )
            if span is None:
                return None
            name_offset, available = span
            try:
                name = _read_cstring(fh, name_offset, min(available, 4096))
            except PEError:
                return None
            return name or None

        # Directory 1: imports. Each descriptor is 20 bytes, DLL name RVA at +12.
        imp_rva, imp_size = directory(1)
        imp_range = descriptor_range(imp_rva, imp_size, 20, 4096)
        if imp_range is not None:
            off, count = imp_range
            for i in range(count):
                desc = _read_at(fh, off + i * 20, 20)
                if desc == b"\0" * 20:
                    break
                (name_rva,) = struct.unpack("<I", desc[12:16])
                name = dll_name(name_rva)
                if name is not None:
                    info.imports.append(name)

        # Directory 13: delay-load imports. 32 bytes each, name RVA at +4.
        # With dlattrRva (bit 0) clear, old linkers stored a VA and it must be
        # rebased by ImageBase before mapping.  The descriptor fields themselves
        # remain DWORDs even in PE32+.
        delay_rva, delay_size = directory(13)
        delay_range = descriptor_range(delay_rva, delay_size, 32, 1024)
        if delay_range is not None:
            off, count = delay_range
            for i in range(count):
                desc = _read_at(fh, off + i * 32, 32)
                if desc == b"\0" * 32:
                    break
                attrs, name_value = struct.unpack("<II", desc[:8])
                if attrs & 1:
                    name_rva = name_value
                elif image_base <= name_value:
                    name_rva = name_value - image_base
                else:
                    continue
                name = dll_name(name_rva)
                if name is not None:
                    info.delay_imports.append(name)

        return info


def graphics_apis(info: PEInfo) -> list[dict]:
    """Map a PE's imports onto apitrace APIs, in import-table order; rank with
    server-side priority."""
    found = []
    for dll in info.all_imports:
        if dll in GRAPHICS_DLLS:
            api, wrapper = GRAPHICS_DLLS[dll]
            found.append(
                {
                    "dll": dll,
                    "api": api,
                    "wrapper_dll": wrapper,
                    "supported": api is not None,
                }
            )
    return found
