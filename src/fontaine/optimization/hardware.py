"""Detect this machine's CPU and RAM and pick a Yami size tier.

Detection uses only the standard library: ``os.cpu_count`` for logical
cores, ``os.sysconf`` on Linux/macOS, and ``GlobalMemoryStatusEx`` on
Windows. Anything that cannot be read is reported as ``None`` and the
recommendation falls back to core count alone.
"""

import os
import sys
from dataclasses import dataclass

import torch

from fontaine.optimization.quantize import cpu_supports_bf16

YAMI_TIERS = ("nano", "small", "base", "large")
TIER_CONFIGS = {tier: f"configs/model/yami_{tier}.yaml" for tier in YAMI_TIERS}

_GIB = 1024**3


@dataclass(frozen=True)
class HardwareInfo:
    logical_cores: int | None
    total_ram_bytes: int | None
    cpu_capability: str
    bf16: bool
    cuda: bool

    @property
    def total_ram_gib(self) -> float | None:
        if self.total_ram_bytes is None:
            return None
        return self.total_ram_bytes / _GIB


def _windows_total_ram() -> int | None:
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return None
    return int(status.ullTotalPhys)


def total_ram_bytes() -> int | None:
    """Physical RAM in bytes, or None when the platform does not report it."""
    try:
        if sys.platform == "win32":
            return _windows_total_ram()
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        return None
    return None


def detect_hardware() -> HardwareInfo:
    try:
        capability = torch.backends.cpu.get_cpu_capability()
    except (AttributeError, RuntimeError):
        capability = "unknown"
    return HardwareInfo(
        logical_cores=os.cpu_count(),
        total_ram_bytes=total_ram_bytes(),
        cpu_capability=str(capability),
        bf16=cpu_supports_bf16(),
        cuda=torch.cuda.is_available(),
    )


def recommend_tier(info: HardwareInfo) -> str:
    """Largest Yami tier this machine can run comfortably with int8 weights.

    RAM sets the ceiling and core count sets the speed. Nano runs anywhere;
    Large wants 32 GB and 12+ cores. Thresholds sit a little below 8/16/32
    GiB because the OS reserves part of the installed RAM.
    """
    cores = info.logical_cores or 1
    ram = info.total_ram_gib
    if ram is None:
        if cores >= 16:
            return "base"
        return "small" if cores > 4 else "nano"
    if ram < 7.5 or cores <= 4:
        return "nano"
    if ram < 15:
        return "small"
    if ram < 30:
        return "base" if cores >= 8 else "small"
    if cores >= 12:
        return "large"
    return "base" if cores >= 8 else "small"
