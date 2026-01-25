import triton
from functools import lru_cache
import subprocess
import os


@lru_cache(maxsize=1)
def get_arch():
    try:
        arch = (
            triton.runtime.driver.active.get_current_target().arch
        )  # If running with torch
    except RuntimeError:  # else try rocminfo fallback
        # Try to get arch from rocminfo (for ROCm/HIP)
        try:
            result = subprocess.run(
                ["rocminfo"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if "gfx" in line.lower():
                    arch = line.split(":")[-1].strip()
                    return arch
        except Exception:
            pass
        
        # Try JAX as last resort
        try:
            from jax._src.lib import gpu_triton as triton_kernel_call_lib
            arch = triton_kernel_call_lib.get_arch_details("0")
            arch = arch.split(":")[0]
        except ImportError:
            # Default to gfx950 for MI355X if we can't detect
            arch = os.environ.get("GPU_ARCHS", "gfx950")

    return arch


def is_fp4_avail():
    return get_arch() in ("gfx950")


def is_fp8_avail():
    return get_arch() in ("gfx942", "gfx950")
