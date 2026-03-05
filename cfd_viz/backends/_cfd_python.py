"""cfd-python-based backend implementations."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..cfd_python_integration import get_cfd_python, get_cfd_python_version
from ._base import StatsBackend, SystemInfoBackend


class CfdPythonStatsBackend(StatsBackend):
    """Statistics backend using cfd-python's optimized implementations."""

    def calculate_field_stats(
        self, data: Union[NDArray, List[float]]
    ) -> Dict[str, float]:
        cfd = get_cfd_python()
        flat = data.flatten().tolist() if hasattr(data, "flatten") else list(data)
        return cfd.calculate_field_stats(flat)

    def compute_flow_statistics(
        self,
        u: NDArray,
        v: NDArray,
        nx: int,
        ny: int,
        p: Optional[NDArray],
    ) -> Dict[str, Dict[str, float]]:
        cfd = get_cfd_python()
        u_flat = u.flatten().tolist()
        v_flat = v.flatten().tolist()
        if p is not None:
            p_flat = p.flatten().tolist()
        else:
            # cfd-python requires p, use zeros as placeholder
            p_flat = [0.0] * len(u_flat)

        result = cfd.compute_flow_statistics(u_flat, v_flat, p_flat, nx, ny)

        # Remove p stats if original data had no pressure
        if p is None and "p" in result:
            del result["p"]

        return result

    def compute_velocity_magnitude(
        self, u: NDArray, v: NDArray, nx: int, ny: int
    ) -> NDArray:
        cfd = get_cfd_python()
        u_flat = u.flatten().tolist()
        v_flat = v.flatten().tolist()
        mag_flat = cfd.compute_velocity_magnitude(u_flat, v_flat, nx, ny)
        return np.array(mag_flat).reshape((ny, nx))


class CfdPythonSystemInfoBackend(SystemInfoBackend):
    """System info backend using cfd-python's capability queries."""

    def get_system_info(self) -> Dict[str, Any]:
        cfd = get_cfd_python()
        simd = cfd.get_simd_name()
        return {
            "cfd_python_available": True,
            "cfd_python_version": get_cfd_python_version(),
            "backends": cfd.get_available_backends(),
            "simd": simd,
            "has_simd": simd != "none",
            "has_avx2": cfd.has_avx2(),
            "has_neon": cfd.has_neon(),
            "gpu_available": cfd.backend_is_available(cfd.BACKEND_CUDA),
        }
