"""NumPy-based backend implementations (always available)."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ._base import StatsBackend, SystemInfoBackend


class NumPyStatsBackend(StatsBackend):
    """Statistics backend using NumPy (fallback when cfd-python is unavailable)."""

    def calculate_field_stats(
        self, data: Union[NDArray, List[float]]
    ) -> Dict[str, float]:
        arr = np.asarray(data)
        return {
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "avg": float(np.nanmean(arr)),
            "sum": float(np.nansum(arr)),
        }

    def compute_flow_statistics(
        self,
        u: NDArray,
        v: NDArray,
        nx: int,
        ny: int,
        p: Optional[NDArray],
    ) -> Dict[str, Dict[str, float]]:
        vel_mag = np.sqrt(u**2 + v**2)
        result: Dict[str, Dict[str, float]] = {
            "u": self.calculate_field_stats(u),
            "v": self.calculate_field_stats(v),
            "velocity_magnitude": self.calculate_field_stats(vel_mag),
        }
        if p is not None:
            result["p"] = self.calculate_field_stats(p)
        return result

    def compute_velocity_magnitude(
        self, u: NDArray, v: NDArray, nx: int, ny: int
    ) -> NDArray:
        return np.sqrt(u**2 + v**2)


class NumPySystemInfoBackend(SystemInfoBackend):
    """System info backend when cfd-python is not available."""

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "cfd_python_available": False,
            "cfd_python_version": None,
            "backends": [],
            "simd": "unknown",
            "has_simd": False,
            "has_avx2": False,
            "has_neon": False,
            "gpu_available": False,
        }
