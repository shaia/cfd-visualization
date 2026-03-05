"""Abstract base classes for computation backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from numpy.typing import NDArray


class StatsBackend(ABC):
    """Interface for field statistics computations."""

    @abstractmethod
    def calculate_field_stats(
        self, data: Union[NDArray, List[float]]
    ) -> Dict[str, float]:
        """Calculate field statistics.

        Returns:
            Dict with keys: min, max, avg, sum.
        """
        ...

    @abstractmethod
    def compute_flow_statistics(
        self,
        u: NDArray,
        v: NDArray,
        nx: int,
        ny: int,
        p: Optional[NDArray],
    ) -> Dict[str, Dict[str, float]]:
        """Compute flow statistics for all fields.

        Returns:
            Dict with keys: u, v, velocity_magnitude, and optionally p.
            Each value is a dict with keys: min, max, avg, sum.
        """
        ...

    @abstractmethod
    def compute_velocity_magnitude(
        self, u: NDArray, v: NDArray, nx: int, ny: int
    ) -> NDArray:
        """Compute velocity magnitude field.

        Returns:
            2D array of velocity magnitudes shaped (ny, nx).
        """
        ...


class SystemInfoBackend(ABC):
    """Interface for system information queries."""

    @abstractmethod
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information.

        Returns:
            Dict with keys: cfd_python_available, cfd_python_version,
            backends, simd, has_simd, has_avx2, has_neon, gpu_available.
        """
        ...
