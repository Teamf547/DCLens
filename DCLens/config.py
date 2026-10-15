"""Simulation controls, separate from the physical lens-system description."""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class SimulationConfig:
    coarse_ps: float = 1e-3
    finest_ps: float = 1e-4
    distortion_threshold: float = 0.2
    PM_grid: int = 2048
    img_center: tuple = (0.0, 0.0)
    src_considered_region: tuple = (0.1, 0.1)
    buffer_size: float = 0.1
    max_refine_iters: int = 20
    srcp_ps: float = 1e-4
    ss: int = 4
    max_coarse_per_chunk: int = 200
    # Two 48-byte Pixel queues + one 80-byte ResultLeaf queue:
    # 20,000,000 * (2*48 + 80) = 3.52 GB (3.28 GiB), allocated only for IPM.
    max_active_cells: int = 20_000_000
    p3m_a_pm_over_H: float = 6.0
    p3m_padding_factor: float = 2.0
    show_progress: bool = True
    verbose: bool = False
    deflector: str = 'p3m'

    def __post_init__(self):
        if self.deflector not in ('pp', 'p3m'):
            raise ValueError("deflector must be 'pp' or 'p3m'")
        for name in ('coarse_ps', 'finest_ps', 'srcp_ps', 'p3m_a_pm_over_H',
                     'p3m_padding_factor'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('PM_grid', 'max_refine_iters', 'ss', 'max_coarse_per_chunk',
                     'max_active_cells'):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('buffer_size', 'distortion_threshold'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if len(self.img_center) != 2 or not all(map(math.isfinite, self.img_center)):
            raise ValueError('img_center must contain two finite angles')
        if len(self.src_considered_region) != 2 or not all(
                math.isfinite(x) and x > 0 for x in self.src_considered_region):
            raise ValueError('src_considered_region must contain positive width and height')
        if self.max_active_cells > (2**31 - 1) // 4:
            raise ValueError('max_active_cells exceeds CUDA queue counter range')


def _resolve_config(config, keywords):
    return replace(config or SimulationConfig(), **keywords)
