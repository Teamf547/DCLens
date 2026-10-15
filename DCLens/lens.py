"""Lazy lens snapshots: masses in Msun, coordinates and distances in pc."""
from pathlib import Path
import re

import numpy as np
import pandas as pd


def _natural_key(path):
    return [int(s) if s.isdigit() else s.casefold()
            for s in re.split(r'(\d+)', path.name)]


class Lens:
    """Describe a file, naturally sorted directory, or explicitly ordered list.

    Velocities are input columns only. A sequence contains already evolved
    snapshots; this class never integrates an orbit or retains loaded frames.
    """
    def __init__(self, source, D_L_pc=1e4, D_S_pc=1e9, N_planes=1,
                 pattern='*.dat'):
        if not np.isfinite([D_L_pc, D_S_pc]).all() or not 0 < D_L_pc < D_S_pc:
            raise ValueError('Require finite distances 0 < D_L_pc < D_S_pc')
        if not isinstance(N_planes, (int, np.integer)) or N_planes < 1:
            raise ValueError('N_planes must be a positive integer')
        self.source = source
        self.D_L_pc, self.D_S_pc = float(D_L_pc), float(D_S_pc)
        self.N_planes = int(N_planes)
        if isinstance(source, (str, Path)):
            path = Path(source).expanduser()
            self.mode = 'sequence' if path.is_dir() else 'single'
            files = sorted((p for p in path.glob(pattern) if p.is_file()),
                           key=lambda p: (_natural_key(p), p.name)) if path.is_dir() else [path]
        else:
            self.mode = 'sequence'
            files = [Path(p).expanduser() for p in source]
        if not files:
            raise ValueError('No lens snapshots found')
        self.files = tuple(p.resolve() for p in files)
        for path in self.files:
            if not path.is_file():
                raise FileNotFoundError(path)

    @property
    def frame_count(self):
        return len(self.files)

    def load_frame(self, i):
        """Read one seven-column snapshot, preserving the original z convention."""
        df = pd.read_csv(self.files[i], sep=r'\s+', header=None, skiprows=1,
                         usecols=range(7),
                         names=['M_Msun', 'x_pc', 'y_pc', 'z_pc', 'vx', 'vy', 'vz'])
        table = df[['x_pc', 'y_pc', 'M_Msun', 'z_pc']].astype(np.float64).copy()
        if not np.isfinite(table.to_numpy()).all():
            raise ValueError(f'Non-finite lens data: {self.files[i]}')
        # File z is relative to the system distance. The existing convention
        # discards lenses behind the observer after converting to absolute pc.
        table['z_pc'] += self.D_L_pc
        return table.loc[table['z_pc'] > 0].reset_index(drop=True)

    def iter_frames(self):
        for i in range(self.frame_count):
            yield self.load_frame(i)
