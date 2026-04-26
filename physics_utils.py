# physics_utils.py

import numpy as np
try:
    import cupy as cp
    _HAVE_CUPY = True
except Exception:
    cp = None
    _HAVE_CUPY = False

from .constants import G, M_sun, pc_in_m, c, H_0

def pc_to_redshift(distance_pc: float) -> float:
    # distance_pc -> m, H_0 in km/s/Mpc -> convert to SI consistent estimate
    return ((H_0 * distance_pc / 1e6) * 1e3) / c

def generate_BH_size_samples(mass_array, D_L, D_S):

    mass_array = np.asarray(mass_array)
    theta_radians = np.sqrt((4 * G * mass_array * M_sun / c**2 / pc_in_m) * ((D_S - D_L) / (D_S * D_L)))
    return theta_radians * (180.0 / np.pi) * 3600.0 # arcsec

def ray_shoot_batched(theta_images, lens_planes, D_S, batch_size=200000, use_gpu=None):

    if theta_images is None:
        return np.zeros((0,2), dtype=np.float64)
    N = int(theta_images.shape[0])
    out = np.empty((N, 2), dtype=np.float64)
    # decide backend
    if use_gpu is None:
        use_gpu = (_HAVE_CUPY)
    tiny = 1e-12
    start = 0

    # prepare planar arrays for CPU/GPU (we copy per plane later)
    if use_gpu:
        lens_planes_gpu = []
        for plane in lens_planes:
            lens_planes_gpu.append({
                'z': float(plane['z']),
                'lx': cp.asarray(plane['lenses']['x'], dtype=cp.float64),
                'ly': cp.asarray(plane['lenses']['y'], dtype=cp.float64),
                'lE': cp.asarray(plane['lenses']['theta_E'], dtype=cp.float64)
            })
    else:
        lens_planes_cpu = []
        for plane in lens_planes:
            lens_planes_cpu.append({
                'z': float(plane['z']),
                'lx': np.asarray(plane['lenses']['x'], dtype=np.float64),
                'ly': np.asarray(plane['lenses']['y'], dtype=np.float64),
                'lE': np.asarray(plane['lenses']['theta_E'], dtype=np.float64)
            })

    while start < N:
        end = min(N, start + batch_size)
        batch = theta_images[start:end]
        if use_gpu:
            beta = cp.asarray(batch, dtype=cp.float64)
            for plane in lens_planes_gpu:
                D_i = float(plane['z'])
                D_iS = D_S - D_i
                lx = plane['lx']
                ly = plane['ly']
                lE = plane['lE']
                rx = beta[:, 0:1]
                ry = beta[:, 1:2]
                dx = rx - lx[None, :]
                dy = ry - ly[None, :]
                r2 = dx * dx + dy * dy
                r2 = cp.where(r2 == 0, tiny, r2)
                E2 = (lE * lE)[None, :]
                ax = E2 * dx / r2
                ay = E2 * dy / r2
                ax_sum = cp.sum(ax, axis=1)
                ay_sum = cp.sum(ay, axis=1)
                beta = beta - (D_iS / D_S) * cp.stack([ax_sum, ay_sum], axis=1)
            out[start:end] = cp.asnumpy(beta)
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        else:
            beta = np.asarray(batch, dtype=np.float64)
            for plane in lens_planes_cpu:
                D_i = float(plane['z'])
                D_iS = D_S - D_i
                lx = plane['lx']
                ly = plane['ly']
                lE = plane['lenses']['theta_E'] if 'theta_E' in plane['lenses'] else plane['lE'] if 'lE' in plane else plane.get('lE')
                # above line is defensive; normal case uses plane['lE'] or lE var from plane dict
                lx = plane['lx']
                ly = plane['ly']
                lE = plane['lE']
                rx = beta[:, 0:1]
                ry = beta[:, 1:2]
                dx = rx - lx[None, :]
                dy = ry - ly[None, :]
                r2 = dx * dx + dy * dy
                r2 = np.where(r2 == 0, tiny, r2)
                E2 = (lE * lE)[None, :]
                ax = E2 * dx / r2
                ay = E2 * dy / r2
                ax_sum = np.sum(ax, axis=1)
                ay_sum = np.sum(ay, axis=1)
                beta = beta - (D_iS / D_S) * np.stack([ax_sum, ay_sum], axis=1)
            out[start:end] = beta
        start = end
    return out
