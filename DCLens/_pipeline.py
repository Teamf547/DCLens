"""Shared plane preparation and upload for probing and simulation."""
from .physics.pm_builder import (split_into_planes, PMBuilder,
                                 _prepare_engine_planes)


def _bounds(config, buffer):
    xc, yc = config.img_center
    width, height = config.src_considered_region
    return (xc-width/2-buffer, xc+width/2+buffer,
            yc-height/2-buffer, yc+height/2+buffer)


def _prepare_planes(table, lens, config, bounds):
    planes = split_into_planes(table, lens.N_planes)
    if config.deflector == 'pp':
        return planes
    return PMBuilder(
        planes, grid_nx=config.PM_grid,
        grid_ny=config.PM_grid, theta_bounds_arcsec=bounds,
        a_pm_over_H=config.p3m_a_pm_over_H,
        padding_factor=config.p3m_padding_factor)


def _upload_planes(engine, planes, lens, config):
    if config.deflector == 'p3m':
        engine.load_planes(*_prepare_engine_planes(planes, lens.D_S_pc))
    else:
        # Direct PP needs physical lenses and projection geometry only.
        # No dummy PM grids or short-range bucket arrays are constructed.
        coordinates = [[p['lenses'][key] for p in planes]
                       for key in ('x_pc', 'y_pc', 'M_Msun')]
        engine.load_pp_planes(
            *coordinates, [p['z_pc'] for p in planes],
            [(lens.D_S_pc-p['z_pc'])/lens.D_S_pc for p in planes])


