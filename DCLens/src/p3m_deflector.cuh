#pragma once
// P3MDeflector: ray propagation using PM interpolation + PP correction, or PP.
// Included by ipm.cu only; retain the original device function linkage.
#include "ipm.cuh"

// ray_shoot_device — selectable PP/P3M deflection
// Multi-plane inverse ray shooting with GPU PP or P3M lens-force calculation.
// Internal units: xi in pc, Lens.x/y in pc, Lens.M in Msun.
// Geometric field A has units Msun/pc, physical deflection = KAPPA_CONST * A.
__device__ Point ray_shoot_device(double theta_x_arcsec, double theta_y_arcsec,
                                 const Lens* all_lenses,
                                 const Bucket* all_buckets,
                                 const double* pm_Ax,
                                 const double* pm_Ay,
                                 const PlaneInfo* planes,
                                 int num_planes) {

    constexpr double ARCSEC2RAD = 4.84813681109536e-6;
    constexpr double RAD2ARCSEC = 206264.806247096;
    constexpr double G_val = 6.67430e-11;
    constexpr double M_sun = 1.98847e30;
    constexpr double c_val = 299792458.0;
    constexpr double pc_in_m = 3.085677581e16;
    constexpr double KAPPA_CONST = (4.0 * G_val * M_sun) / (c_val * c_val * pc_in_m);

    double theta_x = theta_x_arcsec * ARCSEC2RAD;
    double theta_y = theta_y_arcsec * ARCSEC2RAD;

    double beta_x = theta_x;
    double beta_y = theta_y;
    double gamma_x = theta_x;
    double gamma_y = theta_y;
    double xi_x = 0.0;
    double xi_y = 0.0;
    double current_z = 0.0;

    for (int i = 0; i < num_planes; ++i) {
        const PlaneInfo& p = planes[i];

        // 1. Propagate to current plane
        double dz = p.z - current_z;
        xi_x += gamma_x * dz;
        xi_y += gamma_y * dz;
        current_z = p.z;

        // 2. Compute geometric deflection field A via P3M
        double A_x = 0.0;
        double A_y = 0.0;

        // Short-circuit before accessing absent PM arrays in pure PP mode.
        bool pm_ok = p.use_p3m &&
            interpolate_pm_tsc(xi_x, xi_y, p, pm_Ax, pm_Ay, A_x, A_y);

        if (pm_ok) {
            // Add PP short-range correction
            add_pp_correction(xi_x, xi_y, p, all_lenses, all_buckets, A_x, A_y);
        } else {
            // Pure PP or P3M outside mesh support: sum every lens in this plane.
            A_x = 0.0;
            A_y = 0.0;
            direct_point_field_plane(xi_x, xi_y, p, all_lenses, A_x, A_y);
        }

        // 3. Convert geometric field to physical deflection
        double phyalpha_x_sum = KAPPA_CONST * A_x;
        double phyalpha_y_sum = KAPPA_CONST * A_y;

        // 4. Update beta (source position)
        const double weight = p.factor;
        beta_x -= weight * phyalpha_x_sum;
        beta_y -= weight * phyalpha_y_sum;

        // 5. Update gamma (propagation direction)
        gamma_x -= phyalpha_x_sum;
        gamma_y -= phyalpha_y_sum;
    }

    return {beta_x * RAD2ARCSEC, beta_y * RAD2ARCSEC};
}

