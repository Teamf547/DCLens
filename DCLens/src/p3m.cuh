#pragma once
#include "cuda_memory.cuh"
#include <cmath>
#include <vector>

struct Lens {
    double x, y, M; // x,y in pc, M in Msun
};

struct Bucket {
    int lens_start_idx; // relative to current plane's lens array
    int lens_count;
};

struct PlaneInfo {
    bool use_p3m;  // False: whole-plane direct PP, no mesh/buckets allocated.
    double z;       // lens-plane distance [pc]
    double factor;  // precomputed projection factor (D_S - D_i) / D_S

    int lens_start_idx;  // global lens offset
    int lens_count;

    int bucket_start_idx;  // global PP bucket offset

    // PP bucket geometry
    int pp_nx;
    int pp_ny;
    double pp_xmin;
    double pp_ymin;
    double pp_h;       // bucket size = a_pm

    // PM mesh
    size_t pm_offset;  // global PM field offset

    int pm_nx;
    int pm_ny;
    double pm_xmin;
    double pm_ymin;
    double H;          // PM mesh spacing [pc]
    double a_pm;       // PP/PM split radius [pc]
};


// Input positions are pc, masses Msun. These arrays are replaced, never
// appended, when the engine advances to the next lens snapshot.
struct PlaneData {
    Lens* d_all_lenses = nullptr;
    Bucket* d_all_buckets = nullptr;
    double* d_pm_Ax = nullptr;
    double* d_pm_Ay = nullptr;
    PlaneInfo* d_planes = nullptr;
    int num_planes = 0;
    bool verbose;
    explicit PlaneData(bool verbose=false) : verbose(verbose) {}
    PlaneData(const PlaneData&) = delete;
    PlaneData& operator=(const PlaneData&) = delete;
    ~PlaneData() { clear(); }
    void clear() noexcept;
    void load_pp(const std::vector<std::vector<double>>& lx,
                 const std::vector<std::vector<double>>& ly,
                 const std::vector<std::vector<double>>& mass,
                 const std::vector<double>& zs, const std::vector<double>& factors);
    void load(const std::vector<std::vector<double>>& lx,
              const std::vector<std::vector<double>>& ly,
              const std::vector<std::vector<double>>& lM,
              const std::vector<std::vector<int>>& bucket_start,
              const std::vector<std::vector<int>>& bucket_count,
              const std::vector<std::vector<double>>& p3m_params,
              const std::vector<double>& zs, const std::vector<double>& factors);
};

// --- TSC (Triangular Shaped Cloud) weight function ---
__device__ inline double tsc_weight(double q) {
    double aq = fabs(q);
    if (aq < 0.5) return 0.75 - q * q;
    if (aq < 1.5) { double t = 1.5 - aq; return 0.5 * t * t; }
    return 0.0;
}

// Interpolate the row-major PM field with the same 3x3 TSC shape used for
// mass assignment. False requests the exact whole-plane fallback.
__device__ inline bool interpolate_pm_tsc(
    double x, double y,
    const PlaneInfo& p,
    const double* pm_Ax, const double* pm_Ay,
    double& Ax, double& Ay)
{
    double u = (x - p.pm_xmin) / p.H;
    double v = (y - p.pm_ymin) / p.H;

    int ic = (int)floor(u + 0.5);
    int jc = (int)floor(v + 0.5);

    // Check bounds: need ic-1..ic+1 and jc-1..jc+1 all valid
    if (ic < 1 || ic >= p.pm_nx - 1 || jc < 1 || jc >= p.pm_ny - 1)
        return false;

    double wx[3];
    double wy[3];
    for (int d = -1; d <= 1; ++d) {
        wx[d + 1] = tsc_weight(u - (double)(ic + d));
        wy[d + 1] = tsc_weight(v - (double)(jc + d));
    }

    Ax = 0.0;
    Ay = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        int jj = jc + dy;
        for (int dx = -1; dx <= 1; ++dx) {
            int ii = ic + dx;
            double w = wx[dx + 1] * wy[dy + 1];
            size_t idx = p.pm_offset + (size_t)jj * p.pm_nx + ii;
            Ax += w * pm_Ax[idx];
            Ay += w * pm_Ay[idx];
        }
    }
    return true;
}

// f(r;a) is the softened radial field magnitude per unit mass. Vector form:
// A_soft = M f(r;a) r_hat = M [f(r;a)/r] r_vec. Consequently the PP
// correction is A_corr = M [1/r^2 - f(r;a)/r] r_vec.
__device__ inline double p3m_soft_force_scalar(double r, double a) {
    double xi = 2.0 * r / a;
    if (xi < 1.0) {
        double xi2 = xi * xi;
        double xi3 = xi2 * xi;
        double xi4 = xi3 * xi;
        return (1.0 / a) * (0.743081 * xi4 - 1.832992 * xi3
                           - 0.057116 * xi2 + 2.672707 * xi - 0.000082);
    } else if (xi < 2.0) {
        double xi2 = xi * xi;
        double xi3 = xi2 * xi;
        double xi4 = xi3 * xi;
        return (1.0 / a) * (-0.524104 * xi4 + 4.081230 * xi3
                           - 11.856245 * xi2 + 15.107021 * xi
                           - 6.823192 + 1.539967 / xi);
    } else {
        return 1.0 / r;
    }
}

// --- PP correction: add (point - PM_soft) for lenses within a_pm ---
__device__ inline void add_pp_correction(
    double xi_x, double xi_y,
    const PlaneInfo& p,
    const Lens* all_lenses,
    const Bucket* all_buckets,
    double& Ax, double& Ay)
{
    int bx0 = (int)floor((xi_x - p.pp_xmin) / p.pp_h);
    int by0 = (int)floor((xi_y - p.pp_ymin) / p.pp_h);

    double a_pm2 = p.a_pm * p.a_pm;
    constexpr double EPS_R2 = 1e-12;

    for (int oy = -1; oy <= 1; ++oy) {
        int by = by0 + oy;
        if (by < 0 || by >= p.pp_ny) continue;
        for (int ox = -1; ox <= 1; ++ox) {
            int bx = bx0 + ox;
            if (bx < 0 || bx >= p.pp_nx) continue;

            int bucket_flat = by * p.pp_nx + bx;
            int bidx = p.bucket_start_idx + bucket_flat;
            const Bucket& b = all_buckets[bidx];

            for (int j = 0; j < b.lens_count; ++j) {
                const Lens& l = all_lenses[p.lens_start_idx + b.lens_start_idx + j];
                double rx = xi_x - l.x;
                double ry = xi_y - l.y;
                double r2 = rx * rx + ry * ry;

                if (r2 >= a_pm2) continue; // outside cutoff

                r2 = fmax(r2, EPS_R2); // singularity protection
                double r = sqrt(r2);
                double fpm = p3m_soft_force_scalar(r, p.a_pm);

                // correction = point_force - PM_softened_force
                // point: M/r^2, softened: M*fpm/r
                double coeff = l.M * (1.0 / r2 - fpm / r);
                Ax += coeff * rx;
                Ay += coeff * ry;
            }
        }
    }
}

// Whole-plane direct PP, also used when P3M interpolation is out of range.
__device__ inline void direct_point_field_plane(
    double xi_x, double xi_y,
    const PlaneInfo& p,
    const Lens* all_lenses,
    double& Ax, double& Ay)
{
    constexpr double EPS_R2 = 1e-12;
    Ax = 0.0;
    Ay = 0.0;
    for (int j = 0; j < p.lens_count; ++j) {
        const Lens& l = all_lenses[p.lens_start_idx + j];
        double rx = xi_x - l.x;
        double ry = xi_y - l.y;
        double r2 = rx * rx + ry * ry;
        r2 = fmax(r2, EPS_R2);
        double inv_r2 = l.M / r2;
        Ax += inv_r2 * rx;
        Ay += inv_r2 * ry;
    }
}

