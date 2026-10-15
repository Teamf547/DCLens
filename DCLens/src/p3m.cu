// PMBuilder GPU stage: mass assignment, FFT field construction and plane upload.
// TSC deposition, optimized S1 Green function and PP/PM preprocessing.
#include "p3m.cuh"
#include <cuComplex.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ inline double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#define ATOMIC_ADD_DOUBLE(addr, val) atomicAddDouble(addr, val)
#else
#define ATOMIC_ADD_DOUBLE(addr, val) atomicAdd(addr, val)
#endif

// --- TSC mass deposit: one thread per lens ---

__global__ void tsc_deposit_kernel(
    const Lens* lenses, int n_lens,
    double* rho,
    int nx, int ny,
    double xmin, double ymin, double H)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_lens) return;

    const Lens& l = lenses[tid];
    double u = (l.x - xmin) / H;
    double v = (l.y - ymin) / H;

    int ic = (int)floor(u + 0.5);
    int jc = (int)floor(v + 0.5);

    double invH2 = 1.0 / (H * H);

    for (int dy = -1; dy <= 1; ++dy) {
        int jj = jc + dy;
        if (jj < 0 || jj >= ny) continue;
        double qy = v - (double)jj;
        double wy = tsc_weight(qy);
        for (int dx = -1; dx <= 1; ++dx) {
            int ii = ic + dx;
            if (ii < 0 || ii >= nx) continue;
            double qx = u - (double)ii;
            double wx = tsc_weight(qx);
            ATOMIC_ADD_DOUBLE(&rho[jj * nx + ii], l.M * wx * wy * invH2);
        }
    }
}

// --- Safe sinc: sinc(x) = sin(x)/x with Taylor series for small x ---
__device__ inline double sinc_safe(double x) {
    double ax = fabs(x);
    if (ax < 1e-8) return 1.0;
    if (ax < 1e-4) {
        double x2 = x * x;
        return 1.0 - x2 / 6.0 + x2 * x2 / 120.0;
    }
    return sin(x) / x;
}

// --- S1 shape Fourier transform (Xu & Jing) ---
// Returns S1_hat(k, a). Small-z Taylor expansion avoids catastrophic cancellation.
__device__ inline double s1_hat(double k, double a) {
    double z = 0.5 * k * a;  // z = ka/2
    if (fabs(z) < 1e-3) {
        double z2 = z * z;
        return 1.0 - z2 / 12.0 + z2 * z2 / 384.0;
    }
    // S1_hat = 16*J1(z)/z^3 - 8*J0(z)/z^2
    double j0z = j0(z);
    double j1z = j1(z);
    return 16.0 * j1z / (z * z * z) - 8.0 * j0z / (z * z);
}

// --- Optimized Green function influence kernel ---
// Multiplies rho_k by the optimized P3M kernel to produce Ax_k, Ay_k
// Each thread handles one Fourier mode
__global__ void apply_p3m_influence_kernel(
    const cufftDoubleComplex* rho_k,
    cufftDoubleComplex* Ax_k,
    cufftDoubleComplex* Ay_k,
    int nx, int ny,
    double H, double a_pm)
{
    int nkx = nx / 2 + 1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ncomplex = ny * nkx;
    if (tid >= ncomplex) return;

    int iy = tid / nkx;
    int ix = tid % nkx;

    // DC mode: zero
    if (ix == 0 && iy == 0) {
        Ax_k[tid] = make_cuDoubleComplex(0.0, 0.0);
        Ay_k[tid] = make_cuDoubleComplex(0.0, 0.0);
        return;
    }

    double TWO_PI = 2.0 * M_PI;
    double k_g = TWO_PI / H;

    double kx = TWO_PI * ix / ((double)nx * H);
    int ky_index = (iy <= ny / 2) ? iy : iy - ny;
    double ky = TWO_PI * ky_index / ((double)ny * H);

    // Finite-difference operator (beta_fd = 4/3, p=2)
    double beta_fd = 4.0 / 3.0;
    double kxH = kx * H;
    double kyH = ky * H;
    double dx_op = beta_fd * sin(kxH) / H + (1.0 - beta_fd) * sin(2.0 * kxH) / (2.0 * H);
    double dy_op = beta_fd * sin(kyH) / H + (1.0 - beta_fd) * sin(2.0 * kyH) / (2.0 * H);

    double d2 = dx_op * dx_op + dy_op * dy_op;
    if (d2 < 1e-30) {
        Ax_k[tid] = make_cuDoubleComplex(0.0, 0.0);
        Ay_k[tid] = make_cuDoubleComplex(0.0, 0.0);
        return;
    }

    // Fixed 3x3 alias sum
    double sum_u2 = 0.0;
    double sum_rx = 0.0;
    double sum_ry = 0.0;

    for (int ay = -1; ay <= 1; ++ay) {
        for (int ax = -1; ax <= 1; ++ax) {
            double knx = kx - ax * k_g;
            double kny = ky - ay * k_g;
            double kn2 = knx * knx + kny * kny;

            if (kn2 < 1e-30) continue;

            // TSC window: U = [sinc(knx*H/2) * sinc(kny*H/2)]^3
            double Ux = sinc_safe(knx * H * 0.5);
            double Uy = sinc_safe(kny * H * 0.5);
            double Un = Ux * Ux * Ux * Uy * Uy * Uy;  // TSC = sinc^3
            double u2 = Un * Un;

            // S1 Fourier transform
            double kn = sqrt(kn2);
            double S = s1_hat(kn, a_pm);

            // Reference force (imaginary part): R*_im = 2*pi * kn_i/kn^2 * S^2
            double S2 = S * S;
            double rstar_im_x = TWO_PI * knx / kn2 * S2;
            double rstar_im_y = TWO_PI * kny / kn2 * S2;

            sum_u2 += u2;
            sum_rx += u2 * rstar_im_x;
            sum_ry += u2 * rstar_im_y;
        }
    }

    if (sum_u2 < 1e-30) {
        Ax_k[tid] = make_cuDoubleComplex(0.0, 0.0);
        Ay_k[tid] = make_cuDoubleComplex(0.0, 0.0);
        return;
    }

    // Optimized Green function:
    // G_opt = (D . sum(U^2 R*)) / (|D|^2 * (sum U^2)^2)
    // Since D = i*d and R* = i*r_im, D.R* = -(d.r_im) [real]
    double numerator = -(dx_op * sum_rx + dy_op * sum_ry);
    double denominator = d2 * sum_u2 * sum_u2;
    double Gopt = numerator / denominator;

    // K = i * d * Gopt (the full kernel is purely imaginary)
    double kim_x = dx_op * Gopt;
    double kim_y = dy_op * Gopt;

    // Multiply: Ax_k = K_x * rho_k, where K_x = (0, kim_x)
    cufftDoubleComplex Kx = make_cuDoubleComplex(0.0, kim_x);
    cufftDoubleComplex Ky = make_cuDoubleComplex(0.0, kim_y);

    // cuCmul: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    Ax_k[tid] = cuCmul(Kx, rho_k[tid]);
    Ay_k[tid] = cuCmul(Ky, rho_k[tid]);
}

// --- Normalize PM field after inverse FFT (cuFFT doesn't normalize) ---
__global__ void normalize_pm_kernel(double* field, size_t n, double invN) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    field[tid] *= invN;
}

void PlaneData::clear() noexcept {
    if (d_all_lenses) cudaFree(d_all_lenses);
    d_all_lenses = nullptr;
    if (d_all_buckets) cudaFree(d_all_buckets);
    d_all_buckets = nullptr;
    if (d_pm_Ax) cudaFree(d_pm_Ax);
    d_pm_Ax = nullptr;
    if (d_pm_Ay) cudaFree(d_pm_Ay);
    d_pm_Ay = nullptr;
    if (d_planes) cudaFree(d_planes);
    d_planes = nullptr;
    num_planes=0;
}

void PlaneData::load(const std::vector<std::vector<double>>& lx,
                 const std::vector<std::vector<double>>& ly,
                 const std::vector<std::vector<double>>& lM,
                 const std::vector<std::vector<int>>& bucket_start,
                 const std::vector<std::vector<int>>& bucket_count,
                 const std::vector<std::vector<double>>& p3m_params,
                 const std::vector<double>& zs,
                 const std::vector<double>& factors) {

    if (lx.size()!=zs.size() || ly.size()!=zs.size() || lM.size()!=zs.size() ||
        bucket_start.size()!=zs.size() || bucket_count.size()!=zs.size() ||
        p3m_params.size()!=zs.size() || factors.size()!=zs.size())
        throw std::invalid_argument("Inconsistent lens-plane input lengths");
    for (size_t i=0; i<zs.size(); ++i) {
        if (lx[i].size()!=ly[i].size() || lx[i].size()!=lM[i].size() ||
            bucket_start[i].size()!=bucket_count[i].size() || p3m_params[i].size()!=11)
            throw std::invalid_argument("Invalid packed P3M plane");
    }
    clear();
    if (zs.empty()) return;  // Empty systems still produce the unlensed map.
    std::vector<Lens> flat_lenses;
    std::vector<Bucket> flat_buckets;
    std::vector<PlaneInfo> h_planes;

    size_t total_lenses = 0;
    size_t total_buckets = 0;
    for (size_t i = 0; i < zs.size(); ++i) {
        total_lenses += lx[i].size();
        total_buckets += bucket_start[i].size();
    }
    flat_lenses.reserve(total_lenses);
    flat_buckets.reserve(total_buckets);
    h_planes.reserve(zs.size());

    size_t total_pm_nodes = 0;

    for (size_t i = 0; i < zs.size(); ++i) {
        PlaneInfo p{};
        p.use_p3m = true;
        p.z = zs[i];
        p.factor = factors[i];
        p.lens_start_idx = (int)flat_lenses.size();
        p.lens_count = (int)lx[i].size();
        p.bucket_start_idx = (int)flat_buckets.size();

        // Positional layout shared with physics.p3m._pack_p3m_params:
        // [pm_nx, pm_ny, pm_xmin, pm_ymin, H, a_pm,
        //  pp_nx, pp_ny, pp_xmin, pp_ymin, pp_h]
        p.pm_nx  = (int)p3m_params[i][0];
        p.pm_ny  = (int)p3m_params[i][1];
        p.pm_xmin = p3m_params[i][2];
        p.pm_ymin = p3m_params[i][3];
        p.H       = p3m_params[i][4];
        p.a_pm    = p3m_params[i][5];
        p.pp_nx   = (int)p3m_params[i][6];
        p.pp_ny   = (int)p3m_params[i][7];
        p.pp_xmin = p3m_params[i][8];
        p.pp_ymin = p3m_params[i][9];
        p.pp_h    = p3m_params[i][10];

        p.pm_offset = total_pm_nodes;
        total_pm_nodes += (size_t)p.pm_nx * (size_t)p.pm_ny;

        for (size_t j = 0; j < lx[i].size(); ++j) {
            flat_lenses.push_back({lx[i][j], ly[i][j], lM[i][j]});
        }
        for (size_t j = 0; j < bucket_start[i].size(); ++j) {
            flat_buckets.push_back({bucket_start[i][j], bucket_count[i][j]});
        }
        h_planes.push_back(p);
    }

    // Upload lenses

    CUDA_CHECK(cudaMalloc(&d_all_lenses, flat_lenses.size() * sizeof(Lens)));
    CUDA_CHECK(cudaMemcpy(d_all_lenses, flat_lenses.data(), flat_lenses.size() * sizeof(Lens), cudaMemcpyHostToDevice));

    // Upload buckets

    CUDA_CHECK(cudaMalloc(&d_all_buckets, flat_buckets.size() * sizeof(Bucket)));
    CUDA_CHECK(cudaMemcpy(d_all_buckets, flat_buckets.data(), flat_buckets.size() * sizeof(Bucket), cudaMemcpyHostToDevice));

    // Allocate global PM field arrays

    CUDA_CHECK(cudaMalloc(&d_pm_Ax, total_pm_nodes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pm_Ay, total_pm_nodes * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_pm_Ax, 0, total_pm_nodes * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_pm_Ay, 0, total_pm_nodes * sizeof(double)));

    // Upload plane info
    num_planes = (int)h_planes.size();

    CUDA_CHECK(cudaMalloc(&d_planes, h_planes.size() * sizeof(PlaneInfo)));
    CUDA_CHECK(cudaMemcpy(d_planes, h_planes.data(), h_planes.size() * sizeof(PlaneInfo), cudaMemcpyHostToDevice));

    // Reuse one scratch allocation across sequential plane preprocessing.
    size_t max_mesh_size = 0;
    size_t max_ncomplex = 0;
    for (const PlaneInfo& p : h_planes) {
        max_mesh_size = std::max(
            max_mesh_size, (size_t)p.pm_nx * (size_t)p.pm_ny
        );
        max_ncomplex = std::max(
            max_ncomplex, (size_t)p.pm_ny * (size_t)(p.pm_nx / 2 + 1)
        );
    }

    DeviceBuffer<double> rho;
    DeviceBuffer<cufftDoubleComplex> rho_k, Ax_k, Ay_k;
    rho.reserve(max_mesh_size);
    rho_k.reserve(max_ncomplex); Ax_k.reserve(max_ncomplex); Ay_k.reserve(max_ncomplex);
    auto d_rho = rho.get();
    auto d_rho_k = rho_k.get();
    auto d_Ax_k = Ax_k.get();
    auto d_Ay_k = Ay_k.get();

    // Plans are reusable because cufftExec receives the actual data pointers.
    struct PlanPair {
        cufftHandle first=0, second=0;
        PlanPair() = default;
        PlanPair(const PlanPair&) = delete;
        ~PlanPair() {
            if (first) cufftDestroy(first);
            if (second) cufftDestroy(second);
        }
    };
    std::map<std::pair<int, int>, PlanPair> plan_cache;
    const auto preprocessing_start = std::chrono::steady_clock::now();
    if (verbose) printf("[P3M] Preprocessing %zu lens plane(s)...\n", h_planes.size());

    for (size_t i = 0; i < h_planes.size(); ++i) {
        const PlaneInfo& p = h_planes[i];
        int pnx = p.pm_nx;
        int pny = p.pm_ny;
        size_t mesh_size = (size_t)pnx * pny;
        int nkx = pnx / 2 + 1;
        size_t ncomplex = (size_t)pny * nkx;

        CUDA_CHECK(cudaMemset(d_rho, 0, mesh_size * sizeof(double)));

        if (verbose) {
            printf("[P3M %zu/%zu] %d lenses | PM %dx%d | H=%.4e pc | "
                   "a_pm=%.4e pc\n",
                   i + 1, h_planes.size(), p.lens_count, pnx, pny,
                   p.H, p.a_pm);
        }

        // 1. TSC mass deposit
        if (p.lens_count > 0) {
            int threads = 256;
            int blocks = (p.lens_count + threads - 1) / threads;
            tsc_deposit_kernel<<<blocks, threads>>>(
                d_all_lenses + p.lens_start_idx, p.lens_count,
                d_rho, pnx, pny, p.pm_xmin, p.pm_ymin, p.H
            );
            CUDA_CHECK(cudaGetLastError());
        }

        const std::pair<int, int> dimensions(pnx, pny);
        auto plan_it = plan_cache.find(dimensions);
        if (plan_it == plan_cache.end()) {
            plan_it = plan_cache.try_emplace(dimensions).first;
            CUFFT_CHECK(cufftPlan2d(&plan_it->second.first, pny, pnx, CUFFT_D2Z));
            CUFFT_CHECK(cufftPlan2d(&plan_it->second.second, pny, pnx, CUFFT_Z2D));
        }

        CUFFT_CHECK(cufftExecD2Z(plan_it->second.first, d_rho, d_rho_k));

        // 3. Apply optimized Green function
        {
            int threads = 256;
            int blocks = ((int)ncomplex + threads - 1) / threads;
            apply_p3m_influence_kernel<<<blocks, threads>>>(
                d_rho_k, d_Ax_k, d_Ay_k, pnx, pny, p.H, p.a_pm
            );
            CUDA_CHECK(cudaGetLastError());
        }

        // 4. Inverse FFT: Z2D for Ax and Ay
        CUFFT_CHECK(cufftExecZ2D(
            plan_it->second.second, d_Ax_k, d_pm_Ax + p.pm_offset
        ));
        CUFFT_CHECK(cufftExecZ2D(
            plan_it->second.second, d_Ay_k, d_pm_Ay + p.pm_offset
        ));

        // 5. Normalize (cuFFT doesn't divide by N)
        double invN = 1.0 / ((double)pnx * pny);
        {
            int threads = 256;
            int blocks = ((int)mesh_size + threads - 1) / threads;
            normalize_pm_kernel<<<blocks, threads>>>(d_pm_Ax + p.pm_offset, mesh_size, invN);
            normalize_pm_kernel<<<blocks, threads>>>(d_pm_Ay + p.pm_offset, mesh_size, invN);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // The final wait is required before scratch buffers and plans are freed.
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - preprocessing_start
    ).count();
    if (verbose) printf("[P3M] Done in %.3f s\n", elapsed);
}


// Pure PP uploads only lenses and plane geometry. clear() also releases any
// previous frame's PM fields and buckets when switching methods on an engine.
void PlaneData::load_pp(const std::vector<std::vector<double>>& lx,
    const std::vector<std::vector<double>>& ly,
    const std::vector<std::vector<double>>& mass,
    const std::vector<double>& zs, const std::vector<double>& factors) {
    if (lx.size()!=zs.size() || ly.size()!=zs.size() || mass.size()!=zs.size() ||
        factors.size()!=zs.size())
        throw std::invalid_argument("Inconsistent PP lens-plane input lengths");
    for (size_t i=0; i<zs.size(); ++i) {
        if (lx[i].size()!=ly[i].size() || lx[i].size()!=mass[i].size())
            throw std::invalid_argument("Invalid PP lens coordinates/masses");
    }
    clear();
    if (zs.empty()) return;
    std::vector<Lens> lenses;
    std::vector<PlaneInfo> host_planes;
    host_planes.reserve(zs.size());
    for (size_t i=0; i<zs.size(); ++i) {
        PlaneInfo plane{};
        plane.use_p3m = false;
        plane.z = zs[i];
        plane.factor = factors[i];
        plane.lens_start_idx = static_cast<int>(lenses.size());
        plane.lens_count = static_cast<int>(lx[i].size());
        for (size_t j=0; j<lx[i].size(); ++j)
            lenses.push_back({lx[i][j], ly[i][j], mass[i][j]});
        host_planes.push_back(plane);
    }
    if (!lenses.empty()) {
        CUDA_CHECK(cudaMalloc(&d_all_lenses, lenses.size()*sizeof(Lens)));
        CUDA_CHECK(cudaMemcpy(d_all_lenses, lenses.data(), lenses.size()*sizeof(Lens),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&d_planes, host_planes.size()*sizeof(PlaneInfo)));
    CUDA_CHECK(cudaMemcpy(d_planes, host_planes.data(), host_planes.size()*sizeof(PlaneInfo),
                          cudaMemcpyHostToDevice));
    num_planes = static_cast<int>(host_planes.size());
    if (verbose) printf("[PP] Uploaded %zu lenses in %zu planes (no PM/FFT/buckets)\n",
                        lenses.size(), host_planes.size());
}
