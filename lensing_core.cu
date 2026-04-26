/* lensing_core.cu */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

// define namespace and structs

namespace py = pybind11;

// test cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error("CUDA Error"); \
        } \
    } while (0)

struct Point { double x, y; };

struct Lens {
    double x, y, theta_E_sq; // square of einstein ring radius
};

struct PlaneInfo {
    double z;      
    double factor; //projection factor
    int lens_start_idx;
    int lens_count;
};

struct Pixel {
    double x0, x1, y0, y1;
    double pix_size;
    int level;
};

struct ResultLeaf {
    Point p0, p1, p2, p3; 
    double magnification;
    double pix_size;
};

// --- Device Functions ---

// ray_shotting
__device__ Point ray_shoot_device(double theta_x, double theta_y, 
                                 const Lens* all_lenses, 
                                 const PlaneInfo* planes, 
                                 int num_planes) {
    double beta_x = theta_x;
    double beta_y = theta_y;
    double TINY = 1e-12;

    for (int i = 0; i < num_planes; ++i) {

        const PlaneInfo& p = planes[i];
        double ax_sum = 0.0; // sum of deflection angle in x projection
        double ay_sum = 0.0; // the same for y projection
        
        for (int j = 0; j < p.lens_count; ++j) {
            const Lens& l = all_lenses[p.lens_start_idx + j];
            double dx = beta_x - l.x;
            double dy = beta_y - l.y;
            double r2 = dx*dx + dy*dy; // distace square between ray and lens
            if (r2 < TINY) r2 = TINY;
            double f = l.theta_E_sq / r2; // deflection angle alpha
            ax_sum += f * dx;
            ay_sum += f * dy;
        }
        beta_x -= p.factor * ax_sum; // positon of deflected ray in x projection
        beta_y -= p.factor * ay_sum; // the same for y projection
    }
    return {beta_x, beta_y};
}

// // tie formula calculate signal area of pixel 
// __device__ double signed_area(Point p0, Point p1, Point p2, Point p3) {
//     double area = 0.0;
//     area += (p0.x * p1.y - p1.x * p0.y);
//     area += (p1.x * p2.y - p2.x * p1.y);
//     area += (p2.x * p3.y - p3.x * p2.y);
//     area += (p3.x * p0.y - p0.x * p3.y);
//     return area * 0.5;
// }

// --- Kernel ---
__global__ void refine_kernel(
    const Pixel* __restrict__ in_queue,
    Pixel* __restrict__ out_queue,
    ResultLeaf* __restrict__ results,
    const Lens* __restrict__ all_lenses,
    const PlaneInfo* __restrict__ planes, 
    int num_planes,                       
    int* __restrict__ out_count,
    int* __restrict__ result_count,
    int num_in,
    double min_pixel_size,
    double mu_upper, double mu_lower,
    double c_xmin, double c_xmax, double c_ymin, double c_ymax
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_in) return;

    Pixel p = in_queue[idx];

    // step1: ray shotting 4 corners of the pixel
    Point m0 = ray_shoot_device(p.x0, p.y1, all_lenses, planes, num_planes); // TL
    Point m1 = ray_shoot_device(p.x1, p.y1, all_lenses, planes, num_planes); // TR
    Point m2 = ray_shoot_device(p.x1, p.y0, all_lenses, planes, num_planes); // BR
    Point m3 = ray_shoot_device(p.x0, p.y0, all_lenses, planes, num_planes); // BL

    // step2:  calculate the area and magnification
    auto cross_prod = [](Point a, Point b, Point c) {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    };

    //// step2.1 :devide the quad into two triangle
    double s1 = cross_prod(m0, m1, m2);
    double s2 = cross_prod(m0, m2, m3);

    //// step2.2: calculate the signed area
    double total_signed_area = (s1 + s2) * 0.5;
    double src_area = abs(total_signed_area);
    double img_area = p.pix_size * p.pix_size;
    
    double mag = -1.0;
    bool is_valid = (src_area > 1e-14);
    if (is_valid) {
        mag = img_area / src_area;
    }

    // step3: refine or not
    bool split = false;
    
    //// step3.1: detect whether ray elements area is self-intersection or high-magnificated
    bool is_self_intersecting = (s1 * s2 < 0.0);

    if (is_self_intersecting || !is_valid) {
        split = true; 
    } else {
        //// step3.2: compare the magnification with thresholds
        if (mag > mu_upper || mag < mu_lower) {
            split = true;
        }
    }
    
    //// step3.3 break or not
    if (p.pix_size <= min_pixel_size) split = false;

    // step4: refine or store result
    if (split) {
        int out_idx = atomicAdd(out_count, 4); //?
        double xm = 0.5 * (p.x0 + p.x1);
        double ym = 0.5 * (p.y0 + p.y1);
        double new_sz = p.pix_size * 0.5;
        int new_lvl = p.level + 1;

        out_queue[out_idx+0] = {p.x0, xm, ym, p.y1, new_sz, new_lvl};
        out_queue[out_idx+1] = {xm, p.x1, ym, p.y1, new_sz, new_lvl};
        out_queue[out_idx+2] = {xm, p.x1, p.y0, ym, new_sz, new_lvl};
        out_queue[out_idx+3] = {p.x0, xm, p.y0, ym, new_sz, new_lvl};
    } else {
        int res_idx = atomicAdd(result_count, 1);
        results[res_idx] = {m0, m1, m2, m3, is_valid ? mag : -1.0, p.pix_size};
    }
}

// --- Host Class ---
class LensingEngineGPU {
    Lens* d_all_lenses = nullptr;
    PlaneInfo* d_planes = nullptr;
    int num_planes = 0;
    
    Pixel* d_queue_A = nullptr;
    Pixel* d_queue_B = nullptr;
    ResultLeaf* d_results = nullptr;
    int* d_counters = nullptr; 
    
    size_t max_queue_size;
    size_t max_results;

public:
    LensingEngineGPU(double D_S, size_t queue_cap, size_t result_cap) 
        : max_queue_size(queue_cap), max_results(result_cap) 
    {
        CUDA_CHECK(cudaMalloc(&d_queue_A, max_queue_size * sizeof(Pixel)));//?
        CUDA_CHECK(cudaMalloc(&d_queue_B, max_queue_size * sizeof(Pixel)));
        CUDA_CHECK(cudaMalloc(&d_results, max_results * sizeof(ResultLeaf)));
        CUDA_CHECK(cudaMalloc(&d_counters, 2 * sizeof(int)));
    }

    ~LensingEngineGPU() {
        if (d_all_lenses) cudaFree(d_all_lenses);
        if (d_planes) cudaFree(d_planes);
        if (d_queue_A) cudaFree(d_queue_A);
        if (d_queue_B) cudaFree(d_queue_B);
        if (d_results) cudaFree(d_results);
        if (d_counters) cudaFree(d_counters);
    }

    void load_planes(const std::vector<std::vector<double>>& lx,
                     const std::vector<std::vector<double>>& ly,
                     const std::vector<std::vector<double>>& lE,
                     const std::vector<double>& zs,
                     const std::vector<double>& factors) {
        
        std::vector<Lens> flat_lenses;
        std::vector<PlaneInfo> h_planes;

        for (size_t i = 0; i < zs.size(); ++i) {
            PlaneInfo p;
            p.z = zs[i];
            p.factor = factors[i];
            p.lens_start_idx = (int)flat_lenses.size();
            p.lens_count = (int)lx[i].size();
            
            for (size_t j = 0; j < lx[i].size(); ++j) {
                flat_lenses.push_back({lx[i][j], ly[i][j], lE[i][j] * lE[i][j]});
            }
            h_planes.push_back(p);
        }

        // load lenses
        if (d_all_lenses) cudaFree(d_all_lenses);
        CUDA_CHECK(cudaMalloc(&d_all_lenses, flat_lenses.size() * sizeof(Lens)));
        CUDA_CHECK(cudaMemcpy(d_all_lenses, flat_lenses.data(), flat_lenses.size() * sizeof(Lens), cudaMemcpyHostToDevice));//?

        // load planes
        num_planes = (int)h_planes.size();
        if (d_planes) cudaFree(d_planes);
        CUDA_CHECK(cudaMalloc(&d_planes, h_planes.size() * sizeof(PlaneInfo)));
        CUDA_CHECK(cudaMemcpy(d_planes, h_planes.data(), h_planes.size() * sizeof(PlaneInfo), cudaMemcpyHostToDevice));
    }

    // --- main process: adaptive quadtree refine ---
    py::dict run_adaptive(
        double img_xmin, double img_xmax, double img_ymin, double img_ymax,
        double coarse_size, double min_size, int max_iters,
        double mu_upper, double mu_lower,
        double c_xmin, double c_xmax, double c_ymin, double c_ymax
    ) {

        // step1: initialize the corase mesh
        std::vector<Pixel> init_pixels;
        int nx = (int)std::ceil((img_xmax - img_xmin) / coarse_size);
        int ny = (int)std::ceil((img_ymax - img_ymin) / coarse_size);
        for(int iy=0; iy<ny; ++iy) {
            for(int ix=0; ix<nx; ++ix) {
                double x0 = img_xmin + ix * coarse_size;
                double y0 = img_ymin + iy * coarse_size;
                init_pixels.push_back({x0, x0+coarse_size, y0, y0+coarse_size, coarse_size, 0});
            }
        }

        int current_count = (int)init_pixels.size();
        CUDA_CHECK(cudaMemcpy(d_queue_A, init_pixels.data(), current_count * sizeof(Pixel), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_counters + 1, 0, sizeof(int)));

        Pixel* in_ptr = d_queue_A;
        Pixel* out_ptr = d_queue_B;

        // step2: iteretive refine
        for (int iter = 0; iter < max_iters; ++iter) {
            if (current_count == 0) break;

            //// step2.1: clear the outqueue d_counters[0]
            CUDA_CHECK(cudaMemset(d_counters, 0, sizeof(int)));

            int threads = 256;
            int blocks = (current_count + threads - 1) / threads;

            //// step2.2: kernel refine
            refine_kernel<<<blocks, threads>>>( //?
                in_ptr, out_ptr, d_results, d_all_lenses, d_planes, num_planes,
                d_counters,
                d_counters + 1,
                current_count, min_size, mu_upper, mu_lower,
                c_xmin, c_xmax, c_ymin, c_ymax
            );
            CUDA_CHECK(cudaDeviceSynchronize());


            int h_counts[2];
            CUDA_CHECK(cudaMemcpy(h_counts, d_counters, 2 * sizeof(int), cudaMemcpyDeviceToHost));
            
            int next_gen_count = h_counts[0];
            int total_leaves_so_far = h_counts[1];

            printf("[Iter %2d] Active: %-8d | Total Leaves: %-8d\n", 
                iter + 1, next_gen_count / 4, total_leaves_so_far);
            fflush(stdout);

            if (next_gen_count == 0) break;
            if (next_gen_count > max_queue_size) next_gen_count = max_queue_size;

            std::swap(in_ptr, out_ptr);
            current_count = next_gen_count;
        }

        int total_results = 0;
        CUDA_CHECK(cudaMemcpy(&total_results, d_counters + 1, sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<ResultLeaf> host_results(total_results);
        CUDA_CHECK(cudaMemcpy(host_results.data(), d_results, total_results * sizeof(ResultLeaf), cudaMemcpyDeviceToHost));

        // transfer results to numpy arrays
        py::array_t<double> mapped_pts({(long)total_results, 4L, 2L});//?
        py::array_t<double> mags({(long)total_results});
        py::array_t<double> sizes({(long)total_results});
        
        auto r_pts = mapped_pts.mutable_unchecked<3>();
        auto r_mags = mags.mutable_unchecked<1>();
        auto r_sizes = sizes.mutable_unchecked<1>();

        for (int i = 0; i < total_results; ++i) {
            r_pts(i, 0, 0) = host_results[i].p0.x; r_pts(i, 0, 1) = host_results[i].p0.y;
            r_pts(i, 1, 0) = host_results[i].p1.x; r_pts(i, 1, 1) = host_results[i].p1.y;
            r_pts(i, 2, 0) = host_results[i].p2.x; r_pts(i, 2, 1) = host_results[i].p2.y;
            r_pts(i, 3, 0) = host_results[i].p3.x; r_pts(i, 3, 1) = host_results[i].p3.y;
            r_mags(i) = host_results[i].magnification;
            r_sizes(i) = host_results[i].pix_size;
        }

        py::dict ret;
        ret["mapped_pts"] = mapped_pts;
        ret["magnifications"] = mags;
        ret["pixel_sizes"] = sizes;
        return ret;
    }
};

PYBIND11_MODULE(lensing_core, m) {
    py::class_<LensingEngineGPU>(m, "LensingEngineGPU")
        .def(py::init<double, size_t, size_t>(), py::arg("D_S"), py::arg("queue_cap")=20000000, py::arg("res_cap")=20000000)
        .def("load_planes", &LensingEngineGPU::load_planes)
        .def("run_adaptive", &LensingEngineGPU::run_adaptive);
}