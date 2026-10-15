#pragma once
// ASR: map cell corners, decide refinement, enqueue children or emit leaves.
// IPM owns the queues and launches this unchanged kernel in ipm.cu.
#include "p3m_deflector.cuh"

// --- Kernel ---
__global__ void refine_kernel(
    const Pixel* __restrict__ in_queue,
    Pixel* __restrict__ out_queue,
    ResultLeaf* __restrict__ results,
    const Lens* __restrict__ all_lenses,
    const Bucket* __restrict__ all_buckets,
    const double* __restrict__ pm_Ax,
    const double* __restrict__ pm_Ay,
    const PlaneInfo* __restrict__ planes, 
    int num_planes,                       
    int* __restrict__ out_count,
    int* __restrict__ result_count,
    int num_in,
    double min_pixel_size,
    double distortion_threshold,
    double c_xmin, double c_xmax, double c_ymin, double c_ymax,
    int max_results, int max_queue
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_in) return;

    Pixel p = in_queue[idx];

    Point m0 = ray_shoot_device(p.x0, p.y1, all_lenses, all_buckets, pm_Ax, pm_Ay, planes, num_planes); // TL
    Point m1 = ray_shoot_device(p.x1, p.y1, all_lenses, all_buckets, pm_Ax, pm_Ay, planes, num_planes); // TR
    Point m2 = ray_shoot_device(p.x1, p.y0, all_lenses, all_buckets, pm_Ax, pm_Ay, planes, num_planes); // BR
    Point m3 = ray_shoot_device(p.x0, p.y0, all_lenses, all_buckets, pm_Ax, pm_Ay, planes, num_planes); // BL

    auto cross_prod = [](Point a, Point b, Point c) {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    };

    double s1 = cross_prod(m0, m1, m2);
    double s2 = cross_prod(m0, m2, m3);

    double total_signed_area = (s1 + s2) * 0.5;
    double src_area = abs(total_signed_area);
    double img_area = p.pix_size * p.pix_size;
    
    bool is_self_intersecting = (s1 * s2 < 0.0);
    bool is_valid = (src_area > 1e-20) && (!is_self_intersecting);
    
    // Relative triangle-area imbalance is the established refinement metric.
    double abs_s1 = abs(s1);
    double abs_s2 = abs(s2);
    double distortion = abs(abs_s1 - abs_s2) / (abs_s1 + abs_s2 + 1e-20);

    double mag = -1.0;
    if (is_valid) {
        mag = img_area / src_area;
    }

    bool split = false;

    if (is_self_intersecting || !is_valid) {
        split = true; 
    } else {

        if (distortion > distortion_threshold) {
            split = true;
        }
    }

    if (p.pix_size <= min_pixel_size) split = false;

    if (split) {
        int out_idx = atomicAdd(out_count, 4);
        if (out_idx > max_queue - 4) return; // Guard BEFORE writing children.
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
        if (res_idx >= max_results) return;
        results[res_idx] = {m0, m1, m2, m3, is_valid ? mag : -1.0, p.pix_size};
    }
}

