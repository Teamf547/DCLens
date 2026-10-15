// Multi-plane rays and adaptive inverse polygon mapping in arcseconds.
#include "ipm.cuh"
#include <algorithm>
#include <vector>
#include "p3m_deflector.cuh"

// Each block evaluates the eight control points of one candidate buffer frame.
__global__ void check_buffer_frames_kernel(
    const Lens* __restrict__ all_lenses,
    const Bucket* __restrict__ all_buckets,
    const double* __restrict__ pm_Ax,
    const double* __restrict__ pm_Ay,
    const PlaneInfo* __restrict__ planes,
    int num_planes,
    double img_cx, double img_cy,
    double base_w, double base_h,
    double buffer_step,
    double src_xmin, double src_xmax,
    double src_ymin, double src_ymax,
    int* __restrict__ frame_valid
) {
    int frame_idx = blockIdx.x; 
    int pt_idx = threadIdx.x;

    if (pt_idx >= 8) return;

    double current_buffer = (frame_idx + 1) * buffer_step;
    double frame_w = base_w + 2.0 * current_buffer;
    double frame_h = base_h + 2.0 * current_buffer;

    double x_min = img_cx - frame_w * 0.5;
    double x_max = img_cx + frame_w * 0.5;
    double y_min = img_cy - frame_h * 0.5;
    double y_max = img_cy + frame_h * 0.5;
    double x_mid = img_cx;
    double y_mid = img_cy;

    double px = 0.0, py = 0.0;
    if (pt_idx == 0) { px = x_min; py = y_min; }
    else if (pt_idx == 1) { px = x_max; py = y_min; }
    else if (pt_idx == 2) { px = x_max; py = y_max; }
    else if (pt_idx == 3) { px = x_min; py = y_max; }
    else if (pt_idx == 4) { px = x_mid; py = y_min; }
    else if (pt_idx == 5) { px = x_max; py = y_mid; }
    else if (pt_idx == 6) { px = x_mid; py = y_max; }
    else if (pt_idx == 7) { px = x_min; py = y_mid; }

    Point src_pt = ray_shoot_device(
        px, py, all_lenses, all_buckets, pm_Ax, pm_Ay, planes, num_planes
    );

    // 判断是否严格落在了目标 Source Region 之外
    bool is_outside = (src_pt.x < src_xmin || src_pt.x > src_xmax ||
                       src_pt.y < src_ymin || src_pt.y > src_ymax);

    // 统计当前 frame 有几个点在外面
    __shared__ int outside_count;
    if (threadIdx.x == 0) outside_count = 0;
    __syncthreads();

    if (is_outside) {
        atomicAdd(&outside_count, 1);
    }
    __syncthreads();

    // 如果 8 个点全部都在外面，标记该 frame 为有效(1)
    if (threadIdx.x == 0) {
        frame_valid[frame_idx] = (outside_count == 8) ? 1 : 0;
    }
}

#include "asr.cuh"

void IPMWorkspace::ensure_ipm_buffers() {
    // 48+48+80 bytes per slot: ~336 MiB at 2 million.
    // Python SimulationConfig defaults to 20 million slots (~3.28 GiB).
    queue_a.reserve(capacity); queue_b.reserve(capacity);
    results.reserve(capacity); counters.reserve(2);
}

int refine_ipm(IPMWorkspace& work, const PlaneData& planes,
    double img_xmin, double img_xmax, double img_ymin, double img_ymax,
    double coarse_size, double min_size, double distortion_threshold, int max_iters,
    double c_xmin, double c_xmax, double c_ymin, double c_ymax) {
    work.ensure_ipm_buffers();
    auto d_queue_A = work.queue_a.get(); auto d_queue_B = work.queue_b.get();
    auto d_results = work.results.get(); auto d_counters = work.counters.get();
    const auto max_queue_size = work.capacity, max_results = work.capacity;
    const auto d_all_lenses = planes.d_all_lenses;
    const auto d_all_buckets = planes.d_all_buckets;
    const auto d_pm_Ax = planes.d_pm_Ax;
    const auto d_pm_Ay = planes.d_pm_Ay;
    const auto d_planes = planes.d_planes;
    const auto num_planes = planes.num_planes;
    // Coarse cells retain the established full-square extent at chunk edges.
    std::vector<Pixel> init_pixels;
    int nx = (int)std::ceil((img_xmax - img_xmin) / coarse_size - 1e-7);
    int ny = (int)std::ceil((img_ymax - img_ymin) / coarse_size - 1e-7);
    if (size_t(nx) * size_t(ny) > max_queue_size)
        throw std::runtime_error("IPM coarse queue capacity exceeded; reduce max_coarse_per_chunk or increase max_active_cells");
    init_pixels.reserve((size_t)nx * (size_t)ny);
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            double x0 = img_xmin + ix * coarse_size;
            double y0 = img_ymin + iy * coarse_size;
            init_pixels.push_back({x0, x0 + coarse_size, y0, y0 + coarse_size, coarse_size, 0});
        }
    }

    int current_count = (int)init_pixels.size();
    CUDA_CHECK(cudaMemcpy(d_queue_A, init_pixels.data(), current_count * sizeof(Pixel), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_counters + 1, 0, sizeof(int)));

    Pixel* in_ptr = d_queue_A;
    Pixel* out_ptr = d_queue_B;

    for (int iter = 0; iter < max_iters; ++iter) {
        if (current_count == 0) break;
        CUDA_CHECK(cudaMemset(d_counters, 0, sizeof(int)));

        int threads = 256;
        int blocks = (current_count + threads - 1) / threads;

        refine_kernel<<<blocks, threads>>>(
            in_ptr, out_ptr, d_results,
            d_all_lenses, d_all_buckets, d_pm_Ax, d_pm_Ay,
            d_planes, num_planes,
            d_counters, d_counters + 1, current_count, min_size, distortion_threshold,
            c_xmin, c_xmax, c_ymin, c_ymax, (int)max_results, (int)max_queue_size
        );
        CUDA_CHECK(cudaGetLastError());

        int h_counts[2];
        CUDA_CHECK(cudaMemcpy(h_counts, d_counters, 2 * sizeof(int), cudaMemcpyDeviceToHost));
        int next_gen_count = h_counts[0];

        if (next_gen_count > (int)max_queue_size || h_counts[1] > (int)max_results)
            throw std::runtime_error("IPM queue capacity exceeded; reduce max_coarse_per_chunk or increase max_active_cells");
        if (next_gen_count == 0) break;
        if (iter + 1 == max_iters)
            throw std::runtime_error("IPM refinement unfinished; increase max_refine_iters");

        std::swap(in_ptr, out_ptr);
        current_count = next_gen_count;
    }

    int total_results = 0;
    CUDA_CHECK(cudaMemcpy(&total_results, d_counters + 1, sizeof(int), cudaMemcpyDeviceToHost));
    return total_results;

}

double probe_buffer(const PlaneData& planes,
    double img_cx, double img_cy,
    double base_w, double base_h,
    double src_xmin, double src_xmax,
    double src_ymin, double src_ymax,
    double buffer_step, int max_steps
) {
    const auto d_all_lenses = planes.d_all_lenses;
    const auto d_all_buckets = planes.d_all_buckets;
    const auto d_pm_Ax = planes.d_pm_Ax;
    const auto d_pm_Ay = planes.d_pm_Ay;
    const auto d_planes = planes.d_planes;
    const auto num_planes = planes.num_planes;
    DeviceBuffer<int> valid;
    valid.reserve(max_steps);
    auto d_frame_valid = valid.get();
    CUDA_CHECK(cudaMemset(d_frame_valid, 0, max_steps * sizeof(int)));

    // 启动 kernel：按步长发射多个 frame，每个 frame 8 个线程（测试8个点）
    check_buffer_frames_kernel<<<max_steps, 8>>>(
        d_all_lenses, d_all_buckets, d_pm_Ax, d_pm_Ay,
        d_planes, num_planes,
        img_cx, img_cy, base_w, base_h, buffer_step,
        src_xmin, src_xmax, src_ymin, src_ymax,
        d_frame_valid
    );
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> h_frame_valid(max_steps);
    CUDA_CHECK(cudaMemcpy(h_frame_valid.data(), d_frame_valid, max_steps * sizeof(int), cudaMemcpyDeviceToHost));

    // 从内向外寻找第一个让 8 个 control points 全部出界的 frame
    for (int i = 0; i < max_steps; ++i) {
        if (h_frame_valid[i] == 1) {
            return (i + 1) * buffer_step;
        }
    }

    // 如果全部搜完都没有，返回最大容许值，让 Python 端去打 Warning
    return max_steps * buffer_step;
}
