#include "engine.cuh"
#include <limits>

Engine::Engine(size_t capacity, bool verbose)
    : planes(verbose), ipm(capacity) {
    if (!capacity || capacity > size_t(std::numeric_limits<int>::max()/4))
        throw std::invalid_argument("max_active_cells exceeds CUDA counter range");
    // The probe only needs lens planes. EGL and huge IPM queues are lazy.
    CUDA_CHECK(cudaFree(nullptr));
}
void Engine::require_open() const {
    if (closed) throw std::runtime_error("Engine is closed");
}
void Engine::close() noexcept {
    rasterizer.close(); ipm.clear(); planes.clear(); closed=true;
}
void Engine::load_pp_planes(const std::vector<std::vector<double>>& lx,
    const std::vector<std::vector<double>>& ly,
    const std::vector<std::vector<double>>& mass,
    const std::vector<double>& zs, const std::vector<double>& factors) {
    require_open();
    planes.load_pp(lx, ly, mass, zs, factors);
}
void Engine::load_planes(const std::vector<std::vector<double>>& lx,
    const std::vector<std::vector<double>>& ly,
    const std::vector<std::vector<double>>& mass,
    const std::vector<std::vector<int>>& bucket_start,
    const std::vector<std::vector<int>>& bucket_count,
    const std::vector<std::vector<double>>& parameters,
    const std::vector<double>& zs, const std::vector<double>& factors) {
    require_open();
    planes.load(lx, ly, mass, bucket_start, bucket_count, parameters, zs, factors);
}
double Engine::estimate_buffer_size(double cx, double cy, double width, double height,
    double xmin, double xmax, double ymin, double ymax, double step, int steps) {
    require_open();
    if (!(step>0) || steps<1) throw std::invalid_argument("Invalid buffer probe range");
    return probe_buffer(planes, cx, cy, width, height, xmin, xmax, ymin, ymax, step, steps);
}
py::array_t<float> Engine::simulate_chunk(
    double img_xmin, double img_xmax, double img_ymin, double img_ymax,
    double coarse_size, double min_size, double distortion_threshold, int max_iters,
    double c_xmin, double c_xmax, double c_ymin, double c_ymax,
    int nx_src, int ny_src, int ss) {
    require_open();
    if (!(img_xmax>img_xmin && img_ymax>img_ymin && c_xmax>c_xmin && c_ymax>c_ymin
            && coarse_size>0 && min_size>0) || max_iters<1 || nx_src<1 || ny_src<1 || ss<1)
        throw std::invalid_argument("Invalid IPM/source geometry");
    int leaves = refine_ipm(ipm, planes, img_xmin, img_xmax, img_ymin, img_ymax,
        coarse_size, min_size, distortion_threshold, max_iters,
        c_xmin, c_xmax, c_ymin, c_ymax);
    py::array_t<float> output({ny_src, nx_src});
    rasterizer.render(ipm.results.get(), leaves, c_xmin, c_xmax, c_ymin, c_ymax,
                      nx_src, ny_src, ss, output.mutable_data());
    return output;
}
