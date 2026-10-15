#pragma once
#include "rasterizer.cuh"
#include <pybind11/numpy.h>

namespace py = pybind11;

class Engine {
    PlaneData planes;
    IPMWorkspace ipm;
    Rasterizer rasterizer;
    bool closed=false;
    void require_open() const;
public:
    explicit Engine(size_t max_active_cells=2000000, bool verbose=false);
    void close() noexcept;
    void load_pp_planes(const std::vector<std::vector<double>>& lx,
        const std::vector<std::vector<double>>& ly,
        const std::vector<std::vector<double>>& mass,
        const std::vector<double>& zs, const std::vector<double>& factors);
    void load_planes(const std::vector<std::vector<double>>& lx,
        const std::vector<std::vector<double>>& ly,
        const std::vector<std::vector<double>>& mass,
        const std::vector<std::vector<int>>& bucket_start,
        const std::vector<std::vector<int>>& bucket_count,
        const std::vector<std::vector<double>>& parameters,
        const std::vector<double>& zs, const std::vector<double>& factors);
    double estimate_buffer_size(double cx, double cy, double width, double height,
        double xmin, double xmax, double ymin, double ymax, double step, int steps);
    py::array_t<float> simulate_chunk(
        double img_xmin, double img_xmax, double img_ymin, double img_ymax,
        double coarse_size, double min_size, double distortion_threshold, int max_iters,
        double c_xmin, double c_xmax, double c_ymin, double c_ymax,
        int nx_src, int ny_src, int ss);
};
