#pragma once
#include "cuda_memory.cuh"
#include <pybind11/numpy.h>

namespace py = pybind11;
using FloatMap = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntSamples = py::array_t<int, py::array::c_style | py::array::forcecast>;
using SourceKernel = py::array_t<double, py::array::c_style | py::array::forcecast>;

class LightCurveWorkspace {
    DeviceBuffer<float> map;
    DeviceBuffer<int> rows, cols;
    DeviceBuffer<double> kernel, output;
    int radius=0, kernel_width=0;
    bool closed=false;
public:
    explicit LightCurveWorkspace(const SourceKernel& weights);
    py::array_t<double> extract(const FloatMap& roi, const IntSamples& sample_rows,
                               const IntSamples& sample_cols, size_t budget_bytes);
    void clear_batches() noexcept;
    void close() noexcept;
};
std::pair<size_t, size_t> memory_info();
