#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.cuh"
#include "lightcurve_extractor.cuh"

PYBIND11_MODULE(_cuda, m) {
    m.doc() = "DCLens CUDA numerical backend";
    py::register_exception<CudaMemoryError>(m, "CudaMemoryError", PyExc_MemoryError);
    m.def("memory_info", &memory_info, "Return current free and total CUDA bytes");
    py::class_<Engine>(m, "Engine")
        .def(py::init<size_t, bool>(), py::arg("max_active_cells")=2000000,
             py::arg("verbose")=false)
        .def("load_planes", &Engine::load_planes)
        .def("load_pp_planes", &Engine::load_pp_planes)
        .def("estimate_buffer_size", &Engine::estimate_buffer_size)
        .def("simulate_chunk", &Engine::simulate_chunk)
        .def("close", &Engine::close);
    py::class_<LightCurveWorkspace>(m, "LightCurveWorkspace")
        .def(py::init<const SourceKernel&>())
        .def("extract", &LightCurveWorkspace::extract)
        .def("clear_batches", &LightCurveWorkspace::clear_batches)
        .def("close", &LightCurveWorkspace::close);
}
