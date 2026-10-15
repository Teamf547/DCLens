#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <limits>

class CudaMemoryError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
inline void check_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    std::string message = std::string(operation) + ": " + cudaGetErrorString(error);
    if (error == cudaErrorMemoryAllocation) {
        // A failed allocation is recoverable by reducing a light-curve batch.
        // Clear the runtime error latch before the subsequent launch check.
        cudaGetLastError();
        throw CudaMemoryError(message);
    }
    throw std::runtime_error(message);
}
#define CUDA_CHECK(call) check_cuda((call), #call)
#define CUFFT_CHECK(call) do { auto status = (call); if (status != CUFFT_SUCCESS) \
    throw std::runtime_error(std::string(#call) + ": cuFFT error " + std::to_string(status)); } while (0)

// CUDA allocations have a single owner. On growth release the old allocation
// before reserving the new capacity, so transient memory cannot double.
template<class T> class DeviceBuffer {
    T* pointer_ = nullptr;
    size_t capacity_ = 0;
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { clear(); }
    void clear() noexcept {
        if (pointer_) cudaFree(pointer_);
        pointer_ = nullptr;
        capacity_ = 0;
    }
    void reserve(size_t count) {
        if (count <= capacity_) return;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::overflow_error("CUDA allocation size overflow");
        clear();
        CUDA_CHECK(cudaMalloc(&pointer_, count * sizeof(T)));
        capacity_ = count;
    }
    T* get() const { return pointer_; }
    size_t capacity() const { return capacity_; }
    size_t bytes() const { return capacity_*sizeof(T); }
};
