// One CUDA block per source position; all points of a batch share one launch.
#include "lightcurve_extractor.cuh"
#include <limits>

__global__ void convolve_path(const float* map, int height, int width,
    const int* rows, const int* cols, const double* kernel, int radius,
    int kernel_width, double* output) {
    int sample = blockIdx.x;
    int row=rows[sample], col=cols[sample];
    // A source window crossing the full map boundary has unit magnification.
    // Python also excludes these samples before calculating a tight path ROI.
    if (row<radius || col<radius || row>=height-radius || col>=width-radius) {
        if (threadIdx.x==0) output[sample]=1.0;
        return;
    }
    double sum=0;
    size_t count=size_t(kernel_width)*kernel_width;
    for (size_t k=threadIdx.x; k<count; k+=blockDim.x) {
        int dy=int(k/kernel_width)-radius;
        int dx=int(k%kernel_width)-radius;
        sum += double(map[size_t(row+dy)*width+col+dx])*kernel[k];
    }
    __shared__ double partial[256];
    partial[threadIdx.x]=sum;
    __syncthreads();
    for (int stride=128; stride>0; stride/=2) {
        if (threadIdx.x<stride) partial[threadIdx.x]+=partial[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) output[sample]=partial[0];
}

std::pair<size_t, size_t> memory_info() {
    size_t free=0, total=0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return {free, total};
}
LightCurveWorkspace::LightCurveWorkspace(const SourceKernel& weights) {
    if (weights.ndim()!=2 || weights.shape(0)!=weights.shape(1) ||
        weights.shape(0)<1 || weights.shape(0)%2!=1 ||
        weights.shape(0)>std::numeric_limits<int>::max())
        throw std::invalid_argument("Source kernel must be square with odd width");
    kernel_width=int(weights.shape(0)); radius=kernel_width/2;
    kernel.reserve(weights.size());
    CUDA_CHECK(cudaMemcpy(kernel.get(), weights.data(), weights.nbytes(), cudaMemcpyHostToDevice));
}
void LightCurveWorkspace::clear_batches() noexcept {
    map.clear(); rows.clear(); cols.clear(); output.clear();
}
void LightCurveWorkspace::close() noexcept {
    clear_batches(); kernel.clear(); closed=true;
}
py::array_t<double> LightCurveWorkspace::extract(const FloatMap& roi,
    const IntSamples& sample_rows, const IntSamples& sample_cols, size_t budget_bytes) {
    if (closed) throw std::runtime_error("Light-curve workspace is closed");
    if (roi.ndim()!=2 || sample_rows.ndim()!=1 || sample_cols.ndim()!=1 ||
        sample_rows.size()!=sample_cols.size() || roi.shape(0)<1 || roi.shape(1)<1 ||
        roi.shape(0)>std::numeric_limits<int>::max() || roi.shape(1)>std::numeric_limits<int>::max() ||
        sample_rows.size()>std::numeric_limits<int>::max())
        throw std::invalid_argument("Invalid map ROI or sample coordinate arrays");
    size_t n=sample_rows.size(), pixels=roi.size();
    size_t required=pixels*sizeof(float)+n*(2*sizeof(int)+sizeof(double))+kernel.bytes();
    if (required>budget_bytes) throw CudaMemoryError("Light-curve batch exceeds GPU memory budget");
    // Independent high-water marks may exceed the budget even when the new
    // batch fits. Release only the batch buffers, retaining the source kernel.
    size_t retained=std::max(map.capacity(),pixels)*sizeof(float)
        + (std::max(rows.capacity(),n)+std::max(cols.capacity(),n))*sizeof(int)
        + std::max(output.capacity(),n)*sizeof(double)+kernel.bytes();
    if (retained>budget_bytes) clear_batches();
    py::array_t<double> result(n);
    if (!n) return result;
    map.reserve(pixels); rows.reserve(n); cols.reserve(n); output.reserve(n);
    CUDA_CHECK(cudaMemcpy(map.get(), roi.data(), roi.nbytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rows.get(), sample_rows.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cols.get(), sample_cols.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    convolve_path<<<unsigned(n),256>>>(map.get(), int(roi.shape(0)), int(roi.shape(1)),
        rows.get(), cols.get(), kernel.get(), radius, kernel_width, output.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(result.mutable_data(), output.get(), n*sizeof(double), cudaMemcpyDeviceToHost));
    return result;
}
