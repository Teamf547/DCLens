#pragma once
#include "p3m.cuh"

struct Point { double x, y; };
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


struct IPMWorkspace {
    DeviceBuffer<Pixel> queue_a, queue_b;
    DeviceBuffer<ResultLeaf> results;
    DeviceBuffer<int> counters;
    size_t capacity;
    explicit IPMWorkspace(size_t capacity) : capacity(capacity) {}
    void ensure_ipm_buffers();
    void clear() noexcept {
        queue_a.clear(); queue_b.clear(); results.clear(); counters.clear();
    }
};

int refine_ipm(IPMWorkspace& work, const PlaneData& planes,
    double img_xmin, double img_xmax, double img_ymin, double img_ymax,
    double coarse_size, double min_size, double distortion_threshold, int max_iters,
    double c_xmin, double c_xmax, double c_ymin, double c_ymax);

double probe_buffer(const PlaneData& planes,
    double img_cx, double img_cy, double base_w, double base_h,
    double src_xmin, double src_xmax, double src_ymin, double src_ymax,
    double buffer_step, int max_steps);
