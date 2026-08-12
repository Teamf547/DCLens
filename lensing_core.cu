/* lensing_core.cu */

/*
update: 2026-03-16
we add the time delay module
we change the lens profile from einstein radius to mass
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

// ==== 替换后的代码 (彻底弃用 GLEW) ====
#define GL_GLEXT_PROTOTYPES 1 // 核心：告诉编译器直接使用系统的 OpenGL 扩展库
#include <GL/gl.h>
#include <GL/glext.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda_gl_interop.h>
#include <stdexcept>

// define namespace and structs

namespace py = pybind11;

// double G = 6.67430e-11;
// double M_sun = 1.98847e30;
// double c = 299792458;
// double pc_in_m = 3.0857e16;
const int MAX_RESULT_SIZE = 20000000;


// test cuda
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error("CUDA Error"); \
        } \
    } while (0)

struct Point { double x, y;};

struct Lens {
    double x, y, M; // square of einstein ring radius
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

// =====================================================================
// 新增：OpenGL 着色器代码 (用于硬件光栅化)
// =====================================================================
const char* vertex_shader_src = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in float aMag;
out float vMag;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vMag = aMag; // 将放大倍数传递给片段着色器
}
)";

const char* fragment_shader_src = R"(
#version 330 core
in float vMag;
out float FragColor;
void main() {
    FragColor = vMag; // 直接输出放大倍数，配合 GL_BLEND 累加
}
)";

// =====================================================================
// 修复后：CUDA Kernel - 将 ResultLeaf 转换为 OpenGL VBO 数据
// =====================================================================
__global__ void map_leaves_to_vbo_kernel(
    const ResultLeaf* __restrict__ leaves,
    int num_leaves,
    float* __restrict__ vbo_data,
    double src_xmin, double src_xmax, double src_ymin, double src_ymax
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_leaves) return;

    const ResultLeaf& leaf = leaves[idx];
    int out_idx = idx * 18; // 6个顶点 * 每个顶点3个float

    // 【核心修复】如果叶子无效，写入退化三角形将其丢弃在视口之外，彻底消除垃圾显存！
    if (leaf.magnification <= 0.0) {
        for (int i = 0; i < 18; i += 3) {
            vbo_data[out_idx + i + 0] = -2.0f; // x: 挪到屏幕外
            vbo_data[out_idx + i + 1] = -2.0f; // y: 挪到屏幕外
            vbo_data[out_idx + i + 2] = 0.0f;  // mag: 放大倍数为0
        }
        return;
    }

    // 转换为 OpenGL 标准设备坐标 (NDC) [-1, 1]
    auto to_ndc = [&](double x, double y, float& nx, float& ny) {
        nx = (float)(2.0 * (x - src_xmin) / (src_xmax - src_xmin) - 1.0);
        ny = (float)(2.0 * (y - src_ymin) / (src_ymax - src_ymin) - 1.0);
    };

    float px[4], py[4];
    to_ndc(leaf.p0.x, leaf.p0.y, px[0], py[0]);
    to_ndc(leaf.p1.x, leaf.p1.y, px[1], py[1]);
    to_ndc(leaf.p2.x, leaf.p2.y, px[2], py[2]);
    to_ndc(leaf.p3.x, leaf.p3.y, px[3], py[3]);

    float dx02 = px[0] - px[2]; float dy02 = py[0] - py[2];
    float dist02_sq = dx02*dx02 + dy02*dy02;
    float dx13 = px[1] - px[3]; float dy13 = py[1] - py[3];
    float dist13_sq = dx13*dx13 + dy13*dy13;

    float mag = (float)leaf.magnification;

    auto write_vertex = [&](int v_idx, int p_idx) {
        vbo_data[out_idx + v_idx*3 + 0] = px[p_idx];
        vbo_data[out_idx + v_idx*3 + 1] = py[p_idx];
        vbo_data[out_idx + v_idx*3 + 2] = mag;
    };

    if (dist02_sq <= dist13_sq) {
        write_vertex(0, 0); write_vertex(1, 1); write_vertex(2, 2);
        write_vertex(3, 0); write_vertex(4, 2); write_vertex(5, 3);
    } else {
        write_vertex(0, 0); write_vertex(1, 1); write_vertex(2, 3);
        write_vertex(3, 1); write_vertex(4, 2); write_vertex(5, 3);
    }
}

// --- Device Functions ---

// ray_shotting
// 【恢复原版薄透镜多平面公式】=================================================
// 说明：早期实现改成了 "绝对物理坐标 + 单位混乱(rad·pc)" 的追踪公式，
// 导致撞击参数 r2 的量级错误，src_area 被系统性放大，最终 mag=img_area/src_area
// 整体偏小约 2 个数量级。此处恢复为 DCLens_RELEASE 的标准薄透镜公式：
//   对每个透镜平面 i：
//     撞击参数(物理距离/pc) ksix = (theta_rad - lens_rad) * D_L_i
//     物理偏折角  phyalpha = 4G M / (c² R) * ksi / |ksi|²
//     归约偏折角  alpha = factor_i (= D_iS/D_S) * phyalpha
//     beta -= alpha
// 其中透镜坐标以 arcsec 传入，内部经 ARCSEC2RAD 转为 rad。
// ===================================================================== */
__device__ Point ray_shoot_device(double theta_x_arcsec, double theta_y_arcsec, 
                                 const Lens* all_lenses, 
                                 const PlaneInfo* planes, 
                                 int num_planes,
                                 double D_S_pc) {

    constexpr double ARCSEC2RAD = 4.84813681109536e-6;
    constexpr double RAD2ARCSEC = 206264.806247096;
    constexpr double G = 6.67430e-11;
    constexpr double M_sun = 1.98847e30;
    constexpr double c = 299792458.0;
    constexpr double pc_in_m = 3.085677581e16;
    constexpr double KAPPA_CONST = (4.0 * G * M_sun) / (c * c * pc_in_m); 

    // 初始视角度 (rad)
    double theta_x = theta_x_arcsec * ARCSEC2RAD;
    double theta_y = theta_y_arcsec * ARCSEC2RAD;

    // 维护三个核心状态变量
    double beta_x = theta_x;  
    double beta_y = theta_y;
    
    double gamma_x = theta_x; 
    double gamma_y = theta_y;
    
    double xi_x = 0.0;        
    double xi_y = 0.0;
    
    double current_z = 0.0;

    for (int i = 0; i < num_planes; ++i) {
        const PlaneInfo& p = planes[i];
        
        // 1. 推进到当前平面的物理位置
        double dz = p.z - current_z; 
        xi_x += gamma_x * dz;
        xi_y += gamma_y * dz;
        current_z = p.z; 

        double phyalpha_x_sum = 0.0; 
        double phyalpha_y_sum = 0.0; 
        
        for (int j = 0; j < p.lens_count; ++j) {
            const Lens& l = all_lenses[p.lens_start_idx + j];
            
            // 2. 减去透镜位置得到物理撞击参数
            double dx = xi_x - l.x; 
            double dy = xi_y - l.y; 
            double r2 = dx * dx + dy * dy; 
            
            if (r2 < 1e-12) r2 = 1e-12; 
            
            // 3. 计算物理偏折角
            double phyalpha = KAPPA_CONST * l.M / r2; 
            phyalpha_x_sum += phyalpha * dx;
            phyalpha_y_sum += phyalpha * dy;
        }

        // 4. 在最终源平面映射角 beta 上累减
        double weight = (D_S_pc - p.z) / D_S_pc; 
        beta_x -= weight * phyalpha_x_sum;
        beta_y -= weight * phyalpha_y_sum;

        // 5. 更新下一层的局部传播角度 gamma
        gamma_x -= phyalpha_x_sum;
        gamma_y -= phyalpha_y_sum;
    }
    
    return {beta_x * RAD2ARCSEC, beta_y * RAD2ARCSEC};
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

// =====================================================================
// 新增：自动推断 Buffer Size 的启发式 Kernel
// =====================================================================
__global__ void check_buffer_frames_kernel(
    const Lens* __restrict__ all_lenses,
    const PlaneInfo* __restrict__ planes,
    int num_planes,
    double D_S_pc,
    double img_cx, double img_cy,
    double base_w, double base_h, // 基础包围盒尺寸
    double buffer_step,
    double src_xmin, double src_xmax,
    double src_ymin, double src_ymax,
    int* __restrict__ frame_valid
) {
    int frame_idx = blockIdx.x; 
    int pt_idx = threadIdx.x; // 每个 block 8 个 thread，分别对应 8 个 control points

    if (pt_idx >= 8) return;

    // 当前 frame 对应的 buffer 大小
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
    // 4个角点 + 4个中点
    if (pt_idx == 0) { px = x_min; py = y_min; } // 左下
    else if (pt_idx == 1) { px = x_max; py = y_min; } // 右下
    else if (pt_idx == 2) { px = x_max; py = y_max; } // 右上
    else if (pt_idx == 3) { px = x_min; py = y_max; } // 左上
    else if (pt_idx == 4) { px = x_mid; py = y_min; } // 下中
    else if (pt_idx == 5) { px = x_max; py = y_mid; } // 右中
    else if (pt_idx == 6) { px = x_mid; py = y_max; } // 上中
    else if (pt_idx == 7) { px = x_min; py = y_mid; } // 左中

    // 逆向光线追踪求源平面坐标
    Point src_pt = ray_shoot_device(px, py, all_lenses, planes, num_planes, D_S_pc);

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
    double c_xmin, double c_xmax, double c_ymin, double c_ymax,
    int max_results, // import the max_results to device for protect overflow / 2026-4-29
    double D_S_pc
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_in) return;

    Pixel p = in_queue[idx];

    // step1: ray shotting 4 corners of the pixel
    Point m0 = ray_shoot_device(p.x0, p.y1, all_lenses, planes, num_planes, D_S_pc); // TL
    Point m1 = ray_shoot_device(p.x1, p.y1, all_lenses, planes, num_planes, D_S_pc); // TR
    Point m2 = ray_shoot_device(p.x1, p.y0, all_lenses, planes, num_planes, D_S_pc); // BR
    Point m3 = ray_shoot_device(p.x0, p.y0, all_lenses, planes, num_planes, D_S_pc); // BL

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
    
    // 【核心修复】：提前检测自交，并将其加入有效性判断
    bool is_self_intersecting = (s1 * s2 < 0.0);
    bool is_valid = (src_area > 1e-20) && (!is_self_intersecting);
    
    // ================== 新增：几何畸变判定 ==================
    // 计算 s1 和 s2 的相对差异率 (范围 0 到 1)
    double abs_s1 = abs(s1);
    double abs_s2 = abs(s2);
    double distortion = abs(abs_s1 - abs_s2) / (abs_s1 + abs_s2 + 1e-20);
    // 设定畸变阈值，例如 0.2 表示两个三角形面积差异超过 20% 就说明发生了严重的非线性扭曲
    double distortion_threshold = 0.2; 
    // =======================================================

    double mag = -1.0;
    if (is_valid) {
        mag = img_area / src_area;
    }

    // step3: refine or not
    bool split = false;
    
    //// step3.1: detect whether ray elements area is self-intersection or high-magnificated
    if (is_self_intersecting || !is_valid) {
        split = true; 
    } else {
        //// step3.2: compare the magnification with thresholds
        // if (mag > mu_upper || mag < mu_lower) {
        if (distortion > distortion_threshold) {
            split = true;
        }
    }
    
    //// step3.3 break or not
    if (p.pix_size <= min_pixel_size) split = false;

    // step4: refine or store result
    if (split) {
        int out_idx = atomicAdd(out_count, 4); //
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
        // if (res_idx >= 200000000) return; // fix the protect of overflow / 2026-4-29
        if (res_idx >= max_results) return; // protect the overflow of results / 2026-4-29
        results[res_idx] = {m0, m1, m2, m3, is_valid ? mag : -1.0, p.pix_size};
    }
}

/* =====================================================================
   模块1：to_pixel_coord
   目的：将源平面坐标（arcsec）换算为源平面像素坐标。
   对应 CPU 端的等价逻辑（gpu_rasterizer.py + MMpS.py）：
       u = (x - src_xmin) / (src_xmax - src_xmin)
       v = (y - src_ymin) / (src_ymax - src_ymin)
       x_pix = u * nx;  y_pix = v * ny
   合并为一步：px = (x - src_xmin)/(src_xmax-src_xmin)*nx
   该函数将被光栅化 kernel 高频调用，用于处理每个叶子节点的角点，
   故封装为 __device__ inline 便捷函数。
   ===================================================================== */
__device__ inline void to_pixel_coord(double x, double y,
                                      double src_xmin, double src_xmax,
                                      double src_ymin, double src_ymax,
                                      int nx, int ny,
                                      float& px, float& py) {
    px = (float)((x - src_xmin) / (src_xmax - src_xmin) * nx);
    py = (float)((y - src_ymin) / (src_ymax - src_ymin) * ny);
}

/* =====================================================================
   模块2：rasterize_triangle_device
   目的：光栅化单个三角形，将覆盖到的每个源像素累加：
       - d_sum[idx] += scale   (放大倍数累加，对应 gpu_rasterizer 的 R 通道)
   输入为三角形三个顶点 (ax,ay),(bx,by),(cx,cy) 的像素坐标 [边函数值正负决定内外]。
   使用半空间测试(edge function)判断像素中心是否在三角形内部。
   对应 CPU 端 gpu_rasterizer.py 中对每个三角形的填充与累加逻辑，
   但全程在 GPU 上执行。
   ===================================================================== */

// 计算三角形 (a,b,c) 的有符号面积的两倍（即未归一化的边函数）。
// 不同顶点顺序会得到正/负值，本函数只用于判断同号性。
__device__ inline float tri_signed_area2(float ax, float ay, float bx, float by,
                                         float cx, float cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

__device__ inline void rasterize_triangle_device(
    float ax, float ay, float bx, float by, float cx, float cy, // 三角形顶点（像素坐标）
    float scale_value,        // 该三角形要写入的放大倍数
    int nx, int ny,           // 源网格尺寸（srcn=像素宽，ny=像素高）
    int ss,                   // 超采样因子 (ss=1 时不超采样)
    float* d_sum           // 放大倍数累加缓冲（size = ny*nx）
) {
    // 1. 计算三角形包围盒（在像素坐标下）
    float minx = fminf(fminf(ax, bx), cx);
    float maxx = fmaxf(fmaxf(ax, bx), cx);
    float miny = fminf(fminf(ay, by), cy);
    float maxy = fmaxf(fmaxf(ay, by), cy);

    // 2. 将包围盒裁剪到有效像素范围 [0, nx-1] × [0, ny-1]
    int x0 = (int)floorf(minx); if (x0 < 0) x0 = 0;
    int x1 = (int)ceilf(maxx);  if (x1 > nx - 1) x1 = nx - 1;
    int y0 = (int)floorf(miny); if (y0 < 0) y0 = 0;
    int y1 = (int)ceilf(maxy);  if (y1 > ny - 1) y1 = ny - 1;

    // 若包围盒为空则直接返回
    if (x0 > x1 || y0 > y1) return;

    float inv_ss = 1.0f / (float)ss;   // 子像素步长

    // 3. 遍历包围盒内每个像素
    for (int jy = y0; jy <= y1; ++jy) {
        for (int jx = x0; jx <= x1; ++jx) {
            int idx = jy * nx + jx;

            // 3.1 【超采样】遍历 ss×ss 个子像素中心并分别做半空间测试，
            //     统计该像素内被三角形覆盖的子像素个数。
            int inside_count = 0;
            for (int sy = 0; sy < ss; ++sy) {
                float pyc = (float)jy + ((float)sy + 0.5f) * inv_ss;
                for (int sx = 0; sx < ss; ++sx) {
                    float pxc = (float)jx + ((float)sx + 0.5f) * inv_ss;

                    float w0 = tri_signed_area2(bx, by, cx, cy, pxc, pyc);
                    float w1 = tri_signed_area2(cx, cy, ax, ay, pxc, pyc);
                    float w2 = tri_signed_area2(ax, ay, bx, by, pxc, pyc);

                    // 若所有边函数同号（三个 w 全非负 或 全非正），则子像素在三角形内部
                    if ((w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) ||
                        (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f)) {
                        inside_count++;
                    }
                }
            }

            // 3.2 若至少有 1 个子像素被覆盖，则累加。
            //     累加 inside_count 次 scale 与计数 —— 这使得最终
            //     等价于 CPU 端超采样的降采样平均 (sum/ss²)。
            if (inside_count > 0) {
                atomicAdd(&d_sum[idx], scale_value * (float)inside_count);
            }
        }
    }
}

/* =====================================================================
   模块3：rasterize_leaves_kernel（光栅化主 kernel）
   目的：遍历 GPU 上的所有叶子节点（ResultLeaf），将每个叶子的四边形
   三角化并光栅化累加到源像素图缓冲中。
   对应 CPU 端 MMpS.py 的三角化 + gpu_rasterizer.py 的三角形提交，
   但全程在 GPU 上执行，不发生设备→主机的中间数据传输。

   线程组织：每个线程负责一个三角形。
       leaf_idx = tid / 2
       tri_idx  = tid % 2
   即每个叶子（四边形 → 2 个三角形）由 2 个线程共同处理。

   三角化规则（与 CPU 端 MMpS.py 完全一致）：
       选择"较短对角线"进行切割（mask02: d02<=d13）。
         d02 较短 -> 方案A: T1=(0,1,2), T2=(0,2,3)
         d13 较短 -> 方案B: T1=(0,1,3), T2=(1,2,3)
   ===================================================================== */
__global__ void rasterize_leaves_kernel(
    const ResultLeaf* __restrict__ results,  // GPU上的叶子节点数组
    int num_leaves,                          // 有效叶子节点数量
    float* __restrict__ d_sum,               // 放大倍数累加缓冲
    int nx, int ny,                          // 源网格尺寸
    int ss,                                  // 超采样因子
    double src_xmin, double src_xmax,        // 源平面 x 边界
    double src_ymin, double src_ymax         // 源平面 y 边界
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ntriangle = 2 * num_leaves;          // 总三角形数 = 2 * 叶子数
    if (tid >= ntriangle) return;

    int leaf_idx = tid / 2;
    int tri_idx  = tid % 2;

    const ResultLeaf& leaf = results[leaf_idx];

    // 跳过无效/自交叶子（放大倍数为负的叶子），对应 CPU 端 valid_mask
    if (leaf.magnification <= 0.0) return;

    // 1. 将四个角点从 arcsec 坐标转换为像素坐标
    float p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y;
    to_pixel_coord(leaf.p0.x, leaf.p0.y, src_xmin, src_xmax, src_ymin, src_ymax,
                   nx, ny, p0x, p0y);
    to_pixel_coord(leaf.p1.x, leaf.p1.y, src_xmin, src_xmax, src_ymin, src_ymax,
                   nx, ny, p1x, p1y);
    to_pixel_coord(leaf.p2.x, leaf.p2.y, src_xmin, src_xmax, src_ymin, src_ymax,
                   nx, ny, p2x, p2y);
    to_pixel_coord(leaf.p3.x, leaf.p3.y, src_xmin, src_xmax, src_ymin, src_ymax,
                   nx, ny, p3x, p3y);

    float scale_value = (float)leaf.magnification;

    // 2. 计算两条对角线的平方距离，选择较短对角线
    float dx02 = p0x - p2x, dy02 = p0y - p2y;
    float dist02_sq = dx02 * dx02 + dy02 * dy02;
    float dx13 = p1x - p3x, dy13 = p1y - p3y;
    float dist13_sq = dx13 * dx13 + dy13 * dy13;

    // 3. 根据 tri_idx 选择对应的三角形并光栅化
    if (tri_idx == 0) {
        // 三角形1
        if (dist02_sq <= dist13_sq) {
            // 方案A: (p0, p1, p2)
            rasterize_triangle_device(p0x, p0y, p1x, p1y, p2x, p2y,
                                      scale_value, nx, ny, ss, d_sum);
        } else {
            // 方案B: (p0, p1, p3)
            rasterize_triangle_device(p0x, p0y, p1x, p1y, p3x, p3y,
                                      scale_value, nx, ny, ss, d_sum);
        }
    } else {
        // 三角形2
        if (dist02_sq <= dist13_sq) {
            // 方案A: (p0, p2, p3)
            rasterize_triangle_device(p0x, p0y, p2x, p2y, p3x, p3y,
                                      scale_value, nx, ny, ss, d_sum);
        } else {
            // 方案B: (p1, p2, p3)
            rasterize_triangle_device(p1x, p1y, p2x, p2y, p3x, p3y,
                                      scale_value, nx, ny, ss, d_sum);
        }
    }
}

/* =====================================================================
   模块4：finalize_map_kernel（归一化 kernel）
   得到每个源像素的平均放大倍数，即最终的源平面放大图 d_map。
   对应 CPU 端 gpu_rasterizer.py 的逻辑：
       valid_mask = (A > eps)
       result_tile[valid_mask] = R[valid_mask] / A[valid_mask]
   即：被三角形覆盖的像素取平均放大倍数，未覆盖的像素为 0。

   线程组织：每个线程处理一个源像素（npix = ny * nx）。
   ===================================================================== */
__global__ void finalize_map_kernel(
    const float* __restrict__ d_sum,   
    float* __restrict__ d_map,         
    int npix,
    int ss                             // 引入超采样因子
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix) return;

    // 完美复现原版 gpu_rasterizer 中物理累加后平均降采样的逻辑 (sum / (ss*ss))
    d_map[i] = d_sum[i] / (float)(ss * ss);
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
    size_t D_S_pc_global; // source distance in parsec

// OpenGL & EGL 相关变量
    EGLDisplay egl_display;
    EGLContext egl_context;
    EGLSurface egl_surface; // 【新增】用于防止驱动崩溃的 Dummy 表面
    GLuint fbo, rbo, vao, vbo, shader_program;
    cudaGraphicsResource_t vbo_resource;
    bool gl_initialized = false;

void init_opengl() {
        if (gl_initialized) return;

        char err_msg[512]; // 用于构建详细的错误信息

        // 1. 尝试硬件直通扩展 (针对无头环境)
        PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = 
            (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
        PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = 
            (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

        egl_display = EGL_NO_DISPLAY;

        if (eglQueryDevicesEXT && eglGetPlatformDisplayEXT) {
            EGLDeviceEXT eglDevs[16];
            EGLint numDevs = 0;
            eglQueryDevicesEXT(16, eglDevs, &numDevs);
            if (numDevs > 0) {
                // 注意：如果你的电脑(如笔记本)有核显+独显，通常 eglDevs[0] 是核显，eglDevs[1] 是独显。
                // 如果发现后续 CUDA 互操作报错，可以尝试把此处的 eglDevs[0] 改为 eglDevs[1]
                egl_display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevs[0], nullptr);
                printf("[OpenGL Info] EGL Device extension found %d GPU(s). Selected Device 0.\n", numDevs);
            } else {
                printf("[OpenGL Warn] eglQueryDevicesEXT found 0 devices.\n");
            }
        } else {
            printf("[OpenGL Warn] EGL device extensions not supported. Falling back to default display.\n");
        }

        if (egl_display == EGL_NO_DISPLAY) {
            egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        }
        if (egl_display == EGL_NO_DISPLAY) {
            throw std::runtime_error("EGL Error: eglGetDisplay failed (No display found).");
        }

        // 2. 初始化 EGL
        EGLint major, minor;
        if (!eglInitialize(egl_display, &major, &minor)) {
            sprintf(err_msg, "EGL Error: eglInitialize failed with EGL code 0x%X. (Hint: driver issue?)", eglGetError());
            throw std::runtime_error(err_msg);
        }

        // 3. 绑定 OpenGL API
        eglBindAPI(EGL_OPENGL_API);
        if (eglGetError() != EGL_SUCCESS) {
            throw std::runtime_error("EGL Error: eglBindAPI failed. Full OpenGL might not be supported on this driver.");
        }

        // 4. 选择配置
        EGLint configAttribs[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, 
            EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_RED_SIZE, 8,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE
        };
        EGLint numConfigs;
        EGLConfig egl_config;
        if (!eglChooseConfig(egl_display, configAttribs, &egl_config, 1, &numConfigs) || numConfigs == 0) {
            sprintf(err_msg, "EGL Error: eglChooseConfig failed with code 0x%X.", eglGetError());
            throw std::runtime_error(err_msg);
        }
        
        // 5. 创建上下文和虚拟表面
        EGLint contextAttribs[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 3, EGL_NONE };
        egl_context = eglCreateContext(egl_display, egl_config, EGL_NO_CONTEXT, contextAttribs);
        if (egl_context == EGL_NO_CONTEXT) {
            sprintf(err_msg, "EGL Error: eglCreateContext failed with code 0x%X.", eglGetError());
            throw std::runtime_error(err_msg);
        }

        EGLint pbufferAttribs[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
        egl_surface = eglCreatePbufferSurface(egl_display, egl_config, pbufferAttribs);
        if (egl_surface == EGL_NO_SURFACE) {
            sprintf(err_msg, "EGL Error: eglCreatePbufferSurface failed with code 0x%X.", eglGetError());
            throw std::runtime_error(err_msg);
        }

        if (!eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)) {
            sprintf(err_msg, "EGL Error: eglMakeCurrent failed with code 0x%X.", eglGetError());
            throw std::runtime_error(err_msg);
        }

        // --- 以下着色器与 VBO 初始化逻辑保持不变 ---
        auto compile_shader = [](GLenum type, const char* src) {
            GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &src, nullptr);
            glCompileShader(shader);
            GLint success;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                char infoLog[512];
                glGetShaderInfoLog(shader, 512, NULL, infoLog);
                throw std::runtime_error("OpenGL Error: Shader compilation failed.");
            }
            return shader;
        };
        GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex_shader_src);
        GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_src);
        shader_program = glCreateProgram();
        glAttachShader(shader_program, vs);
        glAttachShader(shader_program, fs);
        glLinkProgram(shader_program);

        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        gl_initialized = true;
    }


public:
    LensingEngineGPU(double D_S_pc, size_t queue_cap, size_t result_cap) 
        : D_S_pc_global(D_S_pc),max_queue_size(queue_cap), max_results(result_cap) 
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
        if (gl_initialized) {
            // 【移除】cudaGraphicsUnregisterResource(vbo_resource); (因为我们将它移到循环内释放了)
            glDeleteBuffers(1, &vbo);
            glDeleteVertexArrays(1, &vao);
            glDeleteProgram(shader_program);
            eglDestroySurface(egl_display, egl_surface); // 【新增】
            eglDestroyContext(egl_display, egl_context);
            eglTerminate(egl_display);
        }
    }

// =====================================================================
    // 新增：供 Python 调用的 Buffer 预估接口
    // =====================================================================
    double estimate_buffer_size(
        double img_cx, double img_cy,
        double base_w, double base_h,
        double src_xmin, double src_xmax,
        double src_ymin, double src_ymax,
        double buffer_step, int max_steps
    ) {
        int* d_frame_valid;
        CUDA_CHECK(cudaMalloc(&d_frame_valid, max_steps * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_frame_valid, 0, max_steps * sizeof(int)));

        // 启动 kernel：按步长发射多个 frame，每个 frame 8 个线程（测试8个点）
        check_buffer_frames_kernel<<<max_steps, 8>>>(
            d_all_lenses, d_planes, num_planes, D_S_pc_global,
            img_cx, img_cy, base_w, base_h, buffer_step,
            src_xmin, src_xmax, src_ymin, src_ymax,
            d_frame_valid
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int> h_frame_valid(max_steps);
        CUDA_CHECK(cudaMemcpy(h_frame_valid.data(), d_frame_valid, max_steps * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_frame_valid));

        // 从内向外寻找第一个让 8 个 control points 全部出界的 frame
        for (int i = 0; i < max_steps; ++i) {
            if (h_frame_valid[i] == 1) {
                return (i + 1) * buffer_step;
            }
        }

        // 如果全部搜完都没有，返回最大容许值，让 Python 端去打 Warning
        return max_steps * buffer_step;
    }

    void load_planes(const std::vector<std::vector<double>>& lx,
                     const std::vector<std::vector<double>>& ly,
                     const std::vector<std::vector<double>>& lM,
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
                flat_lenses.push_back({lx[i][j], ly[i][j], lM[i][j]});
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
        double c_xmin, double c_xmax, double c_ymin, double c_ymax
    ) {

        // step1: initialize the corase mesh
        std::vector<Pixel> init_pixels;
        int nx = (int)std::ceil((img_xmax - img_xmin) / coarse_size - 1e-7); // make sure the division is correct even with floating point precision
        int ny = (int)std::ceil((img_ymax - img_ymin) / coarse_size - 1e-7);
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
                current_count, min_size,
                c_xmin, c_xmax, c_ymin, c_ymax,
                (int)max_results, // import the max_results to device for protect overflow / 2026-4-29
                D_S_pc_global
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

        int copied_results = std::min(total_results, (int)max_results);//protect/2026-3-23

        std::vector<ResultLeaf> host_results(total_results);
        // CUDA_CHECK(cudaMemcpy(host_results.data(), d_results, total_results * sizeof(ResultLeaf), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_results.data(), d_results, copied_results * sizeof(ResultLeaf), cudaMemcpyDeviceToHost));

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

    /* =====================================================================
       模块5：run_adaptive_rasterize（GPU 端到端主流程）

       目的：将自适应 quadtree refine + 三角化光栅化 + 归一化全部在 GPU
       上串联完成，只把最终的源平面放大图（ny_src × nx_src）复制回主机，
       从而完全消除中间叶子节点数据的 CPU<->GPU 往复 IO。

       流程（纯 GPU）：
           step1: 初始化粗网格，写入 d_queue_A
           step2: 迭代 refine_kernel，叶子节点累积到 d_results
           step4: rasterize_leaves_kernel 光栅化累加（模块3）
           step5: finalize_map_kernel 归一化得到 d_map（模块4）
           step6: 唯一一次 GPU->CPU 拷贝（ny_src*nx_src 的最终图）
           step7: 释放临时缓冲
        ===================================================================== */
    py::array_t<float> run_adaptive_rasterize(
        double img_xmin, double img_xmax, double img_ymin, double img_ymax,
        double coarse_size, double min_size, int max_iters,
        double c_xmin, double c_xmax, double c_ymin, double c_ymax,
        int nx_src, int ny_src,
        int ss = 1                          // 超采样因子 (ss=1 时不超采样)
    ) {

        // ---- step1: initialize the coarse mesh（与 run_adaptive 相同）----
        std::vector<Pixel> init_pixels;
        int nx = (int)std::ceil((img_xmax - img_xmin) / coarse_size - 1e-7);
        int ny = (int)std::ceil((img_ymax - img_ymin) / coarse_size - 1e-7);
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

        // ---- step2: iterative refine（与 run_adaptive 相同）----
        for (int iter = 0; iter < max_iters; ++iter) {
            if (current_count == 0) break;

            CUDA_CHECK(cudaMemset(d_counters, 0, sizeof(int)));

            int threads = 256;
            int blocks = (current_count + threads - 1) / threads;

            refine_kernel<<<blocks, threads>>>(
                in_ptr, out_ptr, d_results, d_all_lenses, d_planes, num_planes,
                d_counters,
                d_counters + 1,
                current_count, min_size, 
                c_xmin, c_xmax, c_ymin, c_ymax,
                (int)max_results,
                D_S_pc_global
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
            if (next_gen_count > (int)max_queue_size) next_gen_count = (int)max_queue_size;

            std::swap(in_ptr, out_ptr);
            current_count = next_gen_count;
        }

        int total_results = 0;
        CUDA_CHECK(cudaMemcpy(&total_results, d_counters + 1, sizeof(int), cudaMemcpyDeviceToHost));
        // 光栅化用的叶子数不能超过 d_results 的容量
        int used_results = std::min(total_results, (int)max_results);

        // ---- step3: 分配并清零源像素累加缓冲 ----
        size_t npix = (size_t)nx_src * (size_t)ny_src;
        float* d_sum = nullptr;
        float* d_map = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sum, npix * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_map, npix * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sum, 0, npix * sizeof(float)));

        // ---- step4: 光栅化（模块3）----
        {
            int threads = 256;
            long ntriangle = 2L * (long)used_results;
            int blocks = (int)((ntriangle + threads - 1) / threads);
            if (ntriangle > 0) {
                rasterize_leaves_kernel<<<blocks, threads>>>(
                    d_results, used_results, d_sum,
                    nx_src, ny_src, ss,
                    c_xmin, c_xmax, c_ymin, c_ymax
                );
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- step5: 归一化（模块4）----
        {
            int threads = 256;
            long npix_l = (long)npix;
            int blocks = (int)((npix_l + threads - 1) / threads);
            finalize_map_kernel<<<blocks, threads>>>(d_sum, d_map, (int)npix, ss);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- step6: 唯一一次 GPU->CPU 拷贝 ----
        py::array_t<float> result({(long)ny_src, (long)nx_src});
        // 直接使用 py::array_t 的 mutable_data() 获取底层连续缓冲区指针
        CUDA_CHECK(cudaMemcpy(result.mutable_data(), d_map, npix * sizeof(float), cudaMemcpyDeviceToHost));

        // ---- step7: 释放临时缓冲 ----
        cudaFree(d_sum);
        cudaFree(d_map);

        return result;
    }

    // =====================================================================
    // 新增：在此处粘贴端到端 CUDA-OpenGL 硬件光栅化代码！
    // 它是 LensingEngineGPU 类的最后一个成员函数。
    // =====================================================================
    py::array_t<float> run_adaptive_gl_rasterize(
        double img_xmin, double img_xmax, double img_ymin, double img_ymax,
        double coarse_size, double min_size, int max_iters,
        double c_xmin, double c_xmax, double c_ymin, double c_ymax,
        int nx_src, int ny_src,
        int ss = 1 // 超采样因子
    ) {
        // 1. 初始化 OpenGL (如果未初始化)
        init_opengl();

        // 2. 运行原有的自适应四叉树划分逻辑获取叶子节点 (复用原版逻辑)
        std::vector<Pixel> init_pixels;
        int nx = (int)std::ceil((img_xmax - img_xmin) / coarse_size - 1e-7);
        int ny = (int)std::ceil((img_ymax - img_ymin) / coarse_size - 1e-7);
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
                in_ptr, out_ptr, d_results, d_all_lenses, d_planes, num_planes,
                d_counters, d_counters + 1, current_count, min_size, 
                c_xmin, c_xmax, c_ymin, c_ymax, (int)max_results, D_S_pc_global
            );
            CUDA_CHECK(cudaDeviceSynchronize());

            int h_counts[2];
            CUDA_CHECK(cudaMemcpy(h_counts, d_counters, 2 * sizeof(int), cudaMemcpyDeviceToHost));
            int next_gen_count = h_counts[0];
            
            if (next_gen_count == 0) break;
            if (next_gen_count > (int)max_queue_size) next_gen_count = (int)max_queue_size;

            std::swap(in_ptr, out_ptr);
            current_count = next_gen_count;
        }

        int total_results = 0;
        CUDA_CHECK(cudaMemcpy(&total_results, d_counters + 1, sizeof(int), cudaMemcpyDeviceToHost));
        int used_results = std::min(total_results, (int)max_results);

        // 3. 配置高精度 FBO (R32F，用于精确累加浮点数放大倍数)
        int render_nx = nx_src * ss;
        int render_ny = ny_src * ss;
        GLuint current_fbo, texture;
        glGenFramebuffers(1, &current_fbo);
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, render_nx, render_ny, 0, GL_RED, GL_FLOAT, nullptr);
        glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

        // 4. 按需动态分配 VBO 并绑定 CUDA (极大降低显存压力和崩溃率)
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        size_t vbo_size = (size_t)used_results * 18 * sizeof(float);
        if (vbo_size == 0) vbo_size = 18 * sizeof(float); // 防止分配 0 字节引发 OpenGL 错误
        glBufferData(GL_ARRAY_BUFFER, vbo_size, nullptr, GL_DYNAMIC_DRAW);

        // 用安全的宏包装互操作，如果失败会抛出异常供 Jupyter 捕获
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));

        float* d_vbo_data;
        size_t num_bytes;
        CUDA_CHECK(cudaGraphicsMapResources(1, &vbo_resource, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_data, &num_bytes, vbo_resource));

        // 5. CUDA 并行生成 OpenGL 顶点
        if (used_results > 0) {
            int gl_threads = 256;
            int gl_blocks = (used_results + gl_threads - 1) / gl_threads;
            map_leaves_to_vbo_kernel<<<gl_blocks, gl_threads>>>(
                d_results, used_results, d_vbo_data, c_xmin, c_xmax, c_ymin, c_ymax
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 【核心修复】映射完顶点后，立刻解绑，把控制权还给 OpenGL
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &vbo_resource, 0));
        CUDA_CHECK(cudaGraphicsUnregisterResource(vbo_resource));

        // 6. 硬件光栅化渲染
        glViewport(0, 0, render_nx, render_ny);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // 启用加法混合模式 (等价于 atomicAdd)
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE); 

        glUseProgram(shader_program);
        glBindVertexArray(vao);
        if (used_results > 0) {
            glDrawArrays(GL_TRIANGLES, 0, used_results * 6); 
        }
        glDisable(GL_BLEND);

        // 7. 回读数据到 CPU
        std::vector<float> pixels(render_nx * render_ny);
        glReadPixels(0, 0, render_nx, render_ny, GL_RED, GL_FLOAT, pixels.data());

        // 清理当前帧的 FBO
        glDeleteTextures(1, &texture);
        glDeleteFramebuffers(1, &current_fbo);

        // 8. 降采样 (等价于 finalize_map_kernel 的平均化)
        py::array_t<float> result({(long)ny_src, (long)nx_src});
        auto r_ptr = result.mutable_unchecked<2>();
        
        float inv_ss2 = 1.0f / (ss * ss);
        for (int y = 0; y < ny_src; ++y) {
            for (int x = 0; x < nx_src; ++x) {
                float sum = 0.0f;
                for (int sy = 0; sy < ss; ++sy) {
                    for (int sx = 0; sx < ss; ++sx) {
                        sum += pixels[(y * ss + sy) * render_nx + (x * ss + sx)];
                    }
                }
                r_ptr(y, x) = sum * inv_ss2;
            }
        }
        return result;
    }
};



// PYBIND11_MODULE(lensing_core, m) {
//     py::class_<LensingEngineGPU>(m, "LensingEngineGPU")
//         .def(py::init<double, size_t, size_t>(), py::arg("D_S"), py::arg("queue_cap")=MAX_RESULT_SIZE, py::arg("res_cap")=MAX_RESULT_SIZE)
//         .def("load_planes", &LensingEngineGPU::load_planes)
//         .def("run_adaptive", &LensingEngineGPU::run_adaptive)
//         .def("run_adaptive_rasterize", &LensingEngineGPU::run_adaptive_rasterize)
//         .def("run_adaptive_gl_rasterize", &LensingEngineGPU::run_adaptive_gl_rasterize); // 注册新函数
// }

PYBIND11_MODULE(lensing_core, m) {
    py::class_<LensingEngineGPU>(m, "LensingEngineGPU")
        .def(py::init<double, size_t, size_t>(), py::arg("D_S"), py::arg("queue_cap")=MAX_RESULT_SIZE, py::arg("res_cap")=MAX_RESULT_SIZE)
        .def("load_planes", &LensingEngineGPU::load_planes)
        .def("run_adaptive", &LensingEngineGPU::run_adaptive)
        .def("run_adaptive_rasterize", &LensingEngineGPU::run_adaptive_rasterize)
        .def("run_adaptive_gl_rasterize", &LensingEngineGPU::run_adaptive_gl_rasterize)
        // 确保导出了 estimate_buffer_size 接口：
        .def("estimate_buffer_size", &LensingEngineGPU::estimate_buffer_size,
             py::arg("img_cx"), py::arg("img_cy"), py::arg("base_w"), py::arg("base_h"),
             py::arg("src_xmin"), py::arg("src_xmax"), py::arg("src_ymin"), py::arg("src_ymax"),
             py::arg("buffer_step"), py::arg("max_steps"));
}