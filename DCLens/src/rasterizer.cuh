#pragma once
#include "ipm.cuh"
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda_gl_interop.h>

// One EGL context per simulation. Targets are released after each image chunk;
// mapped leaves stay on the GPU until converted into the triangle VBO.
class Rasterizer {
    EGLDisplay egl_display = EGL_NO_DISPLAY;
    EGLContext egl_context = EGL_NO_CONTEXT;
    EGLSurface egl_surface = EGL_NO_SURFACE;
    GLuint vao=0, vbo=0, shader_program=0, current_fbo=0, texture=0;
    cudaGraphicsResource_t vbo_resource = nullptr;
    bool gl_initialized=false, resource_mapped=false;
    void init_opengl();
    void release_target() noexcept;
public:
    Rasterizer() = default;
    Rasterizer(const Rasterizer&) = delete;
    Rasterizer& operator=(const Rasterizer&) = delete;
    ~Rasterizer() { close(); }
    void close() noexcept;
    void render(const ResultLeaf* d_results, int used_results,
        double c_xmin, double c_xmax, double c_ymin, double c_ymax,
        int nx_src, int ny_src, int ss, float* output);
};
