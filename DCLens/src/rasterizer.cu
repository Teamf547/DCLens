// Rasterization Shader stage: leaf triangulation, additive rendering and readback.
// Short-diagonal triangulation, additive R32F rasterization, ss*ss averaging.
#include "rasterizer.cuh"
#include <cstdio>
#include <algorithm>
#include <vector>

// OpenGL shaders accumulate per-leaf magnification in an R32F target.
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

// One thread maps one leaf (two triangles, six vertices) into the GL VBO.
__global__ void map_leaves_to_vbo_kernel(
    const ResultLeaf* __restrict__ leaves,
    int num_leaves,
    float* __restrict__ vbo_data,
    double src_xmin, double src_xmax, double src_ymin, double src_ymax
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_leaves) return;

    const ResultLeaf& leaf = leaves[idx];
    int out_idx = idx * 18; // six vertices, each with x, y, magnification

    // Invalid leaves become degenerate triangles outside the viewport.
    if (leaf.magnification <= 0.0) {
        for (int i = 0; i < 18; i += 3) {
            vbo_data[out_idx + i + 0] = -2.0f;
            vbo_data[out_idx + i + 1] = -2.0f;
            vbo_data[out_idx + i + 2] = 0.0f;
        }
        return;
    }

    // Convert source-plane arcseconds to OpenGL NDC coordinates.
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

void Rasterizer::init_opengl() {
        if (gl_initialized) return;

        char err_msg[512]; // 用于构建详细的错误信息

        // Select the EGL device belonging to the current CUDA device, which
        // matters on laptops with both integrated graphics and an NVIDIA GPU.
        auto query_devices = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(
            eglGetProcAddress("eglQueryDevicesEXT"));
        auto platform_display = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
            eglGetProcAddress("eglGetPlatformDisplayEXT"));
        auto device_attribute = reinterpret_cast<PFNEGLQUERYDEVICEATTRIBEXTPROC>(
            eglGetProcAddress("eglQueryDeviceAttribEXT"));
        int cuda_device=0;
        CUDA_CHECK(cudaGetDevice(&cuda_device));
        if (query_devices && platform_display && device_attribute) {
            EGLDeviceEXT devices[16]; EGLint count=0;
            query_devices(16, devices, &count);
            for (int i=0; i<count; ++i) {
                EGLAttrib ordinal=-1;
                if (device_attribute(devices[i], EGL_CUDA_DEVICE_NV, &ordinal)
                        && ordinal==cuda_device) {
                    egl_display=platform_display(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
                    break;
                }
            }
        }
        if (egl_display==EGL_NO_DISPLAY) egl_display=eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (egl_display==EGL_NO_DISPLAY)
            throw std::runtime_error("EGL: no display for CUDA rasterization");

        // 2. 初始化 EGL
        EGLint major, minor;
        if (!eglInitialize(egl_display, &major, &minor)) {
            sprintf(err_msg, "EGL Error: eglInitialize failed with EGL code 0x%X. (Hint: driver issue?)", eglGetError());
            throw std::runtime_error(err_msg);
        }

        // 3. 绑定 OpenGL API
        if (!eglBindAPI(EGL_OPENGL_API)) {
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
                glDeleteShader(shader);
                throw std::runtime_error(std::string("OpenGL shader: ") + infoLog);
            }
            return shader;
        };
        GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex_shader_src);
        GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_src);
        shader_program = glCreateProgram();
        glAttachShader(shader_program, vs);
        glAttachShader(shader_program, fs);
        glLinkProgram(shader_program);
        glDeleteShader(vs); glDeleteShader(fs);
        GLint linked=0; glGetProgramiv(shader_program, GL_LINK_STATUS, &linked);
        if (!linked) throw std::runtime_error("OpenGL shader link failed");

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

void Rasterizer::render(const ResultLeaf* d_results, int used_results,
    double c_xmin, double c_xmax, double c_ymin, double c_ymax,
    int nx_src, int ny_src, int ss, float* output) {
    init_opengl();
    if (!eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context))
        throw std::runtime_error("EGL: cannot activate rasterizer context");
    GLint max_texture=0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture);
    if (nx_src > max_texture/ss || ny_src > max_texture/ss)
        throw std::runtime_error("Source grid times ss exceeds GL_MAX_TEXTURE_SIZE; reduce ss or map dimensions");
    int render_nx = nx_src * ss, render_ny = ny_src * ss;
    size_t free_bytes=0, total_bytes=0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t target_bytes = size_t(render_nx)*render_ny*sizeof(float);
    size_t vbo_size = std::max(size_t(used_results),size_t(1))*18*sizeof(float);
    if (target_bytes + vbo_size > size_t(free_bytes*0.8))
        throw CudaMemoryError("Insufficient GPU memory for rasterizer target/VBO; reduce ss, map dimensions or chunk size");

    // R32F with additive blending preserves the per-leaf magnification sum.
    glGenFramebuffers(1, &current_fbo);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, render_nx, render_ny, 0, GL_RED, GL_FLOAT, nullptr);
    glBindFramebuffer(GL_FRAMEBUFFER, current_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("OpenGL R32F framebuffer allocation failed");

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vbo_size, nullptr, GL_DYNAMIC_DRAW);
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsMapResources(1, &vbo_resource, 0));
    resource_mapped=true;
    float* d_vbo_data=nullptr;
    size_t num_bytes=0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_data, &num_bytes, vbo_resource));
    if (used_results > 0) {
        int threads=256, blocks=(used_results+threads-1)/threads;
        map_leaves_to_vbo_kernel<<<blocks, threads>>>(
            d_results, used_results, d_vbo_data, c_xmin, c_xmax, c_ymin, c_ymax);
    }
    CUDA_CHECK(cudaGetLastError());
    // Unmapping transfers ownership back to OpenGL before drawing the VBO.
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &vbo_resource, 0));
    resource_mapped=false;
    CUDA_CHECK(cudaGraphicsUnregisterResource(vbo_resource));
    vbo_resource=nullptr;

    glViewport(0, 0, render_nx, render_ny);
    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glUseProgram(shader_program);
    glBindVertexArray(vao);
    if (used_results > 0) glDrawArrays(GL_TRIANGLES, 0, used_results*6);
    glDisable(GL_BLEND);

    std::vector<float> pixels(size_t(render_nx)*render_ny);
    glReadPixels(0, 0, render_nx, render_ny, GL_RED, GL_FLOAT, pixels.data());
    if (glGetError()!=GL_NO_ERROR) throw std::runtime_error("OpenGL rasterization/readback failed");
    release_target();
    // Keep the established CPU supersample summation order and normalization.
    float inv_ss2 = 1.0f / (ss * ss);
    for (int y=0; y<ny_src; ++y) {
        for (int x=0; x<nx_src; ++x) {
            float sum=0.f;
            for (int sy=0; sy<ss; ++sy) {
                for (int sx=0; sx<ss; ++sx) {
                    sum += pixels[size_t(y*ss+sy)*render_nx+(x*ss+sx)];
                }
            }
            output[size_t(y)*nx_src+x] = sum*inv_ss2;
        }
    }
}

void Rasterizer::release_target() noexcept {
    if (resource_mapped) cudaGraphicsUnmapResources(1, &vbo_resource, 0);
    resource_mapped=false;
    if (vbo_resource) cudaGraphicsUnregisterResource(vbo_resource);
    vbo_resource=nullptr;
    if (texture) glDeleteTextures(1, &texture);
    if (current_fbo) glDeleteFramebuffers(1, &current_fbo);
    texture=0; current_fbo=0;
}
void Rasterizer::close() noexcept {
    if (egl_display!=EGL_NO_DISPLAY) {
        if (egl_context!=EGL_NO_CONTEXT) {
            eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);
            release_target();
            if (vbo) glDeleteBuffers(1, &vbo);
            if (vao) glDeleteVertexArrays(1, &vao);
            if (shader_program) glDeleteProgram(shader_program);
            eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            eglDestroyContext(egl_display, egl_context);
        }
        if (egl_surface!=EGL_NO_SURFACE) eglDestroySurface(egl_display, egl_surface);
        eglTerminate(egl_display);
    }
    vbo=vao=shader_program=0;
    egl_display=EGL_NO_DISPLAY; egl_context=EGL_NO_CONTEXT; egl_surface=EGL_NO_SURFACE;
    gl_initialized=false;
}
