# gpu_rasterizer.py

import math
import numpy as np
import logging

logger = logging.getLogger("gpu_rasterizer")

try:
    import moderngl
    HAVE_MGL = True
    logger.info("moderngl 可用 -> 使用 GPU 硬件光栅化路径。")
except Exception as e:
    HAVE_MGL = False
    logger.error("moderngl 不可用，请安装 moderngl。错误: %s", e)

# GLSL shaders
VS_SRC = """
#version 330
in vec2 in_pos; // 归一化局部坐标 [0,1]
in float in_scale; // per-vertex scale 值（每个三角形相同）
out float v_scale;
void main() {
    v_scale = in_scale;
    vec2 ndc = in_pos * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""
FS_SRC = """
#version 330
in float v_scale;
out vec4 fragColor;
void main() {
    // R accumulates contribution (v_scale)
    // A accumulates coverage (1.0 per fragment)
    fragColor = vec4(v_scale, 0.0, 0.0, 1.0);
}
"""

def rasterize_triangles_to_texture_tiled(tri_verts_uv_scale, nx_tex, ny_tex,
                                         ss=1, max_tile_dim=4096, batch_triangles=200000):

    if not HAVE_MGL:
        raise RuntimeError("moderngl 不可用")
    if tri_verts_uv_scale is None or tri_verts_uv_scale.shape[0] == 0:
        logger.error("没有三角形顶点，无法渲染。")
        return np.zeros((ny_tex, nx_tex), dtype=np.float32)
    n_vertices = tri_verts_uv_scale.shape[0]
    if n_vertices % 3 != 0:
        raise ValueError("顶点数量不是三的倍数")
    n_tris = n_vertices // 3
    tris = tri_verts_uv_scale.reshape((n_tris, 3, 3))
    # bbox in pixel coords
    u_vals = tris[:,:,0]
    v_vals = tris[:,:,1]
    tri_min_u = np.min(u_vals, axis=1)
    tri_max_u = np.max(u_vals, axis=1)
    tri_min_v = np.min(v_vals, axis=1)
    tri_max_v = np.max(v_vals, axis=1)
    tri_min_x = np.floor(tri_min_u * nx_tex).astype(np.int64)
    tri_max_x = np.ceil(tri_max_u * nx_tex).astype(np.int64) - 1
    tri_min_y = np.floor(tri_min_v * ny_tex).astype(np.int64)
    tri_max_y = np.ceil(tri_max_v * ny_tex).astype(np.int64) - 1
    tri_min_x = np.clip(tri_min_x, 0, nx_tex - 1)
    tri_max_x = np.clip(tri_max_x, 0, nx_tex - 1)
    tri_min_y = np.clip(tri_min_y, 0, ny_tex - 1)
    tri_max_y = np.clip(tri_max_y, 0, ny_tex - 1)
    max_fbo_dim = max_tile_dim
    tile_w = min(nx_tex, max(1, max_fbo_dim // max(1, ss)))
    tile_h = tile_w
    ntx = int(math.ceil(nx_tex / tile_w))
    nty = int(math.ceil(ny_tex / tile_h))
    logger.info("分块渲染：nx_tex=%d ny_tex=%d, tile_w=%d tile_h=%d, ntx=%d nty=%d, ss=%d, n_tris=%d",
                nx_tex, ny_tex, tile_w, tile_h, ntx, nty, ss, n_tris)

    try:
        ctx = moderngl.create_standalone_context(backend='egl')
    except Exception:
        ctx = moderngl.create_standalone_context()
    prog = ctx.program(vertex_shader=VS_SRC, fragment_shader=FS_SRC)
    pixel_mu_map = np.zeros((ny_tex, nx_tex), dtype=np.float32)
    for ty in range(nty):
        y0 = ty * tile_h
        y1 = min((ty + 1) * tile_h, ny_tex) - 1
        th = y1 - y0 + 1
        for tx in range(ntx):
            x0 = tx * tile_w
            x1 = min((tx + 1) * tile_w, nx_tex) - 1
            tw = x1 - x0 + 1
            mask_x = (tri_min_x <= x1) & (tri_max_x >= x0)
            mask_y = (tri_min_y <= y1) & (tri_max_y >= y0)
            tri_mask = mask_x & mask_y
            tri_indices = np.nonzero(tri_mask)[0]
            if tri_indices.size == 0:
                continue
            selected_tris = tris[tri_indices]
            sel_vertices = selected_tris.reshape((-1, 3)).copy()
            u_global = sel_vertices[:,0]
            v_global = sel_vertices[:,1]
            scales = sel_vertices[:,2]
            x_pix = u_global * float(nx_tex)
            y_pix = v_global * float(ny_tex)
            u_tile = (x_pix - float(x0)) / float(tw)
            v_tile = (y_pix - float(y0)) / float(th)
            vbo_data = np.stack([u_tile, v_tile, scales], axis=1).astype(np.float32)
            verts_per_triangle = 3
            verts_per_batch = int(batch_triangles * verts_per_triangle)
            fbo_w = int(tw * ss)
            fbo_h = int(th * ss)
            fbo_w = max(1, fbo_w)
            fbo_h = max(1, fbo_h)

            try:
                fbo = ctx.simple_framebuffer((fbo_w, fbo_h), components=4, dtype='f4')
            except Exception as e:
                logger.error("创建 tile framebuffer 失败：tile(%d,%d) size=%dx%d, ss=%d, err=%s.", tx, ty, fbo_w, fbo_h, ss, e)
                raise
            fbo.use()
            # additive accumulate
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.ONE, moderngl.ONE)
            
            n_total_verts = vbo_data.shape[0]
            start = 0
            if start == 0:
                fbo.clear(0.0, 0.0, 0.0, 0.0)
            while start < n_total_verts:
                end = min(start + verts_per_batch, n_total_verts)
                sub = vbo_data[start:end]
                vbo = ctx.buffer(sub.tobytes())
                vao = ctx.vertex_array(prog, [(vbo, '2f 1f', 'in_pos', 'in_scale')])
                vao.render(mode=moderngl.TRIANGLES, vertices=len(sub))
                vbo.release()
                vao.release()
                start = end
            
            raw = fbo.read(components=4, dtype='f4')
            arr = np.frombuffer(raw, dtype=np.float32).copy()
            arr = arr.reshape((fbo_h, fbo_w, 4))
            
            if ss > 1:
                try:
                    # arr shape: (th*ss, tw*ss, 4) -> reshape (th, ss, tw, ss, 4) and sum over ss dims
                    arr = arr.reshape((th, ss, tw, ss, 4)).sum(axis=(1,3))
                except Exception:
                    arr_ds = np.zeros((th, tw, 4), dtype=np.float32)
                    for iy in range(th):
                        for ix in range(tw):
                            ys = slice(iy*ss, (iy+1)*ss)
                            xs = slice(ix*ss, (ix+1)*ss)
                            block = arr[ys, xs, :]
                            if block.size == 0:
                                arr_ds[iy, ix, :] = 0.0
                            else:
                                arr_ds[iy, ix, :] = block.sum(axis=(0,1))
                    arr = arr_ds
            
            R = arr[:, :, 0]   
            A = arr[:, :, 3]   
            eps = 1e-12
            valid_mask = (A > eps)
            result_tile = np.zeros((th, tw), dtype=np.float32)
            result_tile[valid_mask] = (R[valid_mask] / A[valid_mask]).astype(np.float32)
            
            AH, AW = result_tile.shape
            GH = pixel_mu_map[y0:y1, x0:x1].shape[0]
            GW = pixel_mu_map[y0:y1, x0:x1].shape[1]          
            copy_h = min(AH, GH)
            copy_w = min(AW, GW)
            pixel_mu_map[y0:y0+copy_h, x0:x0+copy_w] += result_tile[:copy_h, :copy_w]
            
            fbo.release()

    logger.info("分块渲染完成: 输出尺寸 %d x %d", nx_tex, ny_tex)
    return pixel_mu_map
