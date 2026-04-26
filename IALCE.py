import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import ipywidgets as widgets
from IPython.display import display

from matplotlib.ticker import ScalarFormatter, MaxNLocator, FuncFormatter

class IALightCurveExtractor:
    def __init__(self, mmp_file, ps, title_mmp="Magnification Map", lc_title="Light Curve", cut=0, pmv_arcsec_yr=0.005, gamma=0.1):
        mag_map0 = np.load(mmp_file)
        mag_map = mag_map0[cut:mag_map0.shape[0]-cut, cut:mag_map0.shape[1]-cut]
        self.mag_map = mag_map
        self.h, self.w = mag_map.shape
        self.xmin, self.xmax, self.ymin, self.ymax = -self.w*ps/2, self.w*ps/2, -self.h*ps/2, self.h*ps/2
        self.title_mmp = title_mmp
        self.lc_title = lc_title

        self.gamma = gamma
        
        # source moving speed from arcsec unit to pixel unit
        self.dx = ps
        self.pmv = pmv_arcsec_yr # arcsec / year
        # time per pixel
        # dt = (arcsec/px) / (arcsec/yr) = yr/px
        self.dt_per_px = abs(self.dx) / self.pmv 
        # transform to days
        self.dt_days = self.dt_per_px * 365.25
        
        # initial source size (pixel unit)
        self.src_radius_px = 5
        self.update_source_kernel()
        
        # Interactive
        self.click_points = []

        self.setup_plot()
        self.setup_widgets()
        
    def update_source_kernel(self):
        r = int(self.src_radius_px)
        if r < 1: r = 1
        sigma = r / 2.0
        y, x = np.ogrid[-r:r+1, -r:r+1]
        kernel = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        self.kernel = kernel / kernel.sum()
        self.kernel_r = r

    def world_to_pixel(self, x_world, y_world):
        col = int((x_world - self.xmin) / (self.xmax - self.xmin) * self.w)
        row = int((y_world - self.ymin) / (self.ymax - self.ymin) * self.h)
        return row, col

    def get_light_curve(self, p1, p2):
        # path length in pixel units
        r1, c1 = self.world_to_pixel(p1[0], p1[1])
        r2, c2 = self.world_to_pixel(p2[0], p2[1])
        pixel_dist = np.sqrt((r2-r1)**2 + (c2-c1)**2)
        num_points = int(max(pixel_dist, 10))
        
        # generate time axis
        total_time = pixel_dist * self.dt_days
        time_axis = np.linspace(0, total_time, num_points)
        
        # space path in world coordinates
        path_x = np.linspace(p1[0], p2[0], num_points)
        path_y = np.linspace(p1[1], p2[1], num_points)
        
        mags = []
        r = self.kernel_r
        for px, py in zip(path_x, path_y):
            row, col = self.world_to_pixel(px, py)
            r_min, r_max = row - r, row + r + 1
            c_min, c_max = col - r, col + r + 1
            
            if r_min < 0 or r_max > self.h or c_min < 0 or c_max > self.w:
                mags.append(1.0)
                continue
            
            roi = self.mag_map[r_min:r_max, c_min:c_max]
            if roi.shape == self.kernel.shape:
                mags.append(np.sum(roi * self.kernel))
            else:
                mags.append(1.0)
        
        return time_axis, np.array(mags)

    def setup_plot(self):
        self.fig, (self.ax_map, self.ax_lc) = plt.subplots(1, 2, figsize=(16, 5),gridspec_kw={"wspace": 0.3})
        plt.subplots_adjust(bottom=0.25)
        
        # --- leftplane：Magnification Map ---
        # norm = PowerNorm(gamma=self.gamma, vmin=self.mag_map.min(), vmax=self.mag_map.max())
        norm = PowerNorm(gamma=self.gamma, vmin=0.5, vmax=2100)
        self.im = self.ax_map.imshow(self.mag_map, origin='lower', cmap='magma', 
                                     extent=[self.xmin, self.xmax, self.ymin, self.ymax], 
                                     aspect='equal', norm=norm)
        cbar = self.fig.colorbar(self.im, ax=self.ax_map, label='Magnification')

        # self.formatter = ScalarFormatter(useMathText=True)
        # self.formatter.set_powerlimits((-2, 2))
        # cbar.ax.yaxis.set_major_formatter(self.formatter)
        # cbar.ax.yaxis.set_major_locator(MaxNLocator(4))

        # color bar ticks and labels
        custom_ticks = [0.5, 1, 10, 100, 2000]
        cbar.set_ticks(custom_ticks)

        def custom_fmt(x, pos):
            if x >= 100:
                return f"${x/(10**int(np.log10(x))):g} \\times 10^{{{int(np.log10(x))}}}$"
            else:
                return f"${x:g}$"  # 使用 LaTeX，保持格式一致

        cbar.ax.xaxis.set_major_formatter(FuncFormatter(custom_fmt))

        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_horizontalalignment('left')

        offset_text.set_x(1.1)
        

        curr_radius_world = self.src_radius_px * abs(self.dx)
        self.circ_start = patches.Circle((0,0), curr_radius_world, color="#ff4000", 
                                        alpha=0.4, visible=False) 
        self.circ_end = patches.Circle((0,0), curr_radius_world, color="#0004ff", 
                                    alpha=0.4, visible=False)
        self.ax_map.set_title(self.title_mmp)
        self.ax_map.set_xlim(-0.01, 0.025)
        self.ax_map.set_ylim(-0.025, 0.01)
        self.ax_map.set_xlabel("$\\beta_x$ (arsec)")
        self.ax_map.set_ylabel("$\\beta_y$ (arsec)")
        self.ax_map.add_patch(self.circ_start)
        self.ax_map.add_patch(self.circ_end)

        self.start_leg, = self.ax_map.plot([], [], 'o', color="#ff4000", ms=5, label="Start Point")
        self.end_leg,   = self.ax_map.plot([], [], 'o', color="#0004ff", ms=5, label="End Point")
        
        self.path_line, = self.ax_map.plot([], [], 'k--', lw=1, alpha=0.6,  color="#00F2FF", label='Path')


        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.ax_map.legend(loc='upper right', fontsize='small', framealpha=0.5)

        # --- rightplane：LC ---
        self.lc_line, = self.ax_lc.plot([], [], 'r-', lw=1.5, color='#782281')
        self.ax_lc.set_title(self.lc_title)
        self.ax_lc.set_xlabel("Time (Days)")
        self.ax_lc.set_ylabel(f"$\\mathcal{{M}}$(Magnification of total flux)")
        self.ax_lc.grid(True, alpha=0.2)

    def on_click(self, event):
        if event.inaxes != self.ax_map or event.button != 1: return
        
        self.click_points.append((event.xdata, event.ydata))
        
        if len(self.click_points) == 1:
            self.circ_start.set_center(self.click_points[0])
            self.circ_start.set_visible(True)
            self.circ_end.set_visible(False)
            self.path_line.set_visible(False)
        elif len(self.click_points) == 2:
            self.circ_end.set_center(self.click_points[1])
            self.circ_end.set_visible(True)
            p1, p2 = self.click_points[0], self.click_points[1]
            self.path_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            self.path_line.set_visible(True)
            self.update_lc()
            self.click_points = []
        self.fig.canvas.draw_idle()

    def update_lc(self, change=None):
        if not self.path_line.get_visible(): return
        
        self.src_radius_px = self.size_slider.value
        self.update_source_kernel()
        
        world_r = self.src_radius_px * abs(self.dx)
        self.circ_start.set_radius(world_r)
        self.circ_end.set_radius(world_r)
        
        p1 = self.circ_start.center
        p2 = self.circ_end.center
        t, mags = self.get_light_curve(p1, p2)
        
        self.lc_line.set_data(t, mags)
        self.ax_lc.relim()
        self.ax_lc.autoscale_view()
        self.ang_label.value = f"Current Angular Radius: {world_r*1000:.2f} mas"
        self.fig.canvas.draw_idle()

    def setup_widgets(self):
        self.size_slider = widgets.FloatSlider(
            value=5, min=1, max=100, step=1, description='Src Radius (px):'
        )
        self.ang_label = widgets.Label(value="Angular Radius: -- mas")
        self.size_slider.observe(self.update_lc, names='value')
        display(widgets.VBox([self.size_slider, self.ang_label]))