"""Matplotlib display and interaction; no numerical convolution lives here."""
import numpy as np

from .magnification_map import _as_map, _map_geometry


def MagMapShow(source, ps=None, *, frame=0, cut=0, img_center=(0.,0.),
               vmin=.5, vmax=None, sys_name='N-body Lens System', show=True):
    """Display arrays or lazy maps with physical bounds; return (figure, axes)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    mmp = _as_map(source, frame)
    array = mmp.load()
    geo = _map_geometry(mmp, array, ps, img_center)
    if not isinstance(cut,int) or cut < 0 or 2*cut >= min(array.shape):
        raise ValueError('cut must leave a nonempty map')
    shown = array[cut:array.shape[0]-cut,cut:array.shape[1]-cut]
    scale = geo['srcp_ps']
    extent = [geo['src_xmin']+cut*scale, geo['src_xmax']-cut*scale,
              geo['src_ymin']+cut*scale, geo['src_ymax']-cut*scale]
    lo = .5 if vmin is None else vmin
    hi = max(float(shown.max()),lo*1.01) if vmax is None else vmax
    if lo <= 0 or hi <= lo:
        raise ValueError('Logarithmic display requires 0 < vmin < vmax')
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(shown, origin='lower', extent=extent, cmap='magma',
                   aspect='equal', norm=LogNorm(vmin=lo,vmax=hi))
    ax.set(title=f'Magnification Map ({sys_name})', xlabel='βₓ (arcsec)', ylabel='βᵧ (arcsec)')
    fig.colorbar(im,ax=ax,label='Magnification')
    if show:
        plt.show()
    return fig, ax


class _LightCurveView:
    def __init__(self, extractor):
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm
        from matplotlib.patches import Circle
        from matplotlib.widgets import Slider
        self.extractor = extractor
        self.points = []
        self.start = self.end = None
        self.fig, (self.ax_map,self.ax_lc) = plt.subplots(1,2,figsize=(16,6))
        self.fig.subplots_adjust(bottom=.22,wspace=.3)
        mmp = extractor.series.frame(0) if extractor.series is not None else extractor.map
        array = mmp.load()
        geo = extractor.geometry
        extent = [geo['src_xmin'],geo['src_xmax'],geo['src_ymin'],geo['src_ymax']]
        # A bounded preview keeps notebook display memory independent of map
        # size. Convolution always reads the full-resolution original ROI.
        stride = max(1,int(np.ceil(max(array.shape)/2048)))
        preview = np.array(array[::stride,::stride],copy=True)
        del array
        im = self.ax_map.imshow(preview,origin='lower',extent=extent,cmap='magma',
                               norm=PowerNorm(gamma=extractor.gamma,vmin=.5,
                                              vmax=max(.51,float(preview.max()))))
        self.fig.colorbar(im,ax=self.ax_map,label='Magnification')
        self.ax_map.set(title=extractor.title_mmp,xlabel='βₓ (arcsec)',ylabel='βᵧ (arcsec)')
        self.ax_lc.set(title=extractor.lc_title,xlabel='Time (Days)',ylabel='Magnification')
        self.ax_lc.grid(alpha=.2)
        self.circles = [Circle((0,0),5*geo['srcp_ps'],color=color,alpha=.4,visible=False)
                        for color in ['#ff4000','#0004ff']]
        for circle in self.circles:
            self.ax_map.add_patch(circle)
        self.path, = self.ax_map.plot([],[],'--',color='#00aacc')
        self.curve, = self.ax_lc.plot([],[],color='#782281')
        slider_ax = self.fig.add_axes([.2,.08,.55,.035])
        self.slider = Slider(slider_ax,'Source radius (px)',1,100,valinit=5,valstep=1)
        self.angular_label = self.fig.text(.2,.025,'')
        self.slider.on_changed(self._update)
        self.fig.canvas.mpl_connect('button_press_event',self._on_click)
        self._update()

    def _on_click(self,event):
        if event.inaxes is not self.ax_map or event.button != 1:
            return
        if self.fig.canvas.toolbar is not None and self.fig.canvas.toolbar.mode:
            return
        self.points.append((event.xdata,event.ydata))
        if len(self.points) == 1:
            self.start = self.points[0]
            self.end = None
            self.circles[0].center = self.start
            self.circles[0].set_visible(True)
            self.circles[1].set_visible(False)
            self.path.set_data([],[])
        else:
            self.end = self.points[1]
            self.circles[1].center = self.end
            self.circles[1].set_visible(True)
            self.path.set_data([self.start[0],self.end[0]],[self.start[1],self.end[1]])
            self.points.clear()
            self._update()
        self.fig.canvas.draw_idle()

    def _update(self,value=None):
        radius = self.slider.val
        angular_radius = radius*self.extractor.geometry['srcp_ps']
        for circle in self.circles:
            circle.set_radius(angular_radius)
        self.angular_label.set_text(f'Angular radius: {angular_radius*1000:.3g} mas')
        if self.start is not None and self.end is not None:
            result = self.extractor.extract(self.start,self.end,source_radius_px=radius)
            self.curve.set_data(result.time_days,result.magnification)
            self.ax_lc.relim(); self.ax_lc.autoscale_view()
        self.fig.canvas.draw_idle()

    def show(self):
        import matplotlib.pyplot as plt
        plt.show()
        return self.fig, (self.ax_map,self.ax_lc)
