# DCLens: A Multi-plane Lens Model GPU Geometric Inverse Ray-Shooting Microlensing Simulator

Welcome to the current release of the **discrete N-body** lens system microlensing simulator for the research of Dark Clusters. This version fully supports **geometric optics** calculations.

> **Note:** The **wave optics** module is actively in development and will be included in a future release.

## Table of Contents

- [Dependencies](#dependencies)
- [Installation & Compilation](#installation--compilation)
- [Usage](#usage)
  - [Lens System Generation](#lens-system-generation)
  - [Magnification Map Simulation](#magnification-map-simulation)
  - [Interactive Light Curve Extractor](#interactive-light-curve-extractor)
- [Citation](#citation)

## Dependencies

- **CUDA Toolkit**: Users must have CUDA installed and properly configured on their local machine or server to utilize the GPU-accelerated features of this pipeline.
- **C++ Compiler**: `gcc` / `g++` compatible with your CUDA version.
- **CMake & nvcc**: Required for building the core library.

## Installation & Compilation

The pipeline relies on a CUDA source file (`.cu`) which generates a shared library core (`.so`) used by the Python code. You must compile the `.cu` file before running the scripts.

Run the following commands to configure and build the project:

```shell
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Usage

### Lens System Generation

Please note that this pipeline **does not** include a built-in cluster or lens system generator. Users must pre-generate their lens system configurations and provide them as a standard text file (`.txt`).

The configuration file must strictly follow this data structure (space or tab-separated):

```txt
M[M_sun]    x[pc]    y[pc]    z[pc]
```

### Magnification Map Simulation

The **MMpS (Magnification Map Simulator)** module uses the GIRS (Geometric Inverse Ray-Shooting) scheme, GPU parallelization, and quad-tree refinement to simulate the magnification map on the source plane.

The `MMpS.MagMapSim` function is used for the simulation and accepts the following parameters:

- `ini_con_file`: The path to the system configuration file.
- `D_L_pc`: The distance between the observer and the lens system (in pc).
- `D_S_pc`: The distance between the observer and the source (in pc).
- `N_planes`: The number of planes used to split the lens system.
- `coarse_ps`: The pixel size for the rough sample image plane.
- `finest_ps`: The pixel size for the finest sample image plane.
- `srcp_ps`: The pixel size in the source plane.
- `max_refine_iters`: The maximum steps/depth of quad-tree refinement.
- `img_center`: The center of the projection plane for the considered region.
- `src_considered_region`: The width and height of the considered projection region.

```python
from DCLens.MMpS import MagMapSim as mmpsim
import gc

gc.collect()

MagMap = mmpsim(
    ini_con_file='DATA/DarkCluster.txt',
    save_dir='DATA/MagMap',
    D_L_pc=1e4, D_S_pc=1e8,
    coarse_ps=1e-3, finest_ps=1e-4, srcp_ps=1e-4,
    src_considered_region=(0.1,0.1),
)

gc.collect()
```

To visualize your simulated magnification map, use the `MMpS.MagMapShow` function with the following parameters:

- `mmp_file`: The path to the saved magnification map `.npy` file.
- `ps`: The pixel size in the source plane.
- `gamma`: The exponential distribution parameter to increase the contrast of the colorbar.

```python
from DCLens.MMpS import MagMapShow as mmpshow
%matplotlib widget

MagMapShow(mmp_file='DATA/MagMap/mmp.npy', 
           ps=1e-4, 
           gamma=0.1,
           sys_name='N-body Lens System')
```

### Interactive Light Curve Extractor

The **IALCE** module allows you to interactively set the source trajectory path and extract the corresponding light curve.

```python
from DCLens.IALCE import IALightCurveExtractor as ialce
%matplotlib widget

mmp_file = 'DATA/MagMap/mmp.npy'

LightCurve = ialce(mmp_file, ps=1e-4, pmv_arcsec_yr=1e-3, gamma=0.1)#, title_mmp=' \n ', lc_title='')
```

> The above is a general explanation of DCLens. More details and a comprehensive `quickstarting.ipynb` notebook are coming soon.

## Citation

A formal paper describing this pipeline is currently in preparation and will be linked here upon publication. If you use this software in your research, we would greatly appreciate it if you could cite this repository as follows:

```
@software{DCLens2026,
  author = {Tang Tingfeng},
  title = {DCLens},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/Teamf547/DCLens](https://github.com/Teamf547/DCLens)}}
}
```