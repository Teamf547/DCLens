"""PEP 517/660 installation adapter; CMake owns all CUDA build settings."""
import os
import shlex
from pathlib import Path
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        source = Path(__file__).resolve().parent
        destination = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build = Path(self.build_temp).resolve() / ext.name
        build.mkdir(parents=True, exist_ok=True)
        arguments = [
            "cmake", "-S", str(source), "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={destination}",
        ]
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            arguments += [f"-DCUDAToolkit_ROOT={cuda_home}",
                          f"-DCMAKE_CUDA_COMPILER={Path(cuda_home) / 'bin/nvcc'}"]
        # Forward portable build settings without changing numerical kernels.
        architectures = os.environ.get("CMAKE_CUDA_ARCHITECTURES")
        if architectures:
            arguments.append(f"-DCMAKE_CUDA_ARCHITECTURES={architectures}")
        arguments.extend(shlex.split(os.environ.get("CMAKE_ARGS", "")))
        subprocess.run(arguments, check=True)
        subprocess.run(
            ["cmake", "--build", str(build), "--parallel",
             os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "2")],
            check=True,
        )


setup(ext_modules=[Extension("DCLens._cuda", sources=[])],
      cmdclass={"build_ext": CMakeBuild})
