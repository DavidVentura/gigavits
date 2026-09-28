# Builds the Cython MAS kernel in place: python gigatrain/vits/monotonic_align/setup.py build_ext --inplace
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import setup

_DIR = Path(__file__).parent

setup(
    name="monotonic_align",
    ext_modules=cythonize(str(_DIR / "core.pyx"), language_level=3),
    include_dirs=[numpy.get_include()],
    script_args=["build_ext", "--build-lib", str(_DIR.parent.parent.parent), "--build-temp", str(_DIR / "build")],
)
