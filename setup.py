from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "tribbleclustering.pcvat",
        ["src/tribbleclustering/pcvat.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-march=native"],
    )
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)
