from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import numpy

# Compiler-specific optimization flags.
#
# The previous flags (-O3 -march=native -fopenmp) are GCC/Clang syntax. MSVC —
# the default compiler for CPython on Windows — does not understand them and
# silently ignores each one, so Windows builds got no SIMD/AVX vectorization at
# all (baseline SSE2 only). Select flags by compiler at build time instead.
COMPILE_ARGS = {
    "msvc": [
        "/O2",  # max speed (MSVC has no /O3)
        "/arch:AVX2",  # emit AVX2 + FMA SIMD — the Windows equivalent of -march=native
        "/fp:fast",  # relaxed FP, lets the vectorizer fuse/reorder (~ -ffast-math)
        "/openmp",  # OpenMP (links VCOMP140)
    ],
    "unix": [
        "-O3",
        "-march=native",
        "-ffast-math",
        "-fopenmp",
    ],
}
LINK_ARGS = {
    "msvc": [],  # /openmp pulls in the runtime automatically
    "unix": ["-fopenmp"],
}


class build_ext_opts(build_ext):
    """Inject per-compiler optimization flags once the compiler is known."""

    def build_extensions(self):
        ctype = self.compiler.compiler_type  # 'msvc' or 'unix'
        cargs = COMPILE_ARGS.get(ctype, COMPILE_ARGS["unix"])
        largs = LINK_ARGS.get(ctype, LINK_ARGS["unix"])
        for ext in self.extensions:
            ext.extra_compile_args = cargs
            ext.extra_link_args = largs
        super().build_extensions()


extensions = [
    Extension(
        "tribbleclustering.pcvat",
        ["src/tribbleclustering/pcvat.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "tribbleclustering.cfcm",
        ["src/tribbleclustering/cfcm.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "tribbleclustering.clk",
        ["src/tribbleclustering/clk.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]


def _cythonize(exts):
    from Cython.Build import cythonize

    return cythonize(exts, language_level="3")


setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=_cythonize(extensions),
    cmdclass={"build_ext": build_ext_opts},
    zip_safe=False,
)
