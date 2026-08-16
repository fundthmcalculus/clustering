from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import numpy

# Compiler-specific optimization flags.
#
# Both platforms target a conservative x86-64 baseline with AVX2 support (available
# since ~2013) instead of -march=native, which would produce SIGILL crashes on CPUs
# lacking the build machine's instruction set (e.g., AVX-512). MSVC and Unix both
# emit AVX2 with various other baseline features for broad portability.
COMPILE_ARGS = {
    "msvc": [
        "/O2",  # max speed (MSVC has no /O3)
        "/arch:AVX2",  # emit AVX2 + FMA SIMD — matches Unix x86-64-v3 baseline
        "/fp:fast",  # relaxed FP, lets the vectorizer fuse/reorder (~ -ffast-math)
        "/openmp",  # OpenMP (links VCOMP140)
    ],
    "unix": [
        "-O3",
        "-march=x86-64-v3",
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
