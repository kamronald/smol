# Copyright (c)

"""
smol -- Statistical Mechanics On Lattices

Lighthweight but caffeinated Python implementations of computational methods
for statistical mechanical calculations of configurational states in
crystalline material systems.


smol is a minimal implementation of computational methods to calculate statistical
mechanical and thermodynamic properties of crystalline material systems based on
the cluster expansion method from alloy theory and related methods. Although smol
is intentionally lightweight---in terms of dependencies and built-in functionality
---it has a modular design that closely follows underlying mathematical formalism
and provides useful abstractions to easily extend existing methods or implement and
test new ones. Finally, although conceived mainly for method development, smol can
(and is being) used in production for materials science research applications.
"""

import os
import shutil
import sys

# get numpy to include headers
import numpy
from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext

from smol.utils._build import check_openmp_support, get_openmp_flag

COMPILE_OPTIONS = {
    "msvc": [
        "/O2",
        "/EHsc",
    ],
    "mingw32": ["-Wall", "-Wextra"],
    "other": ["-Wall", "-Wextra", "-ffast-math", "-O3"],
}

LINK_OPTIONS = {
    "msvc": ["-Wl, --allow-multiple-definition"],
    "mingw32": [["-Wl, --allow-multiple-definition"]],
    "other": [],
}

COMPILER_DIRECTIVES = {
    "language_level": 3,
}

if sys.platform.startswith("darwin"):
    COMPILE_OPTIONS["other"] += ["-mcpu=native", "-stdlib=libc++"]


# custom clean command to remove .c files
# taken from sklearn setup.py
class CleanCommand(Command):
    description = "Remove build artifacts from the source tree"

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, "PKG-INFO"))
        if remove_c_files:
            print("Will remove generated .c files")
        if os.path.exists("build"):
            shutil.rmtree("build")
        for dirpath, dirnames, filenames in os.walk("smol"):
            for filename in filenames:
                root, extension = os.path.splitext(filename)

                if extension in [".so", ".pyd", ".dll", ".pyc"]:
                    os.unlink(os.path.join(dirpath, filename))

                if remove_c_files and extension in [".c"]:
                    pyx_file = str.replace(filename, extension, ".pyx")
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))

            for dirname in dirnames:
                if dirname == "__pycache__":
                    shutil.rmtree(os.path.join(dirpath, dirname))


# By subclassing build_extensions we have the actual compiler that will be used which is
# really known only after finalize_options
# http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used  # noqa
class build_ext_options:
    def build_options(self):
        OMP_SUPPORTED = check_openmp_support(self.compiler)
        for e in self.extensions:
            e.extra_compile_args += COMPILE_OPTIONS.get(
                self.compiler.compiler_type, COMPILE_OPTIONS["other"]
            )

            e.extra_link_args += LINK_OPTIONS.get(
                self.compiler.compiler_type, LINK_OPTIONS["other"]
            )
            if OMP_SUPPORTED:
                omp_flag = get_openmp_flag(self.compiler)
                e.extra_compile_args += omp_flag
                e.extra_link_args += omp_flag


class build_ext_subclass(build_ext, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        build_ext.build_extensions(self)


# Compile option for cython extensions
if "--use-cython" in sys.argv:
    USE_CYTHON = True
    cython_kwargs = {}
    sys.argv.remove("--use-cython")
    if "--annotate-cython" in sys.argv:
        cython_kwargs["annotate"] = True
        sys.argv.remove("--annotate-cython")
else:
    USE_CYTHON = False

ext = ".pyx" if USE_CYTHON else ".c"
ext_modules = [
    Extension(
        "smol.utils.cluster.evaluator",
        ["smol/utils/cluster/evaluator" + ext],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c",
    ),
    Extension(
        "smol.utils.cluster.ewald",
        ["smol/utils/cluster/ewald" + ext],
        language="c",
    ),
    Extension(
        "smol.utils.cluster.container",
        ["smol/utils/cluster/container" + ext],
        language="c",
    ),
    Extension(
        "smol.utils.cluster.correlations",
        ["smol/utils/cluster/correlations" + ext],
        language="c",
    ),
]

if USE_CYTHON:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        ext_modules,
        include_path=[numpy.get_include()],
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "nonecheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
        **cython_kwargs
    )

cmdclass = {
    "clean": CleanCommand,
    "build_ext": build_ext_subclass,
}

setup(
    use_scm_version={"version_scheme": "python-simplified-semver"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_dirs=numpy.get_include(),
)
