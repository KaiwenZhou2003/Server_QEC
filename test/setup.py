from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

VERSION = "0.0.0"
f = open("our_bp_decoder/VERSION", "w+")
f.write(VERSION)
f.close()

extension1 = Extension(
    name="our_bp_decoder.mod2sparse",
    sources=[
        "our_bp_decoder/mod2sparse.pyx",
        "our_bp_decoder/include/mod2sparse.c",
        "our_bp_decoder/include/mod2sparse_extra.c",
        "our_bp_decoder/include/binary_char.c",
    ],  # path
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(), "our_bp_decoder/include"],
    extra_compile_args=["-std=c11"],
)

extension2 = Extension(
    name="our_bp_decoder.c_util",
    sources=["our_bp_decoder/c_util.pyx", "our_bp_decoder/include/mod2sparse.c"],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(), "our_bp_decoder/include"],
    extra_compile_args=["-std=c11"],
)

extension3 = Extension(
    name="our_bp_decoder.bp_decoder",
    sources=[
        "our_bp_decoder/bp_decoder.pyx",
        "our_bp_decoder/include/mod2sparse.c",
        "our_bp_decoder/include/mod2sparse_extra.c",
        "our_bp_decoder/include/sort.c",
    ],
    libraries=[],
    library_dirs=[],
    include_dirs=[numpy.get_include(), "our_bp_decoder/include"],
    extra_compile_args=["-std=c11"],
)

setup(
    version=VERSION,
    ext_modules=cythonize(
        [extension1, extension2, extension3],
        compiler_directives={"language_level": "3"},
    ),
)

# setup(
#     ext_modules=cythonize(
#         [
#             "bp_decoder.pyx",
#             "mod2sparse.pyx",
#             "c_util.pyx",
#         ],
#         # language_level=3,
#     ),
#     include_dirs=[numpy.get_include(), "./include"],
# )
