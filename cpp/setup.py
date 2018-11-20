from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='linear_cpp',
    ext_modules=[
        CppExtension('linear_cpp', ['linear.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
