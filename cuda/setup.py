from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='sigmoid_cuda_linear_cpp',
    ext_modules=[
        CUDAExtension('sigmoid_cuda', [
            'sigmoid_cuda.cpp',
            'sigmoid_cuda_kernel.cu',
        ]),
        CppExtension('linear_cpp', ['linear.cpp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
