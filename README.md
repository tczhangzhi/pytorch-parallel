# Custom Dense layer in PyTorch

The project customizes the dense layer in pytorch following the official tutorial. In the project, we first write python code, and then gradually use C++, CUDA to optimize key operations. I hope this project will help your Pytorch, ATen, CUDA, PTX learning.

## How to run

### Python Extensions

Check the grad.

```
# ./
python grad_check.py py
```

### C++ Extensions

Pybind11 is used for Python and C++ interactions. Install these packages:

```
conda install pytest pybind11
```

Enter the C++ folder and compile the code.

```
# ./cpp
python setup.py install
```

Check the grad.

```
# ./
python grad_check.py cpp
```

### CUDA Extensions

Enter the CUDA folder and compile the code.

```
# ./cuda
python setup.py install
```

Check the grad.

```
# ./
python grad_check.py cuda
```

### PTX Example

Enter the PTX folder and compile the code.

```
# ./ptx
sh compile.sh
```

After changing the ` sigmoid_cuda_kernal.ptx` file, recompile your code.

```
# ./ptx
sh recompile.sh
```

Test your result.

```
./sigmoid_cuda_kernal
```

## License

[MIT](http://opensource.org/licenses/MIT)

Copyright (c) 2018-present, Zhi Zhang

