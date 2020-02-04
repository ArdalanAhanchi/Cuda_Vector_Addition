# Vector Addition in CUDA

## Introduction
This is a program which performs vector addition in CUDA, and also on the CPU.
It randomly generates numbers and populates the vectors, it can also print the
vectors, and find the residual vector.

## Compilation
Please make sure the CUDA toolkit is installed first. To Compile:

*nvcc vec_addition.cu -o COMPILED*

## Usage
To use the program, there are a few command line arguments that need to be passed.
This usage example assumes that the executable name is COMPILED:

*./COMPILED <Size of Vectors> <Number of Blocks> <Number of Threads> <Output Mode>*

<Size of Vectors>   :   The number of items in each vector.
<Number of Blocks>  :   The number of blocks used in the GPU.
<Number of Threads> :   The number of threads used in each GPU block.
<Output Mode>       :   'q' for quiet, 'v' for verbose.
                        In quiet mode, only the timing results are shown.
                        In Verbose mode, the matrices and residual are also shown.
