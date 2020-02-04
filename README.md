# Vector Addition in CUDA

## Introduction
This is a program which performs vector addition in CUDA, and also on the CPU.
It randomly generates numbers and populates the vectors, it can also print the
vectors, and find the residual vector.

## Compilation
Please make sure the CUDA toolkit is installed first. To Compile:

    nvcc vec_addition.cu -o COMPILED

## Usage
To use the program, there are a few command line arguments that need to be passed.
This usage example assumes that the executable name is COMPILED:

    ./COMPILED <Size_of_Vectors> <Number_of_Blocks> <Number_of_Threads> <Output_Mode>

* Size_of_Vectors   :  The number of items in each vector.
* Number_of_Blocks  :  The number of blocks used in the GPU.
* Number_of_Threads :  The number of threads used in each GPU block.
* Output_Mode       :  'q' for quiet, 'v' for verbose.
                       In quiet mode, only the timing results are shown.
                       In Verbose mode, the matrices and residual are also shown.
