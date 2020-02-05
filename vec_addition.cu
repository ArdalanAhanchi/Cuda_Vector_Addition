#include <string>                                 //For stoi.
#include <iostream>                               //For stdout.
#include <cstdlib>                                //For random number generator.
#include <chrono>                                 //For getting time.

#include "cuda_runtime.h"                         //For Windows support.
#include "device_launch_parameters.h"

//The type that is used for the calculations.
typedef int type;

//Define constants for min/max.
#define RANDOMIZE_MIN -10
#define RANDOMIZE_MAX 10

//Cuda calculator which will run in each thread.
__global__ void cuda_calculator(type* a, type* b, type* c)
{
    //Calculate the index.
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    //Add the vectors in the current thread index.
    c[index] = a[index] + b[index];
}

//Cuda addition which runs the cuda program.
double cuda_addition(type* a, type* b, type* c, int n, int blocks, int threads)
{
    //Create pointers for the GPU memory allocation
    type* cu_vec_a;
    type* cu_vec_b;
    type* cu_vec_c;

    //Allocate memory on the device for the arrays.
    cudaMalloc((void**) &cu_vec_a, sizeof(type) * n);
    cudaMalloc((void**) &cu_vec_b, sizeof(type) * n);
    cudaMalloc((void**) &cu_vec_c, sizeof(type) * n);

    //Capture the beginning time before the data transfer and calculations.
    auto begin = std::chrono::high_resolution_clock::now();

    //Copy the data from the main memory to Vram.
    cudaMemcpy(cu_vec_a, a, sizeof(type) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_vec_b, b, sizeof(type) * n, cudaMemcpyHostToDevice);

    //Launch the addition kernel on the device.
    cuda_calculator<<<blocks, threads>>>(cu_vec_a, cu_vec_b, cu_vec_c);

    //Copy the results back from Vram to main ram.
    cudaMemcpy(c, cu_vec_c, sizeof(type) * n, cudaMemcpyDeviceToHost);

    //Wait for the thread to finish execution.
    cudaDeviceSynchronize();

    //Calculate the total time in seconds that it took to compute.
    auto total = std::chrono::high_resolution_clock::now() - begin;

    //Deallocate memory in the GPU.
    cudaFree(cu_vec_a);
    cudaFree(cu_vec_b);
    cudaFree(cu_vec_c);

    //Return the total time in seconds that it took to compute.
    return std::chrono::duration<double> (total).count();;
}

//Sequential addition function.
double seq_addition(type* a, type* b, type* c, int size)
{
    //Capture the beginning time before the calculations.
    auto begin = std::chrono::high_resolution_clock::now();

    //Iterate over the vectors and add the elements.
    for(int i = 0; i < size; i++)
        c[i] = a[i] + b[i];

    //Calculate and return the total time in seconds that it took to compute.
    auto total = std::chrono::high_resolution_clock::now() - begin;
    return std::chrono::duration<double> (total).count();;
}

//Sequential subtraction function (used for residual matrix).
void seq_subtraction(type* a, type* b, type* c, int size)
{
    //Iterate over the vectors and subtract the elements.
    for(int i = 0; i < size; i++)
        c[i] = a[i] - b[i];
}

//A function which randomizes the vector, by defualt it only uses values between -10 - 10
void randomize(type* vec, int size, int min = RANDOMIZE_MIN, int max = RANDOMIZE_MAX)
{
    //Perform this to ensure the random number generation is truly random.
    std::srand(std::chrono::system_clock::now().time_since_epoch().count());

    //Iterate through, and generate random numbers for each index.
    for(int i = 0; i < size; i++)
        vec[i] = ((type) std::rand() %
            (type) (RANDOMIZE_MAX * 2) + (type) RANDOMIZE_MIN) % RANDOMIZE_MAX ;
}

//Print the given vector to stdout.
void dump(type* vec, int size)
{
    //Iterate through, and generate random numbers for each index.
    for(int i = 0; i < size - 1; i++)
        std::cout << std::scientific << vec[i] <<  " | " ;

    //Print the last item in a different format and add a new line.
    std::cout << std::scientific << vec[size - 1] << std::endl;
}

//A function which will be called when there is an error.
int error(std::string msg)
{
    //Print the error message.
    std::cout << msg << std::endl;

    //Print the usage message.
    std::cout << std::endl << "Usage Guide:" << std::endl
        << "\t* ./a.out <Size of Vectors> <Number of Blocks> <Number of Threads>"
        << " <Output Mode>" << std::endl << "\t* Output mode is either \'q\' "
        << "(quiet) or \'v\' (verbose)" << std::endl
        << "\t* Number of blocks and threads are for the GPU." << std::endl;

    //Return exit failure for passing it back to the terminal.
    return EXIT_FAILURE;
}

//Main method which parses the arguments, and runs the program.
int main(int argc, char** argv)
{
    //Define values for parameters.
    int n, blocks, threads;
    bool verbose;

    //Check for invalid number of args.
    if(argc != 5)
        return error("Invalid number of arguments.");

    //Parse the arguments.
    try
    {
        n = std::stoi(argv[1]);
        blocks = std::stoi(argv[2]);
        threads = std::stoi(argv[3]);
    }
    catch(...)      //If we get here, there was an error in the arguments.
    {
        return error("Invalid arguments, could not parse.");
    }

    //Check the print mode.
    if(std::string(argv[4]) == "q" || std::string(argv[4]) == "v")
        //If the mode is valid and set to v, set verbose to true, false otherwise.
        verbose = (std::string(argv[4]) == "v" ? true : false);
    else
        //If we get here an invalid mode was passed.
        return error("Invalid print mode.");

    //Check for invalid threads / blocks / n sizes.
    if(n < 1 || blocks < 1 || threads < 1)
        return error("Invalid arguments. All parameters should be positive.");

    //Check for invalid relation between n and blocks/threads.
    if(n > (blocks * threads))
        return error("Invalid arguments. (Blocks * Threads) should be larger than the vector Size.");

    //Allocate memory for the input vectors.
    type* vec_a = new type[n];
    type* vec_b = new type[n];

    //Randomize the input vectors.
    randomize(vec_a, n);
    randomize(vec_b, n);

    //Allocate output matrices for the sequential and cuda executions.
    type* vec_c_seq = new type[n];
    type* vec_c_cuda = new type[n];

    //Perform the sequential addition.
    double seq_time = seq_addition(vec_a, vec_b, vec_c_seq, n);

    //Perform the cuda addition.
    double cuda_time = cuda_addition(vec_a, vec_b, vec_c_cuda, n, blocks, threads);

    //Print the timing results, and the input arguments.
    std::cout << "[Cuda_Time_Seconds]=" << cuda_time
        << "  [Sequential_Time_Seconds]=" << seq_time
        << "  [N]=" << n << "  [Blocks]=" << blocks
        << "  [Threads]=" << threads << std::endl;

    //Calculate residual vector for sequential implementation vs cuda.
    type* residual = new type[n];
    seq_subtraction(vec_c_seq, vec_c_cuda, residual, n);

    //Check if we're in verbose output mode.
    if(verbose)
    {
        //Print out the inputs, calculations and residual vector.
        std::cout << std::endl << "Printing out the First Vector:" << std::endl;
        dump(vec_a, n);

        std::cout << "\nPrinting out the Second Vector:" << std::endl;
        dump(vec_b, n);

        std::cout << "\nPrinting out the Addition results (Sequential):" << std::endl;
        dump(vec_c_seq, n);

        std::cout << "\nPrinting out the Addition results (Cuda):" << std::endl;
        dump(vec_c_cuda, n);

        std::cout << "\nPrinting out the residual matrix (Seq - Cuda):" << std::endl;
        dump(residual, n);
    }

    //Deallocate the memory in the heap.
    delete[] vec_a, vec_b, vec_c_seq, vec_c_cuda, residual;

    return EXIT_SUCCESS;
}
