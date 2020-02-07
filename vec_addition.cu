#include <string>                                 //For stoi.
#include <iostream>                               //For stdout.
#include <cstdlib>                                //For random number generator.
#include <chrono>                                 //For getting time.
#include <climits>                                //For maximum n.

#include "cuda_runtime.h"                         //For Windows support.
#include "device_launch_parameters.h"

//The type that is used for the calculations.
typedef int type;

//Define constants for min/max.
#define RANDOMIZE_MIN -10
#define RANDOMIZE_MAX 10

//Cuda calculator which will run in each thread.
__global__ void cuda_calculator(type* a, type* b, type* c, int* num_calcs)
{
    //Calculate the starting index.
    size_t start_index = (threadIdx.x + blockIdx.x * blockDim.x) * (*num_calcs);

    //Add the vectors in the current thread index.
    for(size_t i = 0; i < *num_calcs; i++)
        c[start_index + i] = a[start_index + i] + b[start_index + i];
}

//Cuda addition which runs the cuda program.
int cuda_addition(type* a, type* b, type* c, size_t n, size_t blocks,
    size_t threads, double times[3])
{
    //Create pointers for the GPU memory allocation
    type* cu_vec_a;
    type* cu_vec_b;
    type* cu_vec_c;
    int* cu_num_calcs;

    //Calculate the number of elements that this thread will take.
    size_t num_calcs = (n / (blocks * threads));

    //Check if it's not rounded, or it it was zero.
    if(n % (blocks * threads) != 0 || num_calcs <= 0)
        num_calcs++;

    //Allocate memory on the device for the arrays.
    cudaMalloc((void**) &cu_vec_a, sizeof(type) * n);
    cudaMalloc((void**) &cu_vec_b, sizeof(type) * n);
    cudaMalloc((void**) &cu_vec_c, sizeof(type) * n);
    cudaMalloc((void**) &cu_num_calcs, sizeof(int));

    //Wait for the thread to finish execution.
    cudaDeviceSynchronize();

    //Capture the beginning time before the data transfer (from host).
    auto begin_transfer_to = std::chrono::high_resolution_clock::now();

    //Copy the data, and the size from the main memory to VRAM.
    cudaMemcpy(cu_vec_a, a, sizeof(type) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_vec_b, b, sizeof(type) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_num_calcs, &num_calcs, sizeof(int), cudaMemcpyHostToDevice);

    //Wait for the thread to finish execution.
    cudaDeviceSynchronize();

    //Calculate the total time in seconds that it took to transfer data to the device
    auto total_transfer_to = std::chrono::high_resolution_clock::now() - begin_transfer_to;
    times[0] = std::chrono::duration<double> (total_transfer_to).count();

    //Capture the beginning time before the calculations.
    auto begin_calcs_only = std::chrono::high_resolution_clock::now();

    //Launch the addition kernel on the device.
    cuda_calculator<<<blocks, threads>>>(cu_vec_a, cu_vec_b, cu_vec_c, cu_num_calcs);

    //Check if we got any errors.
    if(cudaGetLastError() != cudaSuccess)
        return EXIT_FAILURE;

    //Wait for the thread to finish execution.
    cudaDeviceSynchronize();

    //Calculate the total time in seconds that it took to calculate.
    auto total_calcs_only = std::chrono::high_resolution_clock::now() - begin_calcs_only;
    times[1] = std::chrono::duration<double> (total_calcs_only).count();

    //Capture the beginning time before the calculations.
    auto begin_transfer_from = std::chrono::high_resolution_clock::now();

    //Copy the results back from Vram to main ram.
    cudaMemcpy(c, cu_vec_c, sizeof(type) * n, cudaMemcpyDeviceToHost);

    //Wait for the thread to finish execution.
    cudaDeviceSynchronize();

    //Calculate the total time in seconds that it took to transfer back to host.
    auto total_transfer_from = std::chrono::high_resolution_clock::now() - begin_transfer_from;
    times[2] = std::chrono::duration<double> (total_transfer_from).count();

    //Deallocate memory in the GPU.
    cudaFree(cu_vec_a);
    cudaFree(cu_vec_b);
    cudaFree(cu_vec_c);
    cudaFree(cu_num_calcs);

    //Wait for the thread to finish execution.
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}

//Sequential addition function.
double seq_addition(type* a, type* b, type* c, size_t size)
{
    //Capture the beginning time before the calculations.
    auto begin = std::chrono::high_resolution_clock::now();

    //Iterate over the vectors and add the elements.
    for(size_t i = 0; i < size; i++)
        c[i] = a[i] + b[i];

    //Calculate and return the total time in seconds that it took to compute.
    auto total = std::chrono::high_resolution_clock::now() - begin;
    return std::chrono::duration<double> (total).count();;
}

//Sequential subtraction function (used for residual matrix).
void seq_subtraction(type* a, type* b, type* c, size_t size)
{
    //Iterate over the vectors and subtract the elements.
    for(size_t i = 0; i < size; i++)
        c[i] = a[i] - b[i];
}

//A function which randomizes the vector, by defualt it only uses values between -10 - 10
void randomize(type* vec, int size, int min = RANDOMIZE_MIN, int max = RANDOMIZE_MAX)
{
    //Perform this to ensure the random number generation is truly random.
    std::srand(std::chrono::system_clock::now().time_since_epoch().count());

    //Iterate through, and generate random numbers for each index.
    for(size_t i = 0; i < size; i++)
        vec[i] = ((type) std::rand() %
            (type) (RANDOMIZE_MAX * 2) + (type) RANDOMIZE_MIN) % RANDOMIZE_MAX ;
}

//Print the given vector to stdout.
void dump(type* vec, size_t size)
{
    //Iterate through, and generate random numbers for each index.
    for(size_t i = 0; i < size - 1; i++)
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
    size_t n, blocks, threads;
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

    //Check if we're gonna get overflow.
    if(n > UINT_MAX)
        return error("Integer Overflow, please reduce N.");

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

    //Perform the cuda addition, and capture the timings.
    double times[3];
    int stat = cuda_addition(vec_a, vec_b, vec_c_cuda, n, blocks, threads, times);

    //Check the status.
    if(stat == EXIT_FAILURE)
        std::cout << "Error: Failed to execute kernel." << std::endl;

    //Print the timing results, and the input arguments.
    std::cout << "[Cuda_Transfer_To_Device_Seconds]=" << std::scientific << times[0]
        << "  [Cuda_Transfer_To_Host_Seconds]=" << std::scientific << times[2]
        << "  [Cuda_Calculation_Time_Seconds]=" << std::scientific << times[1]
        << "  [Sequential_Time_Seconds]=" << std::scientific << seq_time
        << "  [N]=" << n << "  [Blocks]=" << blocks
        << "  [Threads]=" << threads
        << std::endl;


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
