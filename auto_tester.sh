#! /bin/bash
 
#Compile the program just in case.
nvcc vec_addition.cu -o COMPILED

#Test 8 block values.
for blocks in 1 8 32 128 512 2048 8192 32768; do
    #Test 6 number of threads.
    for threads in 1 4 16 64 256 1024; do
        #Increase N in 15 increments.
        for size in 16 64 128 512 1024 2048 4096 32768 262144 1048576 2097152 4194304 8388608 16777216 33554432; do
            #Run the program with the right parameters.
            ./COMPILED $size $blocks $threads q
        done
    done
done

exit 0;
