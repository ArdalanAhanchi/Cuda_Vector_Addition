#! /bin/bash
 
#Compile the program just in case.
nvcc vec_addition.cu -o COMPILED

#Test 16 block values.
for blocks in {1..32768..2184}; do
    #Test 16 number of threads.
    for threads in {1..1024..68}; do
        #Increase N in 16 increments.
        for size in {100..10000000..666660}; do
            #Run the program with the right parameters.
            ./COMPILED $size $blocks $threads q
        done
    done
done

exit 0;
