/////////////////////////////////////////////////////////////////
//
// Kenta Suzue
//
// Geo-Mean.c
//
// Compile:  mpicc -g -Wall -lm Geo-Mean.c -o Geo-Mean
//
// Run:      mpiexec -n <p> ./Geo-Mean <N>
//
//           <p>: the number of processes
//           <N>: the number of elements in the vector
//
///////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <math.h>

void Error_Check(int argc, char** argv, int* len_vector, MPI_Comm mpi_comm, int comm_sz, int my_rank);
int *Random_Int_Vector_Maker(int len_vector);

int main(int argc, char** argv)
{
    int comm_sz, my_rank;
    int len_vector;
    int *random_int_vector = NULL;
    int int_per_process;
    int i;
    int product_per_process = 1;
    int product_all_processes;
    double geo_mean;
    double local_start;
    double local_finish;
    double local_elapsed;
    double global_elapsed;

    //Seed the random number generator to get different results each run
    //Differently seeded random number generator useful for random numbers in vector
    srand(time(NULL));

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    //Perform necessary error checking for inputs specified by users from command line.
    //Process 0 reads in the elements in the vector into the variable len_vector and broadcasts it to the other processes
    Error_Check(argc, argv, &len_vector, MPI_COMM_WORLD, comm_sz, my_rank);

    //printf("1The number of elements in the vector is %d from rank %d!\n", len_vector, my_rank);    

    //Generate a vector with random integer values according to its size (n) specified by users.
    //Generate random numbers for elements in the vector on process 0
    if (my_rank == 0)
    {
        random_int_vector = Random_Int_Vector_Maker(len_vector);

        //printf("0The random integers in the vector:\n");

        //for (i = 0; i < len_vector; i++)
            //printf("%d ", random_int_vector[i]);

        //printf("\n");
    }

    int_per_process = len_vector / comm_sz;

    // Allocate memory for each process to hold a part of the vector 
    int *random_int_vector_part = (int*) malloc(sizeof(int) * int_per_process);

    //Split the vector into some chunks according to the number of processes (p) with roughly equal size, then distribute chunks among processes. 
    //Scatter the random integer vector from process 0 to all processes 
    MPI_Scatter(random_int_vector, int_per_process, MPI_INT, random_int_vector_part, int_per_process, MPI_INT, 0, MPI_COMM_WORLD);
 
    //for (i = 0; i < int_per_process; i++)
        //printf("random number %d from process %d\n", random_int_vector_part[i], my_rank);

    //Conduct the partial product for the chunk of elements on each process.
    //For each process, find the product of all elements scattered to that process
    for (i = 0; i < int_per_process; i++)
        product_per_process *= random_int_vector_part[i];

    //printf("product_per_process is %d from process %d\n", product_per_process, my_rank);

    free(random_int_vector);
    free(random_int_vector_part);

    //Reduce the resulting product on process 0 (root process).
    //Find the product of all of the product_per_process variables from all processes
    MPI_Reduce(&product_per_process, &product_all_processes, 1, MPI_INT, MPI_PROD, 0, MPI_COMM_WORLD);

    //if (my_rank == 0)
        //printf("The final product is: %d.\n", product_all_processes);

    //Take timing for the part of the program that calculates the value of geometric mean.
    MPI_Barrier(MPI_COMM_WORLD);

    local_start = MPI_Wtime();

    //Calculate the geometric mean (Note: pow() function is very useful here) and print out the result.
    if (my_rank == 0)
    {
        geo_mean = pow((double) product_all_processes, 1.0/((double)len_vector));        
    }    

    local_finish = MPI_Wtime();
    local_elapsed = local_finish - local_start;
    //printf("Local elapsed time from process %d = %e seconds\n", my_rank, local_elapsed);

    MPI_Reduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        printf("The geometric mean is: %f.\n", geo_mean);
    }    

    if (my_rank == 0)
        printf("Global elapsed parallel exceution time for taking the geometric mean = %e seconds.\n", global_elapsed);

    MPI_Finalize();    

    return 0;
}

//Perform necessary error checking for inputs specified by users from command line.
//The function Error_Check uses process 0 to read in the number of elements in the vector from the command line argument.
//The function Error_Check also uses process 0 to perform error checking on the command line arguments.
//The number of arguments must be 2.
//The number of elemeents in the vector must be between 1 and 8, and evenly divisible by the number of processes 
void Error_Check(int argc, char** argv, int* len_vector, MPI_Comm mpi_comm, int comm_sz, int my_rank)
{
    if (my_rank == 0)
    {
        //Error checking that the number of arguments is 2. If not, print help for the format of the run command.
        if (argc != 2)
        {
            fprintf(stderr, "Error: The number of arguments is incorrect!\n");
            fprintf(stderr, "USAGE: mpiexec -n <number_of_processes> ./Geo-Mean <number_of_elements_in_the_vector>\n");

            *len_vector = -1;            
        }

        else
        {
            //Process 0 reads in the number of elements in the vector.
            //Store the argument for the number of elements in the vector into the int variable len_vector, 
            //with the atoi() function that converts a character string to an integer.  

            *len_vector = atoi(argv[1]);

            //Error checking that the number of elements in the vector is between 1 and 8.
            //If not, print help that the the number of elements in the vector should be between 1 and 8.
            if ((*len_vector < 1) || (*len_vector > 8))
            {
                fprintf(stderr, "Error: The number of elements in the vector should be between 1 and 8!\n");
                *len_vector = -1;            
            }

            //Error checking that the number of elements in the vector is 
            //evenly divisible by the number of processes
            //If not, print help that the the number of elements
            //in the vector should be evenly divisible by the number of processes.
            else if (*len_vector % comm_sz != 0)   
            {
                fprintf(stderr, "Error: The number of elements in the vector should be evenly divisible by the number of processes!\n");
                *len_vector = -1;
            }
        }
    }

    //Process 0 broadcasts len_vector to all of the processes
    MPI_Bcast(len_vector, 1, MPI_INT, 0, mpi_comm);

    if (*len_vector < 0)
    {
        MPI_Finalize();
        exit (0);
    }
}

//Generate a vector with random integer values according to its size (n) specified by users. 
//Create an array of random integers, such that the array has len_vector elements, 
//and such that each element of the array has a value in between 1 and len_vector, 
int *Random_Int_Vector_Maker(int len_vector)
{
    int *random_int_vector = (int*) malloc(sizeof(int) * len_vector);
    int i;

    for (i = 0; i < len_vector; i++)
        random_int_vector[i] = (rand() % len_vector) + 1;

    return random_int_vector;
}
