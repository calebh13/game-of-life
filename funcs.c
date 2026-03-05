#include "funcs.h"

#define BIGPRIME 2147483647UL

void GenerateInitialGoL(int n, int rows, int** local_grid, MPI_Comm comm) {
    int rank, p;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    unsigned int seed;

    if(rank == 0) {
        unsigned int *seeds = malloc(p * sizeof(unsigned int));
        srand(12345);
        for(int i = 0; i < p; i++) {
            seeds[i] = rand() % BIGPRIME + 1;
        }

        MPI_Scatter(seeds, 1, MPI_UNSIGNED, &seed, 1, MPI_UNSIGNED, 0, comm);
        free(seeds);
    } else {
        MPI_Scatter(NULL, 1, MPI_UNSIGNED, &seed, 1, MPI_UNSIGNED, 0, comm);
    }

    srand(seed);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < n; j++){
            local_grid[i][j] = rand() % 2 == 0;
        }
    }
}

void free_grid(int n, int rows, int** local_grid){
    for(int i = 0; i < rows; i++){
        free(local_grid[i]);
    }

    free(local_grid);
}

void simulate(int G, int n, int rows, int** local_grid, MPI_Comm comm){
    int** output_grid = malloc(rows * sizeof(int));
    for(int i = 0; i < rows; i++) {
        output_grid[i] = malloc(n * sizeof(int));
    }

    for(int cycle = 0; cycle < G; cycle++){
        // mpi sendrecv the respective rows
        for(int i = 1; i < rows - 1; i++){
            for(int j = 0; j < n; j++){
                output_grid[i][j] = determine_state(i,j,n,rows,local_grid, comm);
            }
        }
    }

    free_grid(n, rows, local_grid);
}

int determine_state(int x,int y,int n,int rows,int** local_grid, MPI_Comm comm){
    
}
