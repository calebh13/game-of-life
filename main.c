#include "funcs.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    const int n = 16;
    int rows = n / p;

    int** local_grid = malloc(rows * sizeof(int*));

    for(int i = 0; i < rows; i++){
        local_grid[i] = malloc(n * sizeof(int));
    }

    GenerateInitialGoL(n, rows, local_grid, MPI_COMM_WORLD);
    simulate(2, n, rows, local_grid, MPI_COMM_WORLD);

    free_grid(n, rows, local_grid);
    MPI_Finalize();
    return 0;
}