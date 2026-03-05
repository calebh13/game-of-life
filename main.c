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

    if(rank == 0){
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < n; j++){
                printf("%d ", local_grid[i][j]);
            }
            printf("\n");
        }
    }

    free_grid(n, rows, local_grid);
    MPI_Finalize();
    return 0;
}