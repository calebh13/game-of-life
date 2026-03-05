#include "funcs.h"
#include <assert.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc < 3) {
        printf("Usage: %s <n> <G>\n", argv[0]);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    const int n = (int)strtol(argv[1], NULL, 10);
    const int G = (int)strtol(argv[2], NULL, 10);

    int rows = n / p;
    if (rank == 0) printf("n = %d, G = %d, rows = %d\n", n, G, rows);
    assert(n > p);

    int** local_grid = malloc(rows * sizeof(int*));

    for(int i = 0; i < rows; i++){
        local_grid[i] = malloc(n * sizeof(int));
    }
    if (rank == 0) printf("Allocated local_grid successfully\n");

    GenerateInitialGoL(n, rows, local_grid, MPI_COMM_WORLD);
    simulate(G, n, rows, local_grid, MPI_COMM_WORLD);

    free_grid(n, rows, local_grid);
    if (rank == 0) printf("Freed grid successfully\n");
    MPI_Finalize();
    return 0;
}