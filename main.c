#include "funcs.h"
#include <assert.h>

#define SEC_TO_US 1000000

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
    // if (rank == 0) printf("n = %d, G = %d, rows = %d\n", n, G, rows);
    assert(n > p && n % p == 0);

    int** local_grid = malloc(rows * sizeof(int*));

    for(int i = 0; i < rows; i++){
        local_grid[i] = malloc(n * sizeof(int));
    }

    GenerateInitialGoL(n, rows, local_grid, MPI_COMM_WORLD);
    double start = MPI_Wtime();
    simulate(G, n, rows, local_grid, MPI_COMM_WORLD);
    double end = MPI_Wtime();
    double max_comm_time;
    MPI_Allreduce(&comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) { 
        // printf("FORMAT:p,n,total runtime,avg time/gen,total comm time,total comp time\n");
        double total_runtime_us = (end - start) * SEC_TO_US;
        double avg_gen_time_us = (total_runtime_us / G);
        max_comm_time *= SEC_TO_US;
        double comp_time_us = (total_runtime_us - (comm_time * SEC_TO_US));
        printf("%d,%d,%.0f,%.0f,%.0f,%.0f\n", p, n, total_runtime_us, avg_gen_time_us, max_comm_time, comp_time_us);
    }
    // if (rank == 0) printf("TOTAL RUNTIME,%.7f\n", end - start);
    // if (rank == 0) printf("Avg time per generation,%.7f\n", (end - start) / G);
    // we need to find the MAXIMUM communication time
    // if (rank == 0) printf("Total communication time,%.7f\n", max_comm_time);
    // if (rank == 0) printf("Total computation time,%.7f\n", (end - start) - comm_time);

    free_grid(n, rows, local_grid);
    MPI_Finalize();
    return 0;
}