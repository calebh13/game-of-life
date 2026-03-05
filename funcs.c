#include "funcs.h"

#define BIGPRIME 2147483647UL
#define OUTFILE_FORMAT "p%dout.txt"

int rank, p;

void GenerateInitialGoL(int n, int rows, int** local_grid, MPI_Comm comm) {
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

void sendLowerRecvUpper(int* upper_row, int n, int rows, int** local_grid, MPI_Comm comm)
{
    MPI_Request req;

    MPI_Isend(
        local_grid[rows - 1],
        n,
        MPI_INT,
        (rank + 1) % p,
        0,
        comm,
        &req
    );

    MPI_Recv(
        upper_row,
        n,
        MPI_INT,
        (rank + p - 1) % p,
        MPI_ANY_TAG,
        comm,
        MPI_STATUS_IGNORE
    );

    MPI_Wait(&req, MPI_STATUS_IGNORE);;
}

void sendUpperRecvLower(int* lower_row, int n, int rows, int** local_grid, MPI_Comm comm)
{
    MPI_Request req;
    MPI_Isend(
        local_grid[0],
        n,
        MPI_INT,
        (rank + p - 1) % p,
        0,
        comm,
        &req
    );

    MPI_Recv(
        lower_row,
        n,
        MPI_INT,
        (rank + 1) % p,
        MPI_ANY_TAG,
        comm,
        MPI_STATUS_IGNORE
    );

    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

void simulate(int G, int n, int rows, int** local_grid, MPI_Comm comm){
    int** output_grid = malloc(rows * sizeof(int*));
    int display_frequency = 1;
    for(int i = 0; i < rows; i++) {
        output_grid[i] = malloc(n * sizeof(int));
    }
    int *upper_row = malloc(n * sizeof(int));
    int *lower_row = malloc(n * sizeof(int));

    for(int cycle = 0; cycle < G; cycle++){
        sendLowerRecvUpper(upper_row, n, rows, local_grid, comm);
        sendUpperRecvLower(lower_row, n, rows, local_grid, comm);

        for(int i = 1; i < rows - 1; i++){
            for(int j = 0; j < n; j++){
                output_grid[i][j] = determine_state(i,j,n,rows,local_grid, comm);
            }
        }
        for(int j = 0; j < n; j++){
            output_grid[0][j] = determine_state_manual(j,n,local_grid[1], upper_row, local_grid[0]);
            output_grid[rows-1][j] = determine_state_manual(j,n,lower_row, local_grid[rows-2], local_grid[rows-1]);
        }

        int** temp = local_grid;
        local_grid = output_grid;
        output_grid = temp;

        if(cycle % display_frequency == 0){
            display_gol(n, rows, local_grid, comm);
        }

        MPI_Barrier(comm);
    }

    free(upper_row);
    free(lower_row);
    free_grid(n, rows, output_grid);
}

int determine_state_manual(int x, int n, int* lower, int* middle, int* upper){
    int west = (x - 1 + n) % n;
    int east = (x + 1) % n;

    int north = upper[x];
    int south = lower[x];
    int w     = middle[west];
    int e     = middle[east];

    int nw = upper[west];
    int ne = upper[east];
    int sw = lower[west];
    int se = lower[east];

    int neighbors = north + south + e + w + ne + nw + se + sw;

    return neighbors > 2 && neighbors < 6;
}

int determine_state(int y, int x, int n, int rows, int** local_grid, MPI_Comm comm) {

    int west = (x - 1 + n) % n;
    int east = (x + 1) % n;

    int north = local_grid[y - 1][x];
    int south = local_grid[y + 1][x];
    int w     = local_grid[y][west];
    int e     = local_grid[y][east];

    int nw = local_grid[y - 1][west];
    int ne = local_grid[y - 1][east];
    int sw = local_grid[y + 1][west];
    int se = local_grid[y + 1][east];

    int neighbors = north + south + e + w + ne + nw + se + sw;

    return neighbors > 2 && neighbors < 6;
}

void display_gol(int n, int rows, int** local_grid, MPI_Comm comm){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    char filename[100];
    sprintf(filename, OUTFILE_FORMAT, rank);
    FILE* outfile = fopen(filename, "w");
    for (int r = 0; r < size; r++) {
        MPI_Barrier(comm);

        if(rank == r){
            for(int i = 0; i < rows; i++){
                for(int j = 0; j < n; j++){
                    fprintf(outfile, "%d ", local_grid[i][j]);
                }
                fprintf(outfile, "\n");
            }

            fflush(stdout);
        }
    }
    fclose(outfile);

    MPI_Barrier(comm);

    // p0 prints all the stuff in order
    if (rank == 0) {
        char* line = malloc((n+1) * 2 * sizeof(char));
        for(int r = 0; r < p; r++) {
            sprintf(filename, OUTFILE_FORMAT, r);
            FILE* fp = fopen(filename, "r");
            size_t linesize = (n+1) * 2 * sizeof(char);
            printf("Reading process %d\n", r);
            while (getline(&line, &linesize, fp) != -1) {
                printf("%s", line); // getline does not remove newline chars
            }
            printf("Done reading process %d\n", r);
            fclose(fp);
        }
        free(line);
    }
}