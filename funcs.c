#include "funcs.h"

#define BIGPRIME 2147483647UL
#define OUTFILE_FORMAT "p%dout"

int rank, p;
FILE* logfile;

#define LOG(...) \
do { \
    if(logfile){ \
        fprintf(logfile, "p%d: ", rank); \
        fprintf(logfile, __VA_ARGS__); \
        fprintf(logfile, "\n"); \
        fflush(logfile); \
    } \
} while(0)

void GenerateInitialGoL(int n, int rows, int** local_grid, MPI_Comm comm) 
{
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &p);

    char logname[64];
    sprintf(logname, "log_p%d.txt", rank);
    // sprintf(logname, "/dev/null");
    logfile = fopen(logname, "w");

    LOG("Entered GenerateInitialGoL (n=%d rows=%d)", n, rows);

    unsigned int seed;

    if(rank == 0) {
        LOG("Allocating seed array");
        unsigned int *seeds = malloc(p * sizeof(unsigned int));

        LOG("Generating seeds");
        srand(12345);

        for(int i = 0; i < p; i++) {
            seeds[i] = rand() % BIGPRIME + 1;
            LOG("Seed[%d]=%u", i, seeds[i]);
        }

        LOG("MPI_Scatter seeds");
        MPI_Scatter(seeds, 1, MPI_UNSIGNED, &seed, 1, MPI_UNSIGNED, 0, comm);

        free(seeds);
        LOG("Freed seed array");
    } 
    else {
        LOG("Waiting for MPI_Scatter seed");
        MPI_Scatter(NULL, 1, MPI_UNSIGNED, &seed, 1, MPI_UNSIGNED, 0, comm);
    }

    LOG("Received seed %u", seed);

    srand(seed);

    LOG("Generating local grid");
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < n; j++){
            local_grid[i][j] = rand() % 2 == 0;
        }
    }

    LOG("Exiting GenerateInitialGoL");
}

void free_grid(int n, int rows, int** local_grid)
{
    LOG("Entered free_grid rows=%d", rows);

    for(int i = 0; i < rows; i++){
        LOG("Freeing row %d", i);
        free(local_grid[i]);
    }

    free(local_grid);

    LOG("Exiting free_grid");
}

void sendLowerRecvUpper(int* upper_row, int n, int rows, int** local_grid, MPI_Comm comm)
{
    LOG("Entered sendLowerRecvUpper");

    MPI_Request req;

    int send_target = (rank + 1) % p;
    int recv_source = (rank + p - 1) % p;

    LOG("MPI_Isend lower row -> p%d", send_target);

    MPI_Isend(
        local_grid[rows - 1],
        n,
        MPI_INT,
        send_target,
        0,
        comm,
        &req
    );

    LOG("MPI_Recv upper row <- p%d", recv_source);

    MPI_Recv(
        upper_row,
        n,
        MPI_INT,
        recv_source,
        MPI_ANY_TAG,
        comm,
        MPI_STATUS_IGNORE
    );

    LOG("Waiting for Isend completion");
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    LOG("Exiting sendLowerRecvUpper");
}

void sendUpperRecvLower(int* lower_row, int n, int rows, int** local_grid, MPI_Comm comm)
{
    LOG("Entered sendUpperRecvLower");

    MPI_Request req;

    int send_target = (rank + p - 1) % p;
    int recv_source = (rank + 1) % p;

    LOG("MPI_Isend upper row -> p%d", send_target);

    MPI_Isend(
        local_grid[0],
        n,
        MPI_INT,
        send_target,
        0,
        comm,
        &req
    );

    LOG("MPI_Recv lower row <- p%d", recv_source);

    MPI_Recv(
        lower_row,
        n,
        MPI_INT,
        recv_source,
        MPI_ANY_TAG,
        comm,
        MPI_STATUS_IGNORE
    );

    LOG("Waiting for Isend completion");
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    LOG("Exiting sendUpperRecvLower");
}

void simulate(int G, int n, int rows, int** local_grid, MPI_Comm comm)
{
    LOG("Entered simulate G=%d n=%d rows=%d", G, n, rows);

    display_gol(n, rows, local_grid, comm);

    LOG("Allocating output grid");
    int** output_grid = malloc(rows * sizeof(int*));

    for(int i = 0; i < rows; i++) {
        output_grid[i] = malloc(n * sizeof(int));
    }

    LOG("Allocating halo rows");
    int *upper_row = malloc(n * sizeof(int));
    int *lower_row = malloc(n * sizeof(int));

    int display_frequency = 1;

    for(int cycle = 0; cycle < G; cycle++){

        LOG("Cycle %d start", cycle);

        sendLowerRecvUpper(upper_row, n, rows, local_grid, comm);
        sendUpperRecvLower(lower_row, n, rows, local_grid, comm);

        LOG("Computing interior cells");

        for(int i = 1; i < rows - 1; i++){
            for(int j = 0; j < n; j++){
                output_grid[i][j] = determine_state(i,j,n,rows,local_grid, comm);
            }
        }

        LOG("Computing boundary rows");

        for(int j = 0; j < n; j++){
            output_grid[0][j] = determine_state_manual(j,n,local_grid[1], local_grid[0], upper_row);
            output_grid[rows-1][j] = determine_state_manual(j,n,lower_row, local_grid[rows-1], local_grid[rows-2]);
        }

        LOG("Swapping grids");

        int** temp = local_grid;
        local_grid = output_grid;
        output_grid = temp;

        if(cycle % display_frequency == 0){
            LOG("Displaying grid");
            display_gol(n, rows, local_grid, comm);
        }

        LOG("MPI_Barrier");
        MPI_Barrier(comm);

        LOG("Cycle %d complete", cycle);
    }

    LOG("Freeing halo rows");
    free(upper_row);
    free(lower_row);

    LOG("Freeing output grid");
    // If we run for an odd number of generations, local grid and output grid will be swapped.
    // Caller expects output_grid to be freed, not local grid. So if it's odd, free local grid.
    if (G % 2 == 1) {
        free_grid(n, rows, local_grid);
    } else {
        free_grid(n, rows, output_grid);
    }

    LOG("Exiting simulate");
}

int determine_state_manual(int x, int n, int* lower, int* middle, int* upper)
{
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

int determine_state(int y, int x, int n, int rows, int** local_grid, MPI_Comm comm) 
{
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

void display_gol(int n, int rows, int** local_grid, MPI_Comm comm)
{
    LOG("Entered display_gol");

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    char filename[100];
    sprintf(filename, OUTFILE_FORMAT, rank);

    LOG("Writing local grid to %s", filename);

    FILE* outfile = fopen(filename, "w");

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < n; j++){
            fprintf(outfile, "%d ", local_grid[i][j]);
        }
        fprintf(outfile, "\n");
    }

    fclose(outfile);

    LOG("Barrier before printing");
    MPI_Barrier(comm);

    if (rank == 0) {

        LOG("Rank0 collecting outputs");

        size_t linesize = (n+1) * 2 * sizeof(char);
        char* line = malloc(linesize);

        for(int r = 0; r < p; r++) {

            sprintf(filename, OUTFILE_FORMAT, r);

            LOG("Reading %s", filename);

            FILE* fp = fopen(filename, "r");

            // printf("Reading process %d\n", r);

            while (getline(&line, &linesize, fp) != -1) {
                printf("%s", line);
            }

            // printf("Done reading process %d\n", r);

            fclose(fp);
        }

        free(line);
        printf("\n");
    }

    LOG("Exiting display_gol");
}