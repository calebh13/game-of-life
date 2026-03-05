#ifndef FUNCS
#define FUNCS

#include <mpi.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>

void GenerateInitialGoL(int n, int rows, int** local_grid, MPI_Comm comm);
void free_grid(int n, int rows, int** local_grid);
void simulate(int G, int n, int rows, int** local_grid, MPI_Comm comm);
int determine_state(int x, int y, int n, int rows, int** local_grid, MPI_Comm comm);
void display_gol(int n, int rows, int** local_grid, MPI_Comm comm);
int determine_state_manual(int x, int n, int* lower, int* middle, int* upper);

#endif