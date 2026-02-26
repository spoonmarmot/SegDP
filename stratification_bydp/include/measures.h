#ifndef MEASURES_H_
#define MEASURES_H_

#include "global.h"

#define MAX_TABLE_SZ 100

/**
 * @brief calculate ARI
 * 
 * @param table: table values
 * @param n_row: number of table rows 
 * @param n_col: number of table columns 
 * @param argc: number of extra arguments
 * @param argv: void pointers to actual arguments 
 * @return double 
 */
double adjusted_rand_index(int **table, int n_row, int n_col, int argc, void **argv);



#endif