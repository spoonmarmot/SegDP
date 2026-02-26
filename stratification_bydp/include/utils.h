#ifndef UTILS_H_
#define UTILS_H_

#include "global.h"

/**
 * a value with index for argsort
 */
typedef struct
{
    int index;
    double value;
} indexed_value;

/**
 * @brief a comparing function to for two indexed_value variables
 *        a >  b ->  1
 *        a == b ->  a.index - b.index
 *        a <  b -> -1
 * 
 * @param a 
 * @param b 
 */
int compare_indexed_value(const void *a, const void *b);

/**
 * @brief a comparing function to for two indexed_value variables only using their values
 *        a >  b ->  1
 *        a == b ->  0
 *        a <  b -> -1
 * 
 * @param a 
 * @param b 
 */
int compare_indexed_value_byval(const void *a, const void *b);


/**
 * @brief argsort using qsort
 * 
 * @param values: values to be sorted 
 * @param size: length of @values 
 * @return int*: an array of indices 
 */
int* argsort(const double *values, const int size);

/**
 * @brief: replace values in a double array with their ranks
 * 
 * @param values: value array
 * @param size: length of the array 
 * @return int* 
 */
int* value_to_rank(const double *values, const int size);

/**
 * @brief  print a double matrix
 * 
 * @param matrix 
 * @param n_row 
 * @param n_col 
 */
void print_matrix(double **matrix, int n_row, int n_col);

/**
 * @brief Create a double matrix object
 * 
 * @param n_row: number of rows 
 * @param n_col: number of columns 
 * @param init_val: initial value for each element 
 * @return double** 
 */
double **create_matrix_d(int n_row, int n_col, double init_val);
int** create_matrix_i(int n_row, int n_col, int init_val);

/**
 * @brief  return zero if the absolute value is smaller than eps
 * 
 * @param val 
 * @return double 
 */
double check_zero(double val);

/**
 * @brief convert values to rank based normal scores
 * 
 * @param x 
 * @param n 
 * @param dest: destined memory 
 */
void rank_based_normal_scores(double *x, int n, double *dest);


/**
 * @brief min max scalar
 * 
 * @param x 
 * @param n 
 * @param dest: destined memory 
 */
void minmax_scalar(double *x, int n, double *dest);

/**
 * @brief sampling a r.v. from Normal(mean, stddev^2) using Marsaglia method
 * 
 * @param mean 
 * @param stddev 
 * @return double 
 */
double rand_normal(double mean, double stddev);


void robust_scaling(double *x, int n, double q1, double q3, double *dest);


void standardize(double *x, int n, double *dest);

#endif
