#ifndef _LINALG_H_
#define _LINALG_H_

#include "global.h"

/**
 * @brief calculate covariance matrix and mean vector
 * 
 * @param dest_cov      :destination of covariance matrix, (m2-m1+1) * (m2-m1+1)
 * @param dest_mu       :destination of mu vector, (m2-m1+1, )
 * @param b_vri/b_value : feature values, row<=>feature, col<=>sample
 * @param l             : index of the left sample 
 * @param r             : index of the right sample (included)
 * @param m1            : index of the first feature
 * @param m2            : index of the last feature (included) 
 */
void cov_mu_lr_vri(double *dest_cov, double *dest_mu, value_rank_index **b_vri, int l, int r, int m1, int m2);

void cov_mu_lr(double *dest_cov, double *dest_mu, double **b_value, int l, int r, int m1, int m2);

/**
 * @brief 
 */

/**
 * @brief calculate trace of the covariance matrix and mu vector
 * 
 * @param dest_trace    : destination of the trace of the covariance matrix 
 * @param dest_mu       : destination of mu vector, (m2-m1+1, )
 * @param b_vri/b_value : feature values, row<=>feature, col<=>sample
 * @param l             : index of the left sample 
 * @param r             : index of the right sample (included)
 * @param m1            : index of the first feature
 * @param m2            : index of the last feature (included) 
 */
void covtrace_mu_lr_vri(double *dest_trace, double *dest_mu, value_rank_index **b_vri, int l, int r, int m1, int m2);

void covtrace_mu_lr(double *dest_trace, double *dest_mu, double **b_double, int l, int r, int m1, int m2);

/**
 * @brief calculate mean vector
 *
 * @param value       : feature values, row<=>feature, col<=>sample
 * @param l             : index of the left sample 
 * @param r             : index of the right sample (included)
 * @param m1            : index of the first feature
 * @param m2            : index of the last feature (included) 
 */
double* comp_mu_lr(double **value, int l, int r, int m1, int m2);

/**
 * @brief square Mahalanobis norm
 * 
 * (x-mu)^T*inv_cov*(x-mu)
 * 
 * @param x: X vector
 * @param mu: mean vector
 * @param inv_cov: inverse covariance matrix
 * @param m: size 
 */
double squared_mahalanobis_norm(double *x, double *mu, double *inv_cov, int m);

/**
 * @brief square Mahalanobis norm
 * 
 * (x-mu)^T*(x-mu)
 * 
 * @param x: X vector
 * @param mu: mean vector
 * @param m: size 
 */
double squared_L2_norm(double *x, double *mu, int m);

/**
 * @brief calculate sum of squared errors of the matrix[l...r, m1...m2]
 * 
 * @param value: feature values, row<=>feature, col<=>sample 
 * @param mu: mean vector of size (m2-m1+1), NULL if not provided
 * @param l:  left sample index (0-based) 
 * @param r:  right sample index (0-based, inclusive) 
 * @param m1: starting feature index (0-based) 
 * @param m2: ending feature index (0-based, inclusive) 
 * @return double 
 */
double sum_of_squared_errors(double **value, double *mu, int l, int r, int m1, int m2);

/**
 * @brief trace of a matrix
 * 
 * @param mat: 1D expansion of the matrix
 * @param m:   size of the matrix, #row 
 * @return double 
 */
double trace(double *mat, int m);

/**
 * @brief get inverse of a covariance matrix 
 * 
 * @param cov: the covariance 
 * @param m: size of the matrix 
 * @return int: =0 means complete, >0 means singular, <0 means bad arguments
 */
int inv_cov(double *cov, int m);

/**
 * @brief Whitening transformation using Cholesky decomposition
 * 
 * @param values: 2D array of size (>=m)*n, m features and n samples 
 * @param n: number of samples 
 * @param m1: starting feature index (0-based)
 * @param m2: ending feature index (0-based, inclusive) 
 * @return int: =0 means complete
 */
int whitening_Cholesky(double **values, int n, int m1, int m2);

/**
 * @brief compute prefix sum matrix, each row is a feature and each column is a sample.
 * The prefix sum is computed along each row, i.e., sum of previous k samples of each feature.
 * 
 * @param src: source matrix, size (>=m2)*n 
 * @param l: left sample index (0-based) 
 * @param r: right sample index (0-based, inclusive) 
 * @param m1: starting feature index (0-based) 
 * @param m2: ending feature index (0-based, inclusive) 
 * @return double** 
 */
double** comp_prefix_sum(double **src, int l, int r, int m1, int m2);

/**
 * @brief compute prefix square sum matrix, each row is a feature and each column is a sample.
 * The prefix sum is computed along each row, i.e., sum of previous k samples of each feature.
 * 
 * @param src: source matrix, size (>=m2)*n 
 * @param l: left sample index (0-based) 
 * @param r: right sample index (0-based, inclusive) 
 * @param m1: starting feature index (0-based) 
 * @param m2: ending feature index (0-based, inclusive) 
 * @return double** 
 */
double** comp_prefix_sq_sum(double **src, int l, int r, int m1, int m2);

#endif
