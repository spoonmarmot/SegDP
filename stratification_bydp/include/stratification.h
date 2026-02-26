#ifndef _STRATIFICATION_BYDP_H_
#define _STRATIFICATION_BYDP_H_

#include "global.h"


/**
 * a contingency table of size (n_row, n_col) 
 */
typedef struct 
{
    
    int n_row;  /* number of table rows */ 
    int n_col;  /* number of table columns */ 
    int **tab;  /* values */
} crosstab;

/**
 * struct to store the prefix counts of each value of a categorical feature
 */
typedef struct 
{
    int n_unique;           /* number of unique values */
    int n_samples;          /* number of samples */
    double **prefix_count;  /* prefix sum of counts, shape=(n_unique, n_samples), being double for the convenience of computation */
} cate_fea_prefix_count;


/* result struct */
typedef struct {
    double score;      /* score of these rank segments  */
    int n_segs;        /* number of rank segments */
    int *seg_heads;    /* indices of the first element of each rank segment; */
    int n_samples;     /* number of samples */
    int *a_strata;     /* discrete feature values after stratification */
} rank_segments;

/**
 * a function pointer referring to a measuring function
 * 
 * @values can be a contingency table with @n_row rows and 
 * @n_col columns, e.g., a contingency table to calculate
 * ARI.
 * 
 * or @values can be a 1D array of size (2, @feature_size),
 * in this case @n_row=1 and @n_feature=@FEATURE_SZ. In this
 * case, measures like Pearson' corr can be evaluated
 */
typedef double (*ptr_measure_func)(double **values, int n_row, int n_col, int argc, void **argv);

void print_rank_segments(FILE *fp, rank_segments rs);

void destroy_rank_segments(rank_segments rs);

double W1D(double *prefix_sum, double *prefix_sq_sum, int l, int r);

double N1D(cate_fea_prefix_count prefix_count, int l, int r);

/**
 * @brief calculate feature weights based on NMI between features
 * 
 * @param value: feature values, shape=(m1+m2, n) with first m1 rows being numerical features and last m2 rows being categorical features. If categorical then its unique set must be continuous int starting from 0 
 * @param thd: threshold of NMI to consider feature correlation 
 * @param n: number of samples 
 * @param m1: number of numerical features 
 * @param m2: number of categorical features 
 * @param bins: number of bins to discretize numerical features for NMI calculation 
 * @return double** 
 */
double** compute_feature_weights(double **value, double thd, double n, int m1, int m2, int bins);

void destroy_cate_fea_prefix_count(cate_fea_prefix_count cfp);

/**
 * @brief stratify main feature a_value into rank segments using CH index
 * 
 * @param a_value: main feature value
 * @param b_value: assistant feature values, shape=(m1+m2, n) with first m1 rows being numerical features and last m2 rows being categorical features. If categorical then its unique set must be continuous int starting from 0
 * @param a_ws:   weight of the main feature 
 * @param weight: weights of each assistant feature 
 * @param n: number of values
 * @param m1: number of numerical features
 * @param m2: number of categorical features
 * @param K_max: maximum number of segments
 * @param min_len: minimum number of a segment length
 * @param target_k: the desired number of segments, if <0 then the function return the best results
 * @return rank_segments
 */
rank_segments stratify(const double *a_value, const double **b_value, const double a_ws, const double *weight, 
    int n, int m1, int m2, int K_max, int min_len, int target_k);

#endif