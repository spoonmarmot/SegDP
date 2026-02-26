#include "utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_statistics_double.h>

/**
 * @brief a comparing function to for two indexed_value variables
 *        a >  b ->  1
 *        a == b ->  results between a_index and b_index
 *        a <  b -> -1
 * 
 * @param a 
 * @param b 
 */
int compare_indexed_value(const void *a, const void *b)
{
    indexed_value *ia = (indexed_value *)a;
    indexed_value *ib = (indexed_value *)b;
    if (check_zero(ia->value - ib->value))
    // if different value, compare by value
    {
        return (ia->value > ib->value) - (ia->value < ib->value);
    }
    else
    // if same value, compare by index
    {
        return (ia->index > ib->index) - (ia->index < ib->index);
    }
}

/**
 * @brief a comparing function to for two indexed_value variables only using their values
 *        a >  b ->  1
 *        a == b ->  0
 *        a <  b -> -1
 * 
 * @param a 
 * @param b 
 */
int compare_indexed_value_byval(const void *a, const void *b)
{
    indexed_value *ia = (indexed_value *)a;
    indexed_value *ib = (indexed_value *)b;
    if (check_zero(ia->value - ib->value) > 0) return 1;
    else if (check_zero(ia->value - ib->value) < 0) return -1;
    else return 0;

}

/**
 * @brief argsort using qsort
 * 
 * @param values: values to be sorted 
 * @param size: length of @values 
 * @return int*: an array of indices 
 */
int* argsort(const double *values, const int size)
{
    int *sorted_indices = (int *)malloc(size * sizeof(int));
    indexed_value *iv = (indexed_value *)malloc(size * sizeof(indexed_value));

    /* check memory */
    CHECK_ALLOC(sorted_indices);
    CHECK_ALLOC(iv);
    /* create index_value array */
    for(int i=0; i<size; i++)
    {
        iv[i].index = i;
        iv[i].value = values[i];
    }
    /* quick sort */
    qsort(iv, size, sizeof(indexed_value), compare_indexed_value); 
    /* retrieve indices to @sorted_indices */
    for(int i=0; i<size; i++)
    {
        sorted_indices[i] = iv[i].index;
    }
    /* free memory */
    free(iv);

    return sorted_indices;
}


/**
 * @brief: replace values in a double array with their ranks by
 * argsort(argsort(values))
 * 
 * @param values: value array
 * @param size: length of the array 
 * @return int* 
 */
int* value_to_rank(const double *values, const int size)
{
    int *srted_indices, *ranks;
    double *d_srted_indices;

    /* inner layer argsort(values) */
    srted_indices = argsort(values, size);

    /* outer layer argsort(argsort(values)) */
    d_srted_indices = (double *)malloc(size * sizeof(double));
    CHECK_ALLOC(d_srted_indices);
    for(int i=0; i<size; i++)
    {
        d_srted_indices[i] = srted_indices[i];
    }
    ranks = argsort(d_srted_indices, size);
    free(d_srted_indices);
    free(srted_indices); 

    return ranks;
}


void print_matrix(double **matrix, int n_row, int n_col)
{
    for(int i=0; i<n_row; i++)
    {
        for(int j=0; j<n_col; j++)
        {
            printf("%f, ", matrix[i][j]);
        }
        printf("\n");
    }
}

double** create_matrix_d(int n_row, int n_col, double init_val)
{
    double **matrix;
    int i, j;

    matrix = (double **)malloc(n_row * sizeof(double *)); CHECK_ALLOC(matrix);
    for(i=0; i<n_row; i++)
    {
        matrix[i] = (double *)malloc(n_col * sizeof(double)); CHECK_ALLOC(matrix[i]);
        for(j=0; j<n_col; j++) matrix[i][j] = init_val;
    }

    return matrix;
}

int** create_matrix_i(int n_row, int n_col, int init_val)
{
    int **matrix;
    int i, j;

    matrix = (int **)malloc(n_row * sizeof(int *)); CHECK_ALLOC(matrix);
    for(i=0; i<n_row; i++)
    {
        matrix[i] = (int *)malloc(n_col * sizeof(int)); CHECK_ALLOC(matrix[i]);
        for(j=0; j<n_col; j++) matrix[i][j] = init_val;
    }

    return matrix;
}

double check_zero(double val)
{
    if (fabs(val) < EPS) return 0.0;
    else return val;
}

void rank_based_normal_scores(double *x, int n, double *dest)
{
    int i, j, r, start, end, orig_idx; 
    double rank_first, rank_last, rank_mid, p, z;
    indexed_value *arr = malloc(n * sizeof(indexed_value)); CHECK_ALLOC(arr);

    for (i = 0; i < n; ++i) {
        arr[i].value = x[i];
        arr[i].index = i;
    }

    qsort(arr, n, sizeof(indexed_value), compare_indexed_value_byval);

    r = 0;
    while (r < n) {
        start = r;
        end   = r;

        // find the end of the tie block: arr[start..end] have (approximately) same value
        while (end + 1 < n && fabs(arr[end + 1].value - arr[start].value) <= EPS)
        {
            end++;
        }

        // ranks are 1-based: start+1 ... end+1
        rank_first = (double)start + 1.0;
        rank_last  = (double)end   + 1.0;
        rank_mid   = 0.5 * (rank_first + rank_last);

        // e.g. van der Waerden-style: p = (R_mid - 0.5) / n
        p = (rank_mid - 0.5) / (double)n;
        if (p <= 0.0) p = EPS;
        if (p >= 1.0) p = 1.0 - EPS;

        z = gsl_cdf_ugaussian_Pinv(p);

        // assign same z to all tied observations
        for (j = start; j <= end; ++j) {
            orig_idx = arr[j].index;
            dest[orig_idx] = z;
        }

        r = end + 1;
    }

    free(arr);
}

void minmax_scalar(double *x, int n, double *dest)
{
    double min = -NEG_INF, max = NEG_INF;
    int i;
    
    for (i=0; i<n; i++)
    {
        if (check_zero(x[i] - min) < 0) min = x[i];
        if (check_zero(x[i] - max) > 0) max = x[i];
    }

    for (i=0; i<n; i++)
    {
        if (check_zero(min - max) == 0)
        {
            dest[i] = 1;
        }
        else 
        {
            dest[i] = (x[i]-min) / (max-min);
        }
    }

}

// Uniform in (0,1), avoiding exact 0 and 1
static double urand01(void) {
    return (rand() + 1.0) / (RAND_MAX + 2.0);
}

// sampling a r.v. from Normal(mean, stddev^2) using Marsaglia method
double rand_normal(double mean, double stddev) {
    static int hasSpare = 0;
    static double spare;

    if (hasSpare) {
        hasSpare = 0;
        return mean + stddev * spare;
    }

    double u, v, s;
    do {
        u = 2.0 * urand01() - 1.0;  // (-1,1)
        v = 2.0 * urand01() - 1.0;  // (-1,1)
        s = u*u + v*v;
    } while (s >= 1.0 || s == 0.0);

    double mul = sqrt(-2.0 * log(s) / s);
    spare = v * mul;
    hasSpare = 1;
    return mean + stddev * (u * mul);
}

void robust_scaling(double *x, int n, double q1, double q3, double *dest)
{
    double *tmp = (double*)malloc((size_t)n * sizeof(double)); CHECK_ALLOC(tmp);

    memcpy(tmp, x, (size_t)n * sizeof(double));
    gsl_sort(tmp, 1, (size_t)n);

    double center = gsl_stats_median_from_sorted_data(tmp, 1, (size_t)n);
    double lo     = gsl_stats_quantile_from_sorted_data(tmp, 1, (size_t)n, q1);
    double hi     = gsl_stats_quantile_from_sorted_data(tmp, 1, (size_t)n, q3);

    free(tmp);

    double scale = hi - lo;
    if (fabs(scale) < EPS) scale = 1.0;  // avoid divide-by-zero

    for (int i = 0; i < n; ++i) {
        dest[i] = (x[i] - center) / scale;
    }

    return 0;

}

void standardize(double *x, int n, double *dest)
{
    double mu = 0, std = 0;
    int i;

    /* mean */
    for (i=0; i<n; i++ ) mu += x[i];
    mu /= n;

    /* std */
    for (i=0; i<n; i++) std += (x[i] - mu) * (x[i] - mu);
    std = sqrt(std / (n-1));

    /* standardize */
    for (i=0; i<n; i++) dest[i] = (x[i] - mu) / std;
}
