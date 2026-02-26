#include "linalg.h"
#include "global.h"
#include "lapacke.h"
#include "utils.h"

/**
 * @brief calculate covariance matrix and mu vector
 */
void cov_mu_lr_vri(
    double *dest_cov,           // destination of covariance matrix, (m2-m1+1) * (m2-m1+1)
    double *dest_mu,            // destination of mu vector, (m2-m1+1, )
    value_rank_index **b_vri,   // feature values, row<=>feature, col<=>sample
    int l,                      // index of the left sample 
    int r,                      // index of the right sample (included)
    int m1,                     // index of the first feature
    int m2                      // index of the last feature (included) 
)
{
    int fidx1, fidx2, sidx1, n_fea, n_sample; 
    n_fea = m2 - m1 + 1;
    n_sample = r - l + 1;
    
    // calculate mean vector
    for (fidx1=0; fidx1<n_fea; fidx1++)
    {
        dest_mu[fidx1] = 0;
        for (sidx1=0; sidx1<n_sample; sidx1++)
        {
            dest_mu[fidx1] += b_vri[m1+fidx1][l+sidx1].value;
        }
        dest_mu[fidx1] /= n_sample;
    }

    // calculate covariance matrix
    for (fidx1=0; fidx1<n_fea; fidx1++)
    {
        for (fidx2=fidx1; fidx2<n_fea; fidx2++)
        {
            dest_cov[fidx1 * n_fea + fidx2] = 0;
            dest_cov[fidx2 * n_fea + fidx1] = 0;
            for (sidx1=0; sidx1<n_sample; sidx1++)
            {
                dest_cov[fidx1 * n_fea + fidx2] 
                += (b_vri[m1+fidx1][l+sidx1].value - dest_mu[fidx1])
                *  (b_vri[m1+fidx2][l+sidx1].value - dest_mu[fidx2]);
            }
            dest_cov[fidx1 * n_fea + fidx2] /= (n_sample==1? 1:(n_sample-1));
            dest_cov[fidx2 * n_fea + fidx1] =  dest_cov[fidx1 * n_fea + fidx2];
        }
    }
}

void cov_mu_lr(
    double *dest_cov,           // destination of covariance matrix, (m2-m1+1) * (m2-m1+1)
    double *dest_mu,            // destination of mu vector, (m2-m1+1, )
    double **b_value,           // feature values, row<=>feature, col<=>sample
    int l,                      // index of the left sample 
    int r,                      // index of the right sample (included)
    int m1,                     // index of the first feature
    int m2                      // index of the last feature (included) 
)
{
    int fidx1, fidx2, sidx1, n_fea, n_sample; 
    n_fea = m2 - m1 + 1;
    n_sample = r - l + 1;
    
    // calculate mean vector
    for (fidx1=0; fidx1<n_fea; fidx1++)
    {
        dest_mu[fidx1] = 0;
        for (sidx1=0; sidx1<n_sample; sidx1++)
        {
            dest_mu[fidx1] += b_value[m1+fidx1][l+sidx1];
        }
        dest_mu[fidx1] /= n_sample;
    }

    // calculate covariance matrix
    for (fidx1=0; fidx1<n_fea; fidx1++)
    {
        for (fidx2=fidx1; fidx2<n_fea; fidx2++)
        {
            dest_cov[fidx1 * n_fea + fidx2] = 0;
            dest_cov[fidx2 * n_fea + fidx1] = 0;
            for (sidx1=0; sidx1<n_sample; sidx1++)
            {
                dest_cov[fidx1 * n_fea + fidx2] 
                += (b_value[m1+fidx1][l+sidx1] - dest_mu[fidx1])
                *  (b_value[m1+fidx2][l+sidx1] - dest_mu[fidx2]);
            }
            dest_cov[fidx1 * n_fea + fidx2] /= (n_sample==1? 1:(n_sample-1));
            dest_cov[fidx2 * n_fea + fidx1] =  dest_cov[fidx1 * n_fea + fidx2];
        }
    }
}


/**
 * @brief calculate trace of the covariance matrix and mu vector
 */
void covtrace_mu_lr_vri(
    double *dest_trace,         // destination of the trace of the covariance matrix
    double *dest_mu,            // destination of mu vector, (m2-m1+1, )
    value_rank_index **b_vri,   // feature values, row<=>feature, col<=>sample
    int l,                      // index of the left sample 
    int r,                      // index of the right sample (included)
    int m1,                     // index of the first feature
    int m2                      // index of the last feature (included) 
)
{
    int fidx1, fidx2, sidx1, n_fea, n_sample; 
    n_fea = m2 - m1 + 1;
    n_sample = r - l + 1;
    
    // calculate mean vector
    for (fidx1=0; fidx1<n_fea; fidx1++)
    {
        dest_mu[fidx1] = 0;
        for (sidx1=0; sidx1<n_sample; sidx1++)
        {
            dest_mu[fidx1] += b_vri[m1+fidx1][l+sidx1].value;
        }
        dest_mu[fidx1] /= n_sample;
    }

    // calculate covariance matrix
    *dest_trace = 0;
    for (fidx1=0; fidx1<n_fea; fidx1++)
    {   
        for (sidx1=0; sidx1<n_sample; sidx1++)
        {
            *dest_trace 
            += (b_vri[m1+fidx1][l+sidx1].value - dest_mu[fidx1])
            *  (b_vri[m1+fidx1][l+sidx1].value - dest_mu[fidx1]);
        }
    }

    *dest_trace /= (n_sample - 1);

}


/**
 * @brief calculate trace of the covariance matrix and mu vector
 */
void covtrace_mu_lr(
    double *dest_trace,         // destination of the trace of the covariance matrix
    double *dest_mu,            // destination of mu vector, (m2-m1+1, )
    double **b_value,   // feature values, row<=>feature, col<=>sample
    int l,                      // index of the left sample 
    int r,                      // index of the right sample (included)
    int m1,                     // index of the first feature
    int m2                      // index of the last feature (included) 
)
{
    int fidx1, fidx2, sidx1, n_fea, n_sample; 
    n_fea = m2 - m1 + 1;
    n_sample = r - l + 1;
    
    // calculate mean vector
    for (fidx1=0; fidx1<n_fea; fidx1++)
    {
        dest_mu[fidx1] = 0;
        for (sidx1=0; sidx1<n_sample; sidx1++)
        {
            dest_mu[fidx1] += b_value[m1+fidx1][l+sidx1];
        }
        dest_mu[fidx1] /= n_sample;
    }

    // calculate covariance matrix
    *dest_trace = 0;
    for (fidx1=0; fidx1<n_fea; fidx1++)
    {   
        for (sidx1=0; sidx1<n_sample; sidx1++)
        {
            *dest_trace 
            += (b_value[m1+fidx1][l+sidx1] - dest_mu[fidx1])
            *  (b_value[m1+fidx1][l+sidx1] - dest_mu[fidx1]);
        }
    }

    *dest_trace /= (n_sample - 1);

}

/**
 * @brief calculate mean vector
 */
double* comp_mu_lr(double **value, int l, int r, int m1, int m2)
{
    int fidx1, sidx1, n_fea, n_sample; 
    double *dest_mu;
    n_fea = m2 - m1 + 1;
    n_sample = r - l + 1;
    
    dest_mu = (double *)calloc(n_fea, sizeof(double)); CHECK_ALLOC(dest_mu);
    // calculate mean vector
    for (fidx1=0; fidx1<n_fea; fidx1++)
    {
        for (sidx1=0; sidx1<n_sample; sidx1++)
        {
            dest_mu[fidx1] += value[m1+fidx1][l+sidx1];
        }
        dest_mu[fidx1] /= n_sample;
    }

    return dest_mu;
}

double squared_mahalanobis_norm(double *x, double *mu, double *inv_cov, int m)
{
    double res = 0;
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<m; j++)
        {
            res += (x[i] - mu[i]) * inv_cov[i * m + j] * (x[j] - mu[j]);
        }
    }
    return res;
}


double squared_L2_norm(double *x, double *mu, int m)
{
    double res = 0;
    for (int i=0; i<m; i++)
    {
        res += (x[i] - mu[i]) * (x[i] - mu[i]);
    }
    return res;
}


double sum_of_squared_errors(double **value, double* mu, int l, int r, int m1, int m2)
{
    double ss, tmp_d, *mu_g;
    int fidx1, sidx1, n_fea, n_sample;

    if (mu == NULL) mu_g = comp_mu_lr(value, l, r, m1, m2);
    else mu_g = mu;

    n_fea = m2 - m1 + 1;
    n_sample = r - l + 1;
    ss = 0;

    for (fidx1=0; fidx1<n_fea; fidx1++)
    {
        for (sidx1=0; sidx1<n_sample; sidx1++)
        {
            tmp_d = check_zero(value[m1+fidx1][l+sidx1] - mu_g[fidx1]);
            ss += tmp_d * tmp_d;        
        }
    }

    if (mu == NULL) free(mu_g);

    return ss;
}

double trace(double *mat, int m)
{
    double res = 0;
    for (int i=0; i<m; i++)
    {
        res += mat[i * m + i];
    }
    return res;
}


int inv_cov(double *cov, int m)
{
    int info;
    int *ipiv = (int*)malloc(m * sizeof(int)); CHECK_ALLOC(ipiv);

    // LU factorization: A = P * L * U
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, m, cov, m, ipiv);
    if (info != 0) {
        free(ipiv);
        return info;  // info < 0: bad arg; info > 0: singular matrix
    }

    // Compute inverse from LU factorization
    info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, m, cov, m, ipiv);
    free(ipiv);

    return info;  // info > 0 means U(k,k) = 0 -> singular
}


int whitening_Cholesky(double **values, int n, int m1, int m2)
{
    double *A, *mu, **whitened_values, *whitened_values_flat;
    int i, j, k, info, m;

    m = m2 - m1 + 1;
    A = (double *)malloc(m * m * sizeof(double)); CHECK_ALLOC(A);
    mu = (double *)malloc(m * sizeof(double)); CHECK_ALLOC(mu);

    // compute covariance matrix and mean vector
    cov_mu_lr(A, mu, values, 0, n-1, m1, m2);

    // centralize values
    whitened_values_flat = (double *)malloc(m * n * sizeof(double)); CHECK_ALLOC(whitened_values_flat);
    for (i=0; i<m; i++)
    {
        for (j=0; j<n; j++)
        {
            whitened_values_flat[i * n + j] = values[m1+i][j] - mu[i];
        }
    }
    
    // Cholesky decomposition of covariance matrix: A = L * L^T
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', m, A, m);
    if (info != 0) 
    {
        free(whitened_values_flat);
        free(A);
        free(mu);
        return info;  // matrix not positive definite
    }

    // Now the variable A is L
    // whitening by soloving L * Z = X <=>  Z = L^{-1} * X
    info = LAPACKE_dtrtrs(
        LAPACK_ROW_MAJOR, 
        'L', 
        'N', 
        'N', 
        m, 
        n, 
        A, 
        m, 
        whitened_values_flat, 
        n
    );  
    if (info != 0) 
    {
        free(whitened_values_flat);
        free(A);
        free(mu);
        return info;  // matrix not positive definite
    }

    // reshape whitened_values_flat to whitened_values
    for (i=0; i<m; i++)
    {
        for (j=0; j<n; j++)
        {
            values[m1+i][j] = whitened_values_flat[i * n + j];
        }
    }
 
    free(whitened_values_flat);
    free(A);
    free(mu);

    return 0;
}


double** comp_prefix_sum(double **src, int l, int r, int beg, int end)
{
    int fidx, sidx;
    double **dest;
    
    dest = (double **)malloc((end - beg + 1) * sizeof(double *)); CHECK_ALLOC(dest);
    for (fidx=0; fidx<=end-beg; fidx++)
    {
        dest[fidx] = (double *)malloc((r - l + 1) * sizeof(double)); CHECK_ALLOC(dest[fidx]);
    }

    for (fidx=0; fidx<=end-beg; fidx++)
    {
        for (sidx=0; sidx<=r-l; sidx++)
        {
            if (sidx == 0)
            {
                dest[fidx][sidx] = src[beg+fidx][l+sidx];
            }
            else
            {
                dest[fidx][sidx] = dest[fidx][sidx-1] + src[beg+fidx][l+sidx];
            }
        }
    }

    return dest;
}

double** comp_prefix_sq_sum( double **src, int l, int r, int beg, int end)
{
    int fidx, sidx;
    double **dest;
    
    dest = (double **)malloc((end - beg + 1) * sizeof(double *)); CHECK_ALLOC(dest);
    for (fidx=0; fidx<=end-beg; fidx++)
    {
        dest[fidx] = (double *)malloc((r - l + 1) * sizeof(double)); CHECK_ALLOC(dest[fidx]);
    }

    for (fidx=0; fidx<=end-beg; fidx++)
    {
        for (sidx=0; sidx<=r-l; sidx++)
        {
            if (sidx == 0)
            {
                dest[fidx][sidx] = src[beg+fidx][l+sidx] * src[beg+fidx][l+sidx];
            }
            else
            {
                dest[fidx][sidx] = dest[fidx][sidx-1] + src[beg+fidx][l+sidx] * src[beg+fidx][l+sidx];
            }
        }
    }

    return dest;
}


