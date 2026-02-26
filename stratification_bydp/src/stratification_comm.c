#include "stratification.h"
#include "utils.h"
#include "linalg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void print_rank_segments(FILE *fp, rank_segments rs)
{
    fprintf(fp, "-----------------------------------------------------------------------\n");
    fprintf(fp, "Rank segments:\n\tscore=%f, |segments|=%d\n", rs.score, rs.n_segs);
    fprintf(fp, "\tsegment_heads: ");
    for (int i=0; i<rs.n_segs; i++) fprintf(fp, "%d ", rs.seg_heads[i]);
    fprintf(fp, "\n");
    // fprintf(stdout, "\tdiscretization: ");
    // for (int i=0; i<rs.n_samples; i++) fprintf(stdout, "%d ", rs.a_strata[i]);
    // fprintf(stdout, "\n");
    // fprintf(fp, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
}

void destroy_rank_segments(rank_segments rs)
{
    free(rs.seg_heads);
    free(rs.a_strata);
}

double W1D(double *prefix_sum, double *prefix_sq_sum, int l, int r)
{
    double tmp_d = 0, Wlr = 0;
    if (l == 0)
    {
        Wlr += check_zero(
            prefix_sq_sum[r] - (prefix_sum[r] / (r+1)) * prefix_sum[r]
        );
    }
    else
    {
        tmp_d = (l==0? 0 : prefix_sum[l-1]);
        Wlr -= check_zero(
            ((prefix_sum[r] - tmp_d) / (r-l+1)) * (prefix_sum[r] - tmp_d)
        );
                        
        tmp_d = (l==0? 0 : prefix_sq_sum[l-1]);
        Wlr += check_zero(
            (prefix_sq_sum[r] - tmp_d)
        );
    }
    if (Wlr < 0) Wlr = 0.0;
    else Wlr = check_zero(Wlr / (r-l+1));

    Wlr = (r-l+1) * (
        Wlr == 0 ? 
        LOG_ZERO : log(Wlr)
    ); 

    return check_zero(Wlr);
}

double nmi_from_table(double **n, int r, int c) 
{
    int i, j;
    double N, Hx, Hy, I, pi, pj, pij, denom, nmi;
    double *row = (double*)calloc(r, sizeof(double)); CHECK_ALLOC(row);
    double *col = (double*)calloc(c, sizeof(double)); CHECK_ALLOC(col);

    N = 0.0;
    for (i = 0; i < r; i++) {
        for (j = 0; j < c; j++) {
            if (n[i][j] < 0.0) { free(row); free(col); return NAN; }
            row[i] += n[i][j];
            col[j] += n[i][j];
            N += n[i][j];
        }
    }

    if (N <= 0.0) { free(row); free(col); return 0.0; }

    /* Entropies Hx, Hy */
    Hx = 0.0, Hy = 0.0;
    for (i = 0; i < r; i++) {
        if (row[i] > 0.0) {
            pi = row[i] / N;
            Hx -= pi * log(pi);
        }
    }
    for (j = 0; j < c; j++) {
        if (col[j] > 0.0) {
            pj = col[j] / N;
            Hy -= pj * log(pj);
        }
    }

    /* Mutual information I */
    I = 0.0;
    for (i = 0; i < r; i++) {
        if (row[i] <= 0.0) continue;
        pi = row[i] / N;

        for (j = 0; j < c; j++) {
            if (n[i][j] <= 0.0) continue;
            if (col[j] <= 0.0) continue;

            pj  = col[j] / N;
            pij = n[i][j] / N;

            I += pij * log(pij / (pi * pj));
        }
    }

    free(row);
    free(col);

    /* Handle degenerate entropies */
    if (Hx < EPS && Hy < EPS) {
        /* Both variables constant: treat as perfectly "matching" */
        return 1.0;
    }
    if (Hx < EPS || Hy < EPS) {
        /* One constant, the other not: no shared info under NMI conventions */
        return 0.0;
    }

    /* Normalize */
    denom = 0.5 * (Hx + Hy);

    if (denom < EPS) return 0.0;

    nmi = I / denom;

    /* Numerical safety clamp */
    if (nmi < 0.0) nmi = 0.0;
    if (nmi > 1.0) nmi = 1.0;

    return nmi;
}

/**
 * @brief calculate NMI between a numerical feature and a categorical feature
 * 
 * @param a_num: numerical feature values
 * @param b_cate: categorical feature values
 * @param a_cp: cutpoints to discretize numerical feature
 * @param n: number of samples
 * @param n_acp: number of cutpoints for numerical feature
 * @param n_buv: number of unique values for categorical feature
 * @return double 
 */
double nmi_num_cate(double *a_num, double *b_cate, double *a_cp,
    int n, int n_acp, int n_buv)
{
    double **tab, nmi;
    int i, k, a_row, b_col;
    
    tab = create_matrix_d(n_acp+1, n_buv, 0.0);

    for (i=0; i<n; i++)
    {
        for (a_row=0; a_row<n_acp; a_row++)
        {
            if (a_num[i]<=a_cp[a_row]) break;
        }
        b_col = (int)(b_cate[i]);

        tab[a_row][b_col] += 1.0;
    }

    nmi = nmi_from_table(tab, n_acp+1, n_buv);

    for (i=0; i<n_acp+1; i++)
    {
        free(tab[i]);
    }
    free(tab);

    return nmi;
}

/**
 * @brief calculate NMI between two numerical features
 * 
 * @param a_num: values of numerical feature a
 * @param b_num: values of numerical feature b
 * @param a_cp: cutpoints to discretize numerical feature a
 * @param b_cp: cutpoints to discretize numerical feature b
 * @param n: number of samples
 * @param n_acp: number of cutpoints for numerical feature a
 * @param n_bcp: number of cutpoints for numerical feature b
 * @return double 
 */
double nmi_num_num(double *a_num, double *b_num, double *a_cp, double *b_cp, 
    int n, int n_acp, int n_bcp)
{
    double **tab, nmi;
    int i, k, a_row, b_col;

    tab = create_matrix_d(n_acp+1, n_bcp+1, 0.0);

    for (i=0; i<n; i++)
    {
        for (a_row=0; a_row<n_acp; a_row++)
        {
            if (a_num[i]<=a_cp[a_row]) break;
        }
        for (b_col=0; b_col<n_bcp; b_col++)
        {
            if (b_num[i]<=b_cp[b_col]) break;
        }

        tab[a_row][b_col] += 1.0;
    }

    nmi = nmi_from_table(tab, n_acp+1, n_bcp+1);

    for (i=0; i<n_acp+1; i++)
    {
        free(tab[i]);
    }
    free(tab);

    return nmi;
}


/**
 * @brief calculate NMI between features
 * 
 * @param value: feature values, shape=(m1+m2, n) with first m1 rows being numerical features and last m2 rows being categorical features. If categorical then its unique set must be continuous int starting from 0 
 * @param n: number of samples 
 * @param m1: number of numerical features 
 * @param m2: number of categorical features 
 * @param bins: number of bins to discretize numerical features for NMI calculation 
 * @return double** 
 */
double** between_feature_nmi(double **value, double n, int m1, int m2, int bins)
{
    double **nmi, *numfea_max, *numfea_min, **numfea_cutpoints;
    int sidx, fidx, fidx2, max_cate_val;
    
    // get cutpoints
    numfea_min = (double *) malloc(m1 * sizeof(double)); CHECK_ALLOC(numfea_min);
    numfea_max = (double *) malloc(m1 * sizeof(double)); CHECK_ALLOC(numfea_max);
    numfea_cutpoints = create_matrix_d(m1, bins-1, NEG_INF);
    for (fidx=0; fidx<m1; fidx++)
    {
        numfea_min[fidx] = - NEG_INF;
        numfea_max[fidx] =   NEG_INF;
        for (sidx=0; sidx<n; sidx++)
        {
            numfea_min[fidx] = (numfea_min[fidx] < value[fidx][sidx] ? numfea_min[fidx] : value[fidx][sidx]);
            numfea_max[fidx] = (numfea_max[fidx] > value[fidx][sidx] ? numfea_max[fidx] : value[fidx][sidx]);
        }

        for (sidx=0; sidx<bins-1; sidx++)
        {
            numfea_cutpoints[fidx][sidx] = numfea_min[fidx] + (sidx + 1) * ((numfea_max[fidx] - numfea_min[fidx]) / bins);
        }
    }

    // nmi
    nmi = create_matrix_d(m1, m1+m2, NEG_INF);
    // nmi between two numerical features
    for (fidx=0; fidx<m1; fidx++)
    {
        for (fidx2=fidx; fidx2<m1; fidx2++)
        {
            if (fidx == fidx2) 
            {
                nmi[fidx][fidx2] = 1;
            }
            else 
            {
                nmi[fidx][fidx2] = nmi_num_num(value[fidx], value[fidx2], 
                    numfea_cutpoints[fidx], numfea_cutpoints[fidx2],
                    n, bins-1, bins-1
                );
                nmi[fidx2][fidx] = nmi[fidx][fidx2];
            }
        }
    }
    // nmi between a numerical feature and a categorical feature
    for (fidx=0; fidx<m1; fidx++)
    {
        for (fidx2=0; fidx2<m2; fidx2++)
        {
            max_cate_val = -1;
            for (sidx=0; sidx<n; sidx++)
            {
                if (value[m1+fidx2][sidx] > max_cate_val) 
                    max_cate_val = (int)(value[m1+fidx2][sidx]);
            }

            nmi[fidx][m1+fidx2] = nmi_num_cate(value[fidx], value[m1+fidx2], 
                numfea_cutpoints[fidx],
                n, bins-1, max_cate_val+1
            );
        }
    }

    // clean up
    free(numfea_max);
    free(numfea_min);
    for (fidx=0; fidx<m1; fidx++)
    {
        free(numfea_cutpoints[fidx]);
    }
    free(numfea_cutpoints);

    return nmi;

}

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
double** compute_feature_weights(double **value, double thd, double n, int m1, int m2, int bins)
{
    int fidx1, fidx2;
    double **nmi, **weights, ws;

    nmi = between_feature_nmi(value, n, m1, m2, bins);

    weights = create_matrix_d(m1, m1+m2, NEG_INF);
    for (fidx1=0; fidx1<m1; fidx1++)
    {
        ws = 0.0;
        for(fidx2=0; fidx2<m1+m2; fidx2++)
        {
            if (fidx2 == fidx1)
            {
                weights[fidx1][fidx2] = 1;
            }
            else{
                weights[fidx1][fidx2] = (nmi[fidx1][fidx2] < thd) ? 0.0 : nmi[fidx1][fidx2];
                ws += weights[fidx1][fidx2];
            }
        }

        for (fidx2=0; fidx2<m1+m2; fidx2++)
        {
            if (fidx2 == fidx1 || ws == 0.0) continue;
            else 
                weights[fidx1][fidx2] /= ws;
        }
    }

    for (fidx1=0; fidx1<m1; fidx1++)
    {
        free(nmi[fidx1]);
    }
    free(nmi);

    return weights;
}

double KL_from_2coltab(double **tab, int r)
{
    double sum_c0 = 0, sum_c1 = 0, kl, p, q, lp, lq;
    int ridx;

    for (ridx=0; ridx<r; ridx++)
    {
        sum_c0 += tab[ridx][0];
        sum_c1 += tab[ridx][1];
    }

    kl = 0.0;
    for (ridx=0; ridx<r; ridx++)
    {
        p = tab[ridx][0] / sum_c0;
        lp = (p <= EPS ? LOG_ZERO : log(p));

        q = tab[ridx][1] / sum_c1;
        lq = (q <= EPS ? LOG_ZERO : log(q));

        kl += p * (lp - lq);
    }

    return check_zero(kl);

}

/**
 * @brief calculate the maximum likelihood of a categorical feature 
 */
double CLL_from_2coltab(double **tab, int r)
{
    double sum_c0 = 0, cll, p, lp;
    int ridx;
    
    for (ridx=0; ridx<r; ridx++)
    {
        sum_c0 += tab[ridx][0];
    }

    cll = 0.0;
    for (ridx=0; ridx<r; ridx++)
    {
        p = tab[ridx][0] / sum_c0;
        lp = (p <= EPS ? LOG_ZERO : log(p));
        cll += tab[ridx][0] * lp;
    }

    return check_zero(cll);
    
}

double N1D(cate_fea_prefix_count prefix_count, int l, int r)
{
    double Nlr = 0.0, **tab, plr;
    int ridx, n = prefix_count.n_samples;

    tab = create_matrix_d(prefix_count.n_unique, 2, 0);

    for (ridx=0; ridx<prefix_count.n_unique; ridx++)
    {
        if (l == 0)
        {
            plr = prefix_count.prefix_count[ridx][r];
        }
        else
        {
            plr = prefix_count.prefix_count[ridx][r] - prefix_count.prefix_count[ridx][l-1];
        }
        tab[ridx][0] = plr;
        tab[ridx][1] = prefix_count.prefix_count[ridx][n-1] - plr;
    }

    // Nlr = nmi_from_table(tab, prefix_count.n_unique, 2);
    // Nlr = KL_from_2coltab(tab, prefix_count.n_unique);
    Nlr = CLL_from_2coltab(tab, prefix_count.n_unique);
    Nlr = -2 * Nlr;

    for (ridx=0; ridx<prefix_count.n_unique; ridx++)
    {
        free(tab[ridx]);
    }
    free(tab);

    return check_zero(Nlr);
}

void destroy_cate_fea_prefix_count(cate_fea_prefix_count cfp)
{
    int i;
    for (i=0; i<cfp.n_unique; i++)
    {
        free(cfp.prefix_count[i]);
    }
    free(cfp.prefix_count);
}
