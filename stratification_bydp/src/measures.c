#include "measures.h"

/* adjusted rand index */
double adjusted_rand_index(int **table, int n_row, int n_col, int argc, void **argv)
{
    /* row marginal sums, columns marginal sums, and the total sum */
    double row_sum[MAX_TABLE_SZ] = {0}, col_sum[MAX_TABLE_SZ] = {0}, tot_sum = 0;
    /* sum of binom(n_ij, 2), sum of binom(n_i*, 2), sum of binom(n_*j, 2), and binom(n, 2) */
    double cell_binom_sum = 0, row_binom_sum = 0, col_binom_sum = 0, tot_binom = 0;
    /* temp variable, RI, E(RI), and ARI*/
    double v, ri, eri, ari;

    /* traverse all cells in the table */
    for(int r=0; r<n_row; r++)
    {
        for(int c=0; c<n_col; c++)
        {
            v = (double)table[r][c];
            /* update sums */
            row_sum[r] += v;
            col_sum[c] += v;
            tot_sum    += v;
            /* update cells' binom sum*/
            cell_binom_sum += v * (v-1) / 2;
        }
    }
    /* calculate row_binom_sum */
    for (int r=0; r<n_row; r++) row_binom_sum += row_sum[r] * (row_sum[r]-1) / 2;
    /* calculate col_binom_sum */
    for (int c=0; c<n_col; c++) col_binom_sum += col_sum[c] * (col_sum[c]-1) / 2;
    /* calculate tot_binom */
    tot_binom = tot_sum * (tot_sum-1) / 2;
    
    /* calculate RI */
    ri = (2 * cell_binom_sum + tot_binom - row_binom_sum - col_binom_sum) / tot_binom;
    /* calculate E(RI) */
    eri = 1 + 2 * row_binom_sum * col_binom_sum / (tot_binom * tot_binom) - (row_binom_sum + col_binom_sum) / tot_binom;
    /* calculate ARI */
    ari = (ri - eri) / (1 - eri);

    return ari; 
}