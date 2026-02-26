#include "stratification.h"
#include "utils.h"
#include "linalg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

rank_segments stratify(
    const double *a_value,  /* main feature value */
    const double **b_value, /* assistant feature values, shape=(m1+m2, n) with first m1 rows being numerical features and last m2 rows being categorical features. If categorical then its unique set must be continuous int starting from 0 */
    const double a_ws,      /* weight of the main feature */
    const double *weight,   /* weights of each assistant feature */
    int n,                  /* number of samples */
    int m1,                 /* number of numerical features */
    int m2,                 /* number of categorical features */
    int K_max,              /* maximum number of segments */
    int min_len,            /* minimum number of a segment length */
    int target_k            /* the desired number of segments, if <0 then the function return the best results */
)
{
    /*-------------------------------------------------------------------*/
    /* main feature variables */
    value_rank_index *a_vri;
    int *a_rank;

    /* cache variables for assistant features */
    double **num_b_value, **cate_b_value, **W, **N, **b_prefix_sum, **b_prefix_sq_sum, *a_prefix_sum, *a_prefix_sq_sum; 
    cate_fea_prefix_count *b_prefix_cf_pc;
 
    /* DP variables */
    // backtrack variables
    int prev_seghead;
    // while-DP variables (early_stop_steps temporarily disabled)
    int r_min, l_min, l_max, **last_segment_head, early_stop_steps = 10, early_stop_flag; 
    double **dp, **dp_N, curr_W, curr_N, b_W_penalty, b_N_penalty;
    
    /* other tool variables */
    int sidx1, sidx2, fidx1, fidx2, k, r, l, c;
    double tmp_d, tmp_d2;

    /* result variable */
    rank_segments res;
    /*-------------------------------------------------------------------*/
    
    
    /*-------------------------------------------------------------------*/
    /* check K_max and min_len */
    if (K_max > (int)floor(n / min_len) || K_max <= 1)
    {
        fprintf(stdout, "[error] invalid K_max!\n"); exit(1);
    }

    if (min_len > (n/2) || min_len < 1)
    {
        fprintf(stdout, "[error] invalid min_len!\n"); exit(1);
    }

    if (target_k > K_max || target_k == 1 )
    {
        target_k = -1; 
        fprintf(stdout, "[warning] invalid target_k. ignored.\n");
    }
    /*-------------------------------------------------------------------*/


    /*-------------------------------------------------------------------*/
    /* re-order feature a and b according to a_value ranks */
    a_rank = value_to_rank(a_value, n);
    a_vri = (value_rank_index *)malloc(n * sizeof(value_rank_index)); CHECK_ALLOC(a_vri);
    for (sidx1=0; sidx1<n; sidx1++)
    {
        a_vri[a_rank[sidx1]].value = a_value[sidx1];
        a_vri[a_rank[sidx1]].index = sidx1;
        a_vri[a_rank[sidx1]].rank = a_rank[sidx1];
    }
    
    num_b_value = create_matrix_d(m1, n, NEG_INF);
    for (fidx1=0; fidx1<m1; fidx1++)
    {
        for (sidx1=0; sidx1<n; sidx1++)
        {
            num_b_value[fidx1][a_rank[sidx1]] = b_value[fidx1][sidx1];
        }
    }

    cate_b_value = create_matrix_d(m2, n, NEG_INF);
    for (fidx1=0; fidx1<m2; fidx1++)
    {
        for (sidx1=0; sidx1<n; sidx1++)
        {
            cate_b_value[fidx1][a_rank[sidx1]] = b_value[m1+fidx1][sidx1];
        }
    }
    /*-------------------------------------------------------------------*/


    /*-------------------------------------------------------------------*/
    /* pre-calculation */ 
    
    /* numerical features */
    // prefix_sum 
    a_prefix_sum = (double *) malloc(n * sizeof(double)); CHECK_ALLOC(a_prefix_sum);
    a_prefix_sum[0] = a_vri[0].value;
    for (sidx1=1; sidx1<n; sidx1++)
    {
        a_prefix_sum[sidx1] = (a_vri[sidx1].value + a_prefix_sum[sidx1-1]);
    } 
    b_prefix_sum = comp_prefix_sum(num_b_value, 0, n-1, 0, m1-1);
    // prefix_sq_sum
    a_prefix_sq_sum = (double *) malloc(n * sizeof(double)); CHECK_ALLOC(a_prefix_sq_sum);
    a_prefix_sq_sum[0] = a_vri[0].value * a_vri[0].value;
    for (sidx1=1; sidx1<n; sidx1++)
    {
        a_prefix_sq_sum[sidx1] = (a_vri[sidx1].value * a_vri[sidx1].value + a_prefix_sq_sum[sidx1-1]);
    } 
    b_prefix_sq_sum = comp_prefix_sq_sum(num_b_value, 0, n-1, 0, m1-1);

    // penalty term of numerical features, i.e., b_W_penalty * log(n) for each segment
    b_W_penalty = 2 * m1 + 2;

    /* categorical features */
    b_prefix_cf_pc = (cate_fea_prefix_count *)malloc(m2 * sizeof(cate_fea_prefix_count)); CHECK_ALLOC(b_prefix_cf_pc);
    b_N_penalty = 0.0;
    for (fidx1=0; fidx1<m2; fidx1++)
    {
        // n_samples
        b_prefix_cf_pc[fidx1].n_samples = n;

        // n_unique
        b_prefix_cf_pc[fidx1].n_unique = -1;
        for (sidx1=0; sidx1<n; sidx1++)
        {
            if (b_prefix_cf_pc[fidx1].n_unique < (int)(cate_b_value[fidx1][sidx1]))
            {
                b_prefix_cf_pc[fidx1].n_unique = (int)(cate_b_value[fidx1][sidx1]);
            }
        }
        b_prefix_cf_pc[fidx1].n_unique += 1; // since starting from 0
        // penalty term of categorical features, i.e., b_N_penalty * log(n) for each segment
        b_N_penalty += (b_prefix_cf_pc[fidx1].n_unique - 1);

        // prefix count
        b_prefix_cf_pc[fidx1].prefix_count = create_matrix_d(b_prefix_cf_pc[fidx1].n_unique, n, 0); 
        
        c = (int)(cate_b_value[fidx1][0]); 
        b_prefix_cf_pc[fidx1].prefix_count[c][0] = 1.0; 
        for (sidx1=1; sidx1<n; sidx1++)
        {
            for (c=0; c<b_prefix_cf_pc[fidx1].n_unique; c++)
            {
                b_prefix_cf_pc[fidx1].prefix_count[c][sidx1] = b_prefix_cf_pc[fidx1].prefix_count[c][sidx1-1];
            }
            c = (int)(cate_b_value[fidx1][sidx1]);
            b_prefix_cf_pc[fidx1].prefix_count[c][sidx1] += 1.0;

        }
    }
    /*-------------------------------------------------------------------*/
   

    /*-------------------------------------------------------------------*/
    /* dynamic programming */ 
    // dp
    dp = create_matrix_d(K_max+1, n, POS_INF);
    // head of the last segment
    last_segment_head = create_matrix_i(K_max+1, n, -1);  
    // fitness term of numerical features  
    W = create_matrix_d(n, n, NEG_INF);
    // fitness term of categorical features  
    N = create_matrix_d(n, n, NEG_INF);
    
    // update DP table
    for (k=1; k<=K_max; k++)
    {

        if (target_k > 1 && k > target_k) break;

        r_min = k * min_len - 1; 
        if (r_min >= n) break;

        for (r=r_min; r<n; r++)
        {
            l_min = (k-1)*min_len;
            l_max =  (k == 1 ? l_min : (r-min_len+1));
            for(l=l_min; l<=l_max; l++)
            {   
                // can not divide elements 0...s-1 into k-1 segments
                if (k > 1 && dp[k-1][l-1] >= POS_INF/2) 
                    continue;
                
                // main feature and numerical features
                curr_W = 0.0;
                if (m1 > 0 || a_ws > 0)
                {
                    if (W[l][r] <= NEG_INF/2)
                    {
                        W[l][r] = a_ws * W1D(a_prefix_sum, a_prefix_sq_sum, l, r);
                        W[l][r] = check_zero(W[l][r]);

                        /**  
                         * When W[l][r] < (LOG_ZERO+EPS) and a_ws!=0, a[l...r] are extremely
                         * close to each other (basically identical) and assistant features 
                         * should be ignored. 
                         */
                        if (W[l][r] > LOG_ZERO || check_zero(a_ws) == 0) 
                        {
                            for (fidx1=0; fidx1<m1; fidx1++)
                            {
                                W[l][r] += weight[fidx1] * W1D(b_prefix_sum[fidx1], b_prefix_sq_sum[fidx1], l, r);
                            } 
                        }
                        W[l][r] = check_zero(W[l][r]);
                    }          
                    curr_W =  W[l][r] + b_W_penalty * log(n);  
                }
 
                // categorical features
                curr_N = 0.0;
                if (m2 > 0)
                {
                    if (N[l][r] <= NEG_INF/2)
                    {
                        N[l][r] = 0.0;

                        /**  
                         * When W[l][r] < (LOG_ZERO+EPS) and a_ws!=0, a[l...r] are extremely
                         * close to each other (basically identical) and assistant features 
                         * should be ignored. 
                         */
                        if (W[l][r] > LOG_ZERO || check_zero(a_ws) == 0) 
                        {
                            for (fidx1=0; fidx1<m2; fidx1++)
                            {
                                if (weight[m1+fidx1] > 0)  // N1D is computational expensive
                                {
                                    N[l][r] += weight[m1+fidx1] * N1D(b_prefix_cf_pc[fidx1], l, r);
                                }
                            }
                        }
                        N[l][r] = check_zero(N[l][r]);
                    }
                    curr_N = N[l][r] + b_N_penalty * log(n);
                }


                // compare with the best
                tmp_d = curr_W + curr_N + (k > 1 ? dp[k-1][l-1] : 0.0);
                if (tmp_d <= dp[k][r])
                {
                    dp[k][r] = tmp_d;
                    last_segment_head[k][r] = l;
                }
            }
        }
        // early stop
        // if (k - early_stop_steps + 1 >= 2 && dp[k-early_stop_steps+1][n-1] <= dp[k][n-1])
        // {
        //     early_stop_flag = 1;
        //     for (int es = 0; es < early_stop_steps-1; es++)
        //     {
        //         if (dp[k-1-es][n-1]>dp[k-es][n-1])
        //         {
        //             early_stop_flag = 0; break;
        //         }
        //     }
        //     if (early_stop_flag) break;
        // }

    }
    /*-------------------------------------------------------------------*/
  

    /*-------------------------------------------------------------------*/
    /* choose the best_K from dp[?][n-1] */
    // update best result
    for (int k=2; k<=(target_k<0? K_max:target_k); k++)
    {
        fprintf(stdout, "dp[%d][%d]=%g\n", k, n-1, dp[k][n-1]);
        fflush(stdout);
    }
    res.score = - NEG_INF;
    res.n_segs = -1;
    if (target_k < 0)
    {
        fprintf(stdout, "Select best dp\n");
        fflush(stdout);
        for (int k=2; k<=K_max; k++)
        {
            // calculate BIC with locally identical variance
            tmp_d = dp[k][n-1];
            if (tmp_d < res.score)
            {
                res.score = tmp_d;
                res.n_segs = k;
            }
        }
        fprintf(stdout, "Best k=%d\n", res.n_segs);
        fflush(stdout);

    }
    else
    {
        res.score = dp[target_k][n-1];
        res.n_segs = target_k;
    } 
    /*-------------------------------------------------------------------*/

    /*-------------------------------------------------------------------*/
    /* backtrack to get segment details*/
    if (res.n_segs > 0)
    {
        res.seg_heads = (int *)malloc(res.n_segs * sizeof(int)); CHECK_ALLOC(res.seg_heads);
        res.seg_heads[res.n_segs - 1] = last_segment_head[res.n_segs][n-1];
        prev_seghead = res.seg_heads[res.n_segs - 1];
        for (int k=res.n_segs-1; k>=1; k--)
        {
            res.seg_heads[k-1] = last_segment_head[k][prev_seghead-1];
            prev_seghead = res.seg_heads[k-1];
        }
    }
    else
    {
        res.seg_heads = NULL;
    }
    /*-------------------------------------------------------------------*/
    

    /*-------------------------------------------------------------------*/
    /* discretize a_values using a_vri */
    res.n_samples = n;
    res.a_strata = (int *)malloc(n * sizeof(int)); CHECK_ALLOC(res.a_strata);
    for (int i=0; i<n; i++)
    {
        for (int k=0; k<res.n_segs; k++)
        {
            if (a_vri[i].rank >= res.seg_heads[k] && (k==res.n_segs-1 || a_vri[i].rank < res.seg_heads[k+1]))
            {
                res.a_strata[a_vri[i].index] = k;
                break;
            }
        }
    }
    /*-------------------------------------------------------------------*/

    /* clean up */
    free(a_rank);
    free(a_vri);
    for (fidx1=0; fidx1<m1; fidx1++)
    {
        free(num_b_value[fidx1]);
    }
    free(num_b_value);    
    
    for (fidx1=0; fidx1<m2; fidx1++)
    {
        free(cate_b_value[fidx1]);
    }
    free(cate_b_value);   


    for(sidx1=0; sidx1<n; sidx1++)
    {
        free(W[sidx1]);
        free(N[sidx1]);
    } 
    free(W);
    free(N);

    for (fidx1=0; fidx1<m2; fidx1++)
    {
        destroy_cate_fea_prefix_count(b_prefix_cf_pc[fidx1]);
    }
    free(b_prefix_cf_pc);

    for(fidx1=0; fidx1<m1; fidx1++)
    {
        free(b_prefix_sum[fidx1]);
        free(b_prefix_sq_sum[fidx1]);
    }
    free(b_prefix_sum);
    free(b_prefix_sq_sum);
    free(a_prefix_sum);
    free(a_prefix_sq_sum);

    for (k=0; k<=K_max; k++) 
    {
        free(dp[k]); 
        free(last_segment_head[k]);
    }
    free(dp);
    free(last_segment_head);

    return res;
} 

