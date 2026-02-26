#include "stratification.h"
#include "linalg.h"
#include "utils.h"
#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void output_features(double *main_feature, double **assistant_features, int SAMPLE_SZ, int FEATURE_SZ, double *feature_weight, int *is_num_af, char *fn)
{
    char line[3000];
    char *DELIMITER = ";";
    FILE *fp = fopen(fn, "a");

    // header
    fprintf(fp, "feature_type%sfeature_weight", DELIMITER);
    for (int c=0; c<SAMPLE_SZ; c++)
    {
        fprintf(fp, "%ssample#%d", DELIMITER, c);
    }
    fprintf(fp, "\n");
    // main feature
    fprintf(fp, "-%s-", DELIMITER);
    for (int c=0; c<SAMPLE_SZ; c++)
    {
        fprintf(fp, "%s%f", DELIMITER, main_feature[c]);
    }
    fprintf(fp, "\n");
    // assistant features
    for(int r=0; r<FEATURE_SZ; r++)
    {
        fprintf(fp, "%d%s%f", is_num_af[r], DELIMITER, feature_weight==NULL?1.0:feature_weight[r]);
        for (int c=0; c<SAMPLE_SZ; c++)
        {
            fprintf(fp, "%s%f", DELIMITER, assistant_features[r][c]);
        }
        fprintf(fp, "\n");
    }
    fflush(fp);
    fclose(fp);
} 

// void simple_case()
// {
//     int n = 10;
//     double a_value[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9 ,10};
//     double **b_value;

//     int min_len = 3;
//     int K_max = 3;
//     rank_segments res;
    
//     b_value = (double **)malloc(2 * sizeof(double *)); CHECK_ALLOC(b_value);
//     for (int i=0; i<2; i++) {b_value[i] = (double *)malloc(10 * sizeof(double)); CHECK_ALLOC(b_value[i]);}
//     b_value[0][0] =    1; b_value[1][0] =  30.1;
//     b_value[0][1] =  1.2; b_value[1][1] =  30.2;
//     b_value[0][2] =  0.9; b_value[1][2] =  30.1;
//     b_value[0][3] =  1.1; b_value[1][3] =  29.8;
//     b_value[0][4] =  0.8; b_value[1][4] =  29.7;
//     b_value[0][5] = 30.1; b_value[1][5] =     1;
//     b_value[0][6] = 30.2; b_value[1][6] =   1.2;
//     b_value[0][7] = 30.1; b_value[1][7] =   0.9;
//     b_value[0][8] = 29.8; b_value[1][8] =   1.1;
//     b_value[0][9] = 29.7; b_value[1][9] =   0.8;

//     // whitening_Cholesky(b_value, n, 0, 1);
//     rank_based_normal_scores(a_value, 10, a_value);
//     rank_based_normal_scores(b_value[0], 10, b_value[0]);
//     rank_based_normal_scores(b_value[1], 10, b_value[1]);


//     res = stratify_byCH(a_value, b_value, n, 2, 0, K_max, min_len, -1);
//     print_rank_segments(stdout, res);
//     destroy_rank_segments(res);

//     res = stratify_byBICg(a_value, b_value, n, 2, 0, K_max, min_len, -1);
//     print_rank_segments(stdout, res);
//     destroy_rank_segments(res);

//     res = stratify(a_value, b_value, n, 2, 0, K_max, min_len, -1);
//     print_rank_segments(stdout, res);
//     destroy_rank_segments(res);

//     free(b_value[0]);
//     free(b_value[1]);
//     free(b_value);
// }


void efew_case()
{
    int n = 300;
    int m = 10;
    int r, c;
    double **value, **corr_weights, *weights, *b_values;

    int min_len = 10;
    int K_max = 30;
    rank_segments res;

    value = create_matrix_d(m+1, n, NEG_INF);
    for (c=0; c<n; c++)
    {
        value[0][c] = rand_normal(5, 3);
        for (r=1; r<=m; r++)
        {
            value[r][c] = rand_normal(r, r);
        }
    }
    corr_weights = compute_feature_weights(value, 0.4, n, 11, 0, K_max);
    weights = (double *)malloc(m * sizeof(double)); CHECK_ALLOC(weights);
    for (r=0; r<m; r++)
    {
        weights[r] = corr_weights[0][r+1];
    }


    for (int r=0; r<=m; r++)
    {
        rank_based_normal_scores(value[r], n, value[r]);
    }

    res = stratify(value[0], &(value[1]), 1, weights, n, m, 0, K_max, min_len, -1);
    print_rank_segments(stdout, res);

}

// void simple_categorical_case()
// {
//     int n = 10;
//     double a_value[10] = {3, 1, 10, 4, 2, 9, 5, 8, 7, 6};
//     double b_value[10] = {0, 0,  2, 1, 0, 2, 1, 2, 2, 1};
//     // double a_value[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//     // double b_value[10] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2 };
//     int min_len = 2;
//     int K_max = 5;
//     rank_segments res;

//     res = stratify(a_value, b_value, n, 0, K_max, min_len, -1, adjusted_rand_index);

//     print_rank_segments(res);
// }

// void random_experiment()
// {
//     double *main_feature, **assistant_feature, min_ari, *feature_weight, running_sec;
//     int min_grp_sz, top_k, *is_num_af, is_descending, true_grp_sz, SAMPLE_SZ, FEATURE_SZ;
//     time_t start, end;
//     rank_segments res;


//     // fixed a random seed
//     srand(20201010);

//     // sample size and feature size
//     SAMPLE_SZ = 2000;
//     FEATURE_SZ = 14;

//     // allocate memory
//     main_feature = (double *)calloc(SAMPLE_SZ, sizeof(double));
//     assistant_feature = (double **)calloc(FEATURE_SZ, sizeof(double *));
//     for(int i=0; i<FEATURE_SZ; i++)
//         assistant_feature[i] = (double *)calloc(SAMPLE_SZ, sizeof(double));
//     is_num_af = (int *)calloc(FEATURE_SZ, sizeof(int));
//     for(int i=0; i<FEATURE_SZ; i++)
//         is_num_af[i] = 1;
//     true_grp_sz = 90;

//     min_grp_sz = 700;

//     for(int i=0; i<SAMPLE_SZ; i++)
//     {
//         main_feature[i] = (double)rand() / (double)RAND_MAX;
//         for(int j=0; j<FEATURE_SZ; j++)
//         {
//             assistant_feature[j][i] = (double)rand() / (double)RAND_MAX;
//         }
//     }

//     output_features(main_feature, assistant_feature, SAMPLE_SZ, FEATURE_SZ, NULL, is_num_af, "features_20201010.csv");

//     whitening_Cholesky(assistant_feature, SAMPLE_SZ, 0, FEATURE_SZ-1);
    
//     time(&start);
//     res = stratify_byCH(main_feature, assistant_feature, SAMPLE_SZ, FEATURE_SZ, 0, (int)floor(SAMPLE_SZ/min_grp_sz), min_grp_sz, -1);
//     print_rank_segments(stdout, res);
//     time(&end);
//     destroy_rank_segments(res);
//     running_sec = difftime(end, start);
//     printf("Running time: %.f seconds\n", running_sec);

//     time(&start);
//     res = stratify_byBICg(main_feature, assistant_feature, SAMPLE_SZ, FEATURE_SZ, 0, (int)floor(SAMPLE_SZ/min_grp_sz), min_grp_sz, -1);
//     print_rank_segments(stdout, res);
//     time(&end);
//     destroy_rank_segments(res);
//     running_sec = difftime(end, start);
//     printf("Running time: %.f seconds\n", running_sec);

//     time(&start);
//     res = stratify(main_feature, assistant_feature, SAMPLE_SZ, FEATURE_SZ, 0, (int)floor(SAMPLE_SZ/min_grp_sz), min_grp_sz, -1);
//     print_rank_segments(stdout, res);
//     time(&end);
//     destroy_rank_segments(res);
//     running_sec = difftime(end, start);
//     printf("Running time: %.f seconds\n", running_sec);

//     free(main_feature);
//     for (int i=0; i<FEATURE_SZ; i++)
//         free(assistant_feature[i]);
//     free(assistant_feature);
//     free(is_num_af);

// }

int main(int argc, void** argv)
{

    // printf("--Simple case:\n");
    // simple_case();

    printf("--EFEW case:\n");
    efew_case();
    
    // printf("\n--Simple categorical case:\n");
    // simple_categorical_case();
    
    // printf("\n--Random data:\n");
    // random_experiment();

    return 0;
}
