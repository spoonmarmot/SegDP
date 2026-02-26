#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <getopt.h>
#include <errno.h>
#include <time.h>
#include <limits.h>
#include "StratificationConfig.h"
#include "stratification.h"
#include "measures.h"
#include "global.h"
#include "utils.h"

struct option_with_usage 
{
    struct option opts[20];
    const char usages[20][300]; 
    int opt_size;

};

struct option_with_usage options = {
    .opts = {
        {"feature", required_argument, 0, 'f'},
        {"min-len", required_argument, 0, 'g'},
        {"max-k", required_argument, 0, 'k'},
        {"target-k", required_argument, 0, 't'},
        {"measure", required_argument, 0, 'm'},
        {"scaling", required_argument, 0, 's'},
        {"main-weight", required_argument, 0, 'a'},
        {"output", required_argument, 0, 'o'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    },
    .usages = {
        "-f, --feature FILE         Feature csv file with header [feature_type (1=numerical, 0=categorical), sample#1, ..., sample#N]. \n\n",
        "-g, --min-len INT          Minimum length of a segment. \n\n",
        "-k, --max-k INT            Maximum segment number. 1< [-k value] <= floor(sample_size/[-g value]) \n\n",
        "-t, --target-k INT         Target segment number. If not provided, then use the best number of segments. \n\n",
        "-m, --corr-thd FLOAT       The threshold for deciding whether an assistant feature is considered. \n\n",
        "-s, --scaling STR          Scaling method, can be [standard|whiten|ranknorm|minmax|robust|none]. Default is none. \n\n",
        "-a, --main-wei FLOAT|STR   The weight of main feature. If a single float then all features use the same values; if different weights are desired then input k floats separated by commas, which will be assigned to first k weights. If only want to assign the first feature, then \"{value},\" . \n\n",
        "-o, --output-fn FILE       File of discretized features (with original categorical ones). [-o value].dscinfo records extra information. If not provides, then ouput to [-f value].dsc and [-f value].dscinfo.\n\n",
        "-h, --help                 Print usage. \n\n"
    },
    .opt_size = 9
};

static void print_usage() 
{
    printf("Usage:\n------------------------------------------\n");
    for (int i=0; i<options.opt_size; i++)
    {
        printf(options.usages[i]);
    }
}

static char* get_optstr()
{
    int string_size = options.opt_size + 1, curr_idx;
    char *optstring;

    for (int i=0; i<options.opt_size; i++)
    {
        string_size += (options.opts[i].has_arg == required_argument);
    }

    optstring = (char *)malloc(string_size * sizeof(char));
    curr_idx = 0;
    for(int i=0; i<options.opt_size; i++)
    {
        optstring[curr_idx++] = options.opts[i].val;
        if (options.opts[i].has_arg == required_argument)
        {
            optstring[curr_idx++] = ':';
        }
    }
    optstring[curr_idx] = '\0';

    return optstring;
}

static double** read_features(const char *filepath, int *n_samples, int *m_num, int *m_cate)
{

    double **num_features, **cate_features, **features;
    int n_row = 0, n_col = 0, *strtol_end = NULL, r, c;
    char *line = NULL, *token = NULL, *strtod_end = NULL;
    size_t len = 0;
    ssize_t read;

    FILE *fp = fopen(filepath, "r");
    if (!fp)
    {
        fprintf(stdout, "[error] Failed to open the file: %s\n", filepath);
        exit(1);
    }

    /* get n_row */
    while (-1 != (read = getline(&line, &len, fp))) n_row++;
    if (n_row < 2)
    {
        fprintf(stdout, "[error] Less than 2 rows, i.e., header(1) + features(>=1).\n");
        exit(1);
    }

    /* get n_col from the header */
    rewind(fp);
    if(-1 != (read = getline(&line, &len, fp)))
    {
        if (line[0] == ',')
        {
            // Each call in the sequence searches the search target for 
            // the first character that is not contained in the separator 
            // string pointed to by delim
            n_col ++;
        }
        line[strcspn(line, "\n")] = '\0';
        token = strtok(line, ",");
        while (token != NULL)
        {
            n_col ++;
            token = strtok(NULL, ",");
        } 
    }
    if (n_col < 2)
    {
        fprintf(stdout, "[error] Less than 2 columns, i.e., feature_type(1) + values(>=1).\n");
        exit(1);
    }
    *n_samples = n_col - 1;

    /* continue to read feature values */
    *m_num = 0; 
    num_features = create_matrix_d(n_row - 1, *n_samples, 0.0);
    *m_cate = 0;
    cate_features = create_matrix_d(n_row - 1, *n_samples, 0.0);
    for (r=0; r<n_row-1; r++)
    {
        read = getline(&line, &len, fp); 
        line[strcspn(line, "\n")] = '\0';

        /* read the feature_type cell */
        token = strtok(line, ","); 
        errno = 0;
        int feature_type = (int) strtol(token, &strtol_end, 10);
        if (errno == ERANGE || errno == EINVAL || strtol_end == token 
            || feature_type * (1 - feature_type) != 0)
        {
            fprintf(stdout, "[error] Invalid feature type of %d-th feature", r+1);
            exit(1);
        }

        /* read feature values */
        if (feature_type == 1)
        {
            for (c=0; c<*n_samples; c++)
            {
                token = strtok(NULL, ",");
                num_features[*m_num][c] = strtold(token, &strtod_end);
                if (errno == ERANGE || errno == EINVAL || strtod_end == token)
                {
                    fprintf(stdout, "[error] Invalid numerical feature value of %d-th feature and %d- sample", r+1, c+1); 
                    exit(1);
                }
            }
            (*m_num) ++;
        }
        else
        {
            for (c=0; c<*n_samples; c++)
            {
                token = strtok(NULL, ",");
                cate_features[*m_cate][c] = strtold(token, &strtod_end);
                if (errno == ERANGE || errno == EINVAL || strtod_end == token)
                {
                    fprintf("[error] Invalid categorical feature value of %d-th feature and %d- sample", r+1, c+1); 
                    exit(1);
                }
            }
            (*m_cate) ++;
        }
    }
     
    /* combine num_features and cate_features to features */
    features = create_matrix_d(*m_num + *m_cate, *n_samples, NEG_INF);
    for (r=0; r<*m_num; r++)
    {
        memcpy(features[r], num_features[r], (*n_samples) * sizeof(double));
    }
    for (r=0; r<*m_cate; r++)
    {
        memcpy(features[*m_num + r], cate_features[r], (*n_samples) * sizeof(double));
    }

    for (r=0; r<n_row-1; r++)
    {
        free(num_features[r]);
        free(cate_features[r]);
    }
    free(num_features);
    free(cate_features);
    if (line != NULL) free(line);

    return features;

}

static int retrieve_int(char* optarg_pt, char opt)
{
    char *strtol_end;
    long long_receiver;
    
    long_receiver = strtol(optarg_pt, &strtol_end, 10);
    if (long_receiver >= INT_MAX ||  long_receiver <= INT_MIN || errno == EINVAL || strtol_end == optarg_pt)
    {
        fprintf(stdout, "[error] Invalid -%c argument, %s.\n", &opt, optarg_pt); exit(1);
    }

    return (int)long_receiver;
}

static double retrieve_double(char* optarg_pt, char opt)
{
    char *strtod_end;
    double double_receiver;

    errno = 0;
    double_receiver = strtod(optarg_pt, &strtod_end); 
    if (errno == ERANGE || errno == EINVAL || strtod_end == optarg_pt)
    {
        fprintf(stdout, "[error] Invalid -%c argument, %s.\n", &opt, optarg_pt); exit(1);
    }

    return double_receiver;
}

static double* retrieve_a_ws(char *optarg, int m1)
{
    double *a_ws = NULL;
    int optlen, fidx, comma_cnt;
    char *line = NULL, *token = NULL, *strtod_end = NULL, *s;

    a_ws = (double *)malloc(m1 * sizeof(double)); CHECK_ALLOC(a_ws);
    for (fidx=0; fidx<m1; fidx++) a_ws[fidx] = 1.0;
    
    optlen = (optarg == NULL ? 0 : strlen(optarg));
    
    // no weights given
    if (optarg == NULL || optlen == 0)
    {
        fprintf(stdout, "No main feature weights are given, Use default value 1.0 for all features. \n"); 
        return a_ws;
    }

    line = (char *)malloc((optlen+1) * sizeof(char)); CHECK_ALLOC(line);
    memset(line, '\0', optlen+1);
    memcpy(line, optarg, optlen);

    // count commas
    comma_cnt = 0;
    s = line;
    for(; *s!='\0'; s++)
    {
        if (*s == ',') comma_cnt ++;
    }

    // retrieve weights
    if (0 == comma_cnt)
    {
        token = strtok(line, ",");
        errno = 0;

        a_ws[0] = strtod(token, &strtod_end);
        if (errno == ERANGE || errno == EINVAL || strtod_end == token)
        {
            fprintf(stdout, "[error] Invalid weight value for the all main features (-a). \n"); 
            exit(1);
        }
        for (fidx=1; fidx<m1; fidx++) a_ws[fidx] = a_ws[0];
    }
    else
    {
        for (fidx=0; fidx<m1; fidx++)
        {
            token = ( fidx == 0 ? strtok(line, ",") : strtok(NULL, ",") );
            if (token == NULL) break;

            errno = 0;
            a_ws[fidx] = strtod(token, &strtod_end);
            if (errno == ERANGE || errno == EINVAL || strtod_end == token)
            {
                fprintf(stdout, "[error] Invalid weight value for the %d-th main feature (-a). \n", fidx+1); 
                exit(1);
            }
        }
    }


    free(line);
    return a_ws;
}

int main(int argc, char** argv)
{   
    double **features, running_time, nmi_thd = -1, *a_value, **b_value, **corr_weights, *indv_weights, *a_ws = NULL;
    int min_len = 1, max_k = 1, target_k = -1, n = 0, m_num = 0, m_cate = 0, opt, optindex, verbose = 0;
    int **output_features, curr_num_fea, r, c, fn_free = 0;
    char *opt_short = NULL, *feature_fn = NULL, *info_fn = NULL, *output_features_fn = NULL, *scaling = NULL, *a_ws_str = NULL;
    time_t start_time, end_time;
    rank_segments res;
    FILE *fp;
    
    /* read and parse program arguments*/
    opt_short = get_optstr();
    while(-1 != (opt = getopt_long(argc, argv, opt_short, options.opts, &optindex)))
    {
        switch(opt)
        {
            case 'f': 
                feature_fn = optarg;
                break;
            case 'g':
                min_len = retrieve_int(optarg, opt); 
                break;
            case 'k':
                max_k = retrieve_int(optarg, opt);
                break;
            case 't': 
                target_k = retrieve_int(optarg, opt);
                break;
            case 'm':
                nmi_thd = retrieve_double(optarg, opt);
                break;
            case 's':
                scaling = optarg;
                break;
            case 'a':
                a_ws_str = optarg;
                break;
            case 'o':
                output_features_fn = optarg;
                break;
            case '?':
            case 'h':
            default:
                print_usage(); free(opt_short); return 0;
        }
    }
    free(opt_short);
    if (feature_fn == NULL || min_len <=0 || max_k <=0)
    {
        fprintf(stdout, "[error] Missing required arguments.\n");
        print_usage(); return 1;
    }

    if (nmi_thd < 0)
    {
        nmi_thd = DEFAULT_NMI_THRESHOLD;
    }

    if (output_features_fn == NULL)
    {
        output_features_fn = (char *)calloc((strlen(feature_fn) + 50), sizeof(char)); CHECK_ALLOC(output_features_fn);
        output_features_fn[0] = '\0';
        strcat(output_features_fn, feature_fn);
        strcat(output_features_fn, ".dsc");
        
        info_fn = (char *)calloc((strlen(feature_fn) + 50), sizeof(char)); CHECK_ALLOC(info_fn);
        info_fn[0] = '\0';
        strcat(info_fn, feature_fn);
        strcat(info_fn, ".dscinfo");

        fn_free = 1;
    }
    else
    {
        info_fn = (char *)calloc((strlen(output_features_fn) + 50), sizeof(char)); CHECK_ALLOC(info_fn);
        info_fn[0] = '\0';
        strcat(info_fn, output_features_fn);
        strcat(info_fn, ".dscinfo");
    }


    /* read feature file */
    features = read_features(feature_fn, &n, &m_num, &m_cate);

    /* retrieve main feature weights */
    a_ws = retrieve_a_ws(a_ws_str, m_num);

    /* scaling */
    if ((scaling == NULL) || (strcasecmp(scaling, "none") == 0))
    {
        fprintf(stdout, "No scaling\n");
    }
    else if (strcasecmp(scaling, "standard") == 0)
    {
        fprintf(stdout, "Standardize ...\n"); 
        for (r=0; r<m_num; r++)
        {
            standardize(features[r], n, features[r]);
        }
    }
    else if (strcasecmp(scaling, "whiten") == 0)
    {
        fprintf(stdout, "Whitening numerical features using Cholesky decomposition ...\n");
        whitening_Cholesky(features, n, 0, m_num - 1);
    }
    else if (strcasecmp(scaling, "ranknorm") == 0)
    {
        fprintf(stdout, "Converting numerical features to rank-based normal scores ...\n"); 
        for (r=0; r<m_num; r++)
        {
            rank_based_normal_scores(features[r], n, features[r]);
        }
    }
    else if (strcasecmp(scaling, "minmax") == 0)
    {
        fprintf(stdout, "Min-Max scaling ...\n"); 
        for (r=0; r<m_num; r++)
        {
            minmax_scalar(features[r], n, features[r]);
        }
    }
    else if (strcasecmp(scaling, "robust") == 0)
    {
        fprintf(stdout, "Robust scaling ...\n"); 
        for (r=0; r<m_num; r++)
        {
            robust_scaling(features[r], n, 0.25, 0.75, features[r]);
        }
    }
    else 
    {
        fprintf(stdout, "Unknown scaling method! Remain data unchanged.\n"); 
    }

    // feature weights
    corr_weights = compute_feature_weights(features, nmi_thd, n, m_num, m_cate, max_k); 
    
    output_features = create_matrix_i(m_num + m_cate, n, -1);
    for (r=m_num; r<m_num + m_cate; r++)
    {
        for (c=0; c<n; c++)
        {
            output_features[r][c] = (int)(features[r][c]);
        }
    }

    /* discrete each numerical feature */
    a_value = (double *)malloc(n * sizeof(double)); CHECK_ALLOC(a_value);
    b_value = create_matrix_d(m_num + m_cate - 1, n, NEG_INF);
    indv_weights = (double *)malloc((m_num + m_cate - 1) * sizeof(double)); CHECK_ALLOC(indv_weights);

    running_time = 0.0;
    curr_num_fea = 0;
    memcpy(a_value, features[curr_num_fea], n * sizeof(double));
    for (r=1; r<m_num + m_cate; r++)
    {
        memcpy(b_value[r-1], features[r], n * sizeof(double));
        indv_weights[r-1] = corr_weights[curr_num_fea][r];
    }
    do
    {
        fprintf(stdout, "Stratifying %d-th numerical feature ...\n", curr_num_fea + 1);
        fflush(stdout);
        time(&start_time);

        res = stratify(a_value, b_value, a_ws[curr_num_fea], indv_weights, n, m_num - 1, m_cate, max_k, min_len, target_k);

        time(&end_time);
        running_time += difftime(end_time, start_time);

        /* save discretization result */
        memcpy(output_features[curr_num_fea], res.a_strata, n * sizeof(int));

        fp = fopen(info_fn, "a"); CHECK_ALLOC(fp);
        print_rank_segments(fp, res);
        fclose(fp); 
        destroy_rank_segments(res);

        if (curr_num_fea + 1 >= m_num) break;
        else
        {
            memcpy(b_value[curr_num_fea], a_value, n * sizeof(double));
            memcpy(a_value, features[curr_num_fea + 1], n * sizeof(double));
            curr_num_fea ++;
            // update weights
            for (r=0; r<m_num+m_cate; r++)
            {
                if (r == curr_num_fea) continue;
                else if (r < curr_num_fea) indv_weights[r] = corr_weights[curr_num_fea][r];
                else indv_weights[r-1] = corr_weights[curr_num_fea][r];
            }
        }
    } while (1);

    fp = fopen(info_fn, "a"); CHECK_ALLOC(fp);
    fprintf(fp, "\nTotal running time: %.f seconds\n", running_time);
    fclose(fp);

    /* save discretization result */
    fp = fopen(output_features_fn, "w"); CHECK_ALLOC(fp);
    for (r=0; r<m_num + m_cate; r++)
    {
        for (c=0; c<n; c++)
        {
            fprintf(fp, "%d%s", output_features[r][c], (c==n-1)?"\n":",");
        }
    }
    fclose(fp);



    /* clean up */
    free(a_ws);
    free(info_fn);
    if (fn_free == 1)
    {
        free(output_features_fn);
    }
    free(indv_weights);
    free(a_value);
    for(r=0; r<m_num + m_cate -1; r++)
        free(b_value[r]);
    free(b_value);
    for (r=0; r<m_num + m_cate; r++)
    {
        free(features[r]);
        free(output_features[r]);
    }
    free(features);
    free(output_features);
    
    for (r=0; r<m_num; r++)
    {
        free(corr_weights[r]);
    }
    free(corr_weights);

    return 0;
}