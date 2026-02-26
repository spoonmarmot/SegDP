#ifndef GLOBAL_H_
#define GLOBAL_H_

#include <stdio.h>

#define POS_INF (1e200)

#define NEG_INF (-1e200)

#define LOG_ZERO (-1e10)

#define EPS (1e-10)

#define DEFAULT_NMI_THRESHOLD (0.4)

#define CHECK_ALLOC(ptr) \
    do { \
        if ((ptr) == NULL) { \
            fprintf(stderr, "[error] NULL pointer %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/* a struct to store feature value, index, and its rank */
typedef struct {
    double value;
    int index;
    int rank;  /* == (int)value if the feature is categorical */ 
} value_rank_index;



#endif
