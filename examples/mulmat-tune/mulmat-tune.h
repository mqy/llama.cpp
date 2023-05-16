#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_BENCH 3

// task flag: byte 0..2 for task stages; byte 3 for extensions (use blas, etc.)

// GGML_TASK_FLAG_T1: total 1 task, workers keep spinning.
// GGML_TASK_FLAG_TN: total N tasks, all workers involve in computing.
// GGML_TASK_FLAG_T1_WAIT: total 1 task, workers go waiting.
#define GGML_TASK_FLAG_T1 0x1
#define GGML_TASK_FLAG_T1_WAIT 0x2
#define GGML_TASK_FLAG_TN 0x3

void ggml_task_flag_set_blas(int32_t *flag, int8_t value);

int8_t ggml_task_flag_get_blas(int32_t flag);

void ggml_task_flag_set(int32_t *flag, int stage, int8_t value);

int8_t ggml_task_flag_get(int32_t v, int stage);

enum ggml_blas_type {
    GGML_BLAS_TYPE_ACCELERATE = 0, // https://developer.apple.com/accelerate
    GGML_BLAS_TYPE_OPENBLAS,       // https://www.openblas.net/
    GGML_BLAS_TYPE_CLBLAST, // https://cnugteren.github.io/clblast/clblast.html
    GGML_BLAS_TYPE_CUBLAS,  // https://developer.nvidia.com/cublas

    GGML_BLAS_TYPE_COUNT
};

struct ggml_mulmat_tune_m {
    int M;

    int cpu_only_time[3];
    int use_blas_time[3];

    // These are not save to data file.
    int cpu_only_records[3][NUM_BENCH];
    int use_blas_records[3][NUM_BENCH];
};

struct ggml_mulmat_tune_nk {
    int N;
    int K;

    struct ggml_mulmat_tune_m *items;
};

struct ggml_mulmat_tune {
    int version;

    char model[4];      // 7B | 13B
    char blas_name[16]; // see `ggml_blas_names`
    char q_type_name[8];
    int q_type;
    int n_groups;
    int step_m;
    int num_m;

    uint32_t cpu_only_stages[3];
    uint32_t use_blas_stages[3];

    struct ggml_mulmat_tune_nk *groups;
};

struct ggml_mulmat_tune_time_stats {
    int cpu_only_stages[3];
    int cpu_only_total;

    int use_blas_stages[3];
    int use_blas_total;
};

void ggml_mulmat_write_tune_data(const struct ggml_mulmat_tune *tune, FILE *fp);

int ggml_mulmat_read_tune_data(struct ggml_mulmat_tune *tune, FILE *file);

int ggml_mulmat_estimate_time(const struct ggml_mulmat_tune *tune, int M, int N,
                              int K, int nth, bool cpu_only);

// return 0: ok, -1: M out of range or no data.
int ggml_mulmat_tune_time_stats(const struct ggml_mulmat_tune *b, int M, int N,
                                int K, int nth,
                                struct ggml_mulmat_tune_time_stats *time_stats);

const char *ggml_get_blas_name(void);

#ifdef __cplusplus
}
#endif
