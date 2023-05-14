#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_BENCH 5

// task flag: byte 0..2 for task stages; byte 3 for extensions (use blas, etc.)

#define GGML_TASK_FLAG_1_THREAD       0x1
#define GGML_TASK_FLAG_1_THREAD__WAIT 0x2
#define GGML_TASK_FLAG_N_THREADS      0x3

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

struct ggml_mulmat_bench_m {
    int M;

    int cpu_time[3];
    int gpu_time[3];

    // These are not save to data file.
    int cpu_records[3][NUM_BENCH];
    int gpu_records[3][NUM_BENCH];
};

struct ggml_mulmat_bench_nk {
    int N;
    int K;

    struct ggml_mulmat_bench_m *items;
};

struct ggml_mulmat_bench {
    int version;

    char model[4];      // 7B | 13B
    char blas_name[16]; // see `ggml_blas_names`
    int n_groups;
    int m_step;
    int num_m;

    uint32_t cpu_stages[3];
    uint32_t gpu_stages[3];

    struct ggml_mulmat_bench_nk *groups;
};

void ggml_mulmat_write_bench_data(const struct ggml_mulmat_bench *bench,
                                  FILE *fp);

int ggml_mulmat_read_bench_data(struct ggml_mulmat_bench *bench, FILE *file);

int ggml_mulmat_estimate_time(const struct ggml_mulmat_bench *bench, int M,
                              int N, int K, int nth, bool is_cpu);

bool ggml_mulmat_bench_use_blas(const struct ggml_mulmat_bench *b, int M, int N,
                                int K, int nth);

const char *ggml_get_blas_name(void);

#ifdef __cplusplus
}
#endif
