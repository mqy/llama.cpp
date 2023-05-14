#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int ggml_compute_stage_flag_t;

#define NUM_BENCH 5

#define COMPUTE_STAGE_FLAG_VALID 1
#define COMPUTE_STAGE_FLAG_PARALLEL (1 << 1)

enum ggml_blas_type {
    GGML_BLAS_TYPE_ACCELERATE = 1, // https://developer.apple.com/accelerate
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

    ggml_compute_stage_flag_t cpu_stages[3];
    ggml_compute_stage_flag_t gpu_stages[3];

    struct ggml_mulmat_bench_nk *groups;
};

const char *ggml_get_blas_name(void);

void ggml_mulmat_write_bench_data(const struct ggml_mulmat_bench *bench,
                                  FILE *fp);

int ggml_mulmat_read_bench_data(struct ggml_mulmat_bench *bench, FILE *file);

int ggml_mulmat_estimate_time(const struct ggml_mulmat_bench *bench, int M,
                              int N, int K, int nth, bool is_cpu);

bool ggml_mulmat_bench_use_blas(const struct ggml_mulmat_bench *b, int M, int N,
                                int K, int nth);

#ifdef __cplusplus
}
#endif
