#include "ggml.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int ggml_compute_stage_flag_t;

#define NUM_BENCH 5
#define COMPUTE_STAGE_FLAG_VALID 1
#define COMPUTE_STAGE_FLAG_NEED_WORKER (1 << 1)

struct ggml_mulmat_bench_data_item {
    int M;

    int cpu_time[3];
    int gpu_time[3];

    int cpu_records[3][NUM_BENCH];
    int gpu_records[3][NUM_BENCH];
};

struct ggml_mulmat_bench_data_shape {
    int N;
    int K;

    struct ggml_mulmat_bench_data_item *items;
};

// top bench data to write/read to/from file.
struct ggml_mulmat_bench_data {
    int version;

    char model[4];     // 7B | 13B
    char gpu_impl[20]; // ACCELERATE, OPENBLAS, CUBLAS
    int n_shapes;
    int m_step;
    int num_m;

    ggml_compute_stage_flag_t cpu_stages[3];
    ggml_compute_stage_flag_t gpu_stages[3];

    struct ggml_mulmat_bench_data_shape *shapes;
};

enum ggml_device_type {
    GGML_DEVICE_AUTO = 0,
    GGML_DEVICE_CPU,
    GGML_DEVICE_GPU,
};

void ggml_mulmat_write_bench_data(struct ggml_mulmat_bench_data *bd, FILE *fp);
int ggml_mulmat_read_bench_data(struct ggml_mulmat_bench_data *bd, FILE *file);
int ggml_mulmat_estimate_time(struct ggml_mulmat_bench_data *bd, int M, int N, int K, int nth,
                  bool is_cpu);
enum ggml_device_type ggml_mulmat_choose_device(struct ggml_mulmat_bench_data *bd, int M, int N, int K,
                                    int nth);

#ifdef __cplusplus
}
#endif
