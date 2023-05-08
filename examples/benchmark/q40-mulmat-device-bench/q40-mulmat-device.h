#include "ggml.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int compute_stage_flag_t;

#define NUM_BENCH 5
#define COMPUTE_STAGE_FLAG_VALID 1
#define COMPUTE_STAGE_FLAG_NEED_WORKER (1 << 1)

struct bench_data_item {
    int M;

    int cpu_time[3];
    int gpu_time[3];

    int cpu_records[3][NUM_BENCH];
    int gpu_records[3][NUM_BENCH];
};

struct bench_data_shape {
    int N;
    int K;

    struct bench_data_item *items;
};

// top bench data to write/read to/from file.
struct bench_data {
    int version;

    char model[4];     // 7B | 13B
    char gpu_impl[20]; // ACCELERATE, OPENBLAS, CUBLAS
    int n_shapes;
    int m_step;
    int num_m;

    // bit 0: valid, bit 1: can parallel
    // TODO: define macros.
    compute_stage_flag_t cpu_stages[3];
    compute_stage_flag_t gpu_stages[3];

    struct bench_data_shape *shapes;
};

struct model_nk_shape {
    int N;
    int K;
};

int read_bench_data(struct bench_data *bd, FILE *file);

void write_bench_data(struct bench_data *bd, FILE *fp);

int estimate_time(struct bench_data *bd, int M, int N, int K, int nth,
                  bool is_cpu);

enum ggml_device_type choose_device(struct bench_data *bd, int M, int N, int K,
                                    int nth);

#ifdef __cplusplus
}
#endif
