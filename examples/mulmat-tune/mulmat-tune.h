#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_BENCH 3

#define GGML_MULMAT_TUNE_MAX_PROFILES 8
#define GGML_MULMAT_TUNE_MAX_SHAPES 4

struct ggml_task_conf {
    int backend; // enum ggml_backend
    bool parallel;
    bool wait;
};

struct ggml_mulmat_tune_m {
    int M;

    int stages_time[3];
};

struct ggml_mulmat_tune_shape {
    int N;
    int K;
};

struct ggml_mulmat_tune {
    int version;

    char model[4];

    int gpu_backend; // enum ggml_backend
    char gpu_backend_name[16];

    int type; // enum ggml_type
    char type_name[8];

    int m_step;
    int m_num;

    int n_shapes;
    struct ggml_mulmat_tune_shape shapes[GGML_MULMAT_TUNE_MAX_SHAPES];

    int n_profiles;
    struct ggml_task_conf conf[GGML_MULMAT_TUNE_MAX_PROFILES][3];

    // n_shapes * m_num * n_profiles
    struct ggml_mulmat_tune_m *items;
};

struct ggml_mulmat_tune_profile_time {
    struct ggml_task_conf *task_conf;
    int stage_time[3];
    int total_time;
};

struct ggml_mulmat_tune_time_stats {
    int n_profiles;
    struct ggml_mulmat_tune_profile_time
        profile_time[GGML_MULMAT_TUNE_MAX_PROFILES];
};

void ggml_task_conf_format_name(struct ggml_task_conf *conf, char *buf,
                                int buf_len);

int ggml_mulmat_tune_validate(struct ggml_mulmat_tune *tune,
                              const char *model_name, int type);

int ggml_mulmat_tune_setup_model(struct ggml_mulmat_tune *tune,
                                 const char *model);

void ggml_mulmat_tune_setup_task_conf(struct ggml_mulmat_tune *tune);

int ggml_mulmat_tune_write_data(const struct ggml_mulmat_tune *tune, FILE *fp);

int ggml_mulmat_tune_read_data(struct ggml_mulmat_tune *tune, FILE *file);

// return 0: ok, -1: M out of range or no data.
int ggml_mulmat_tune_time_stats(const struct ggml_mulmat_tune *b, int M, int N,
                                int K, int nth,
                                struct ggml_mulmat_tune_time_stats *time_stats);

// returns enum ggml_backend
int ggml_get_backend(void);

const char *ggml_get_backend_name(void);

#ifdef __cplusplus
}
#endif
