#include <string.h>

#include "examples/mulmat-tune/mulmat-tune.h"

inline void ggml_task_flag_set_blas(int32_t *flag, int8_t value) {
    *flag |= (value << 24);
}

inline int8_t ggml_task_flag_get_blas(int32_t flag) {
    return (flag >> 24) & 0xFF;
}

inline void ggml_task_flag_set(int32_t *flag, int stage, int8_t value) {
    *flag |= (value << (8 * stage));
}

inline int8_t ggml_task_flag_get(int32_t flag, int stage) {
    return (flag >> (8 * stage)) & 0xFF;
}

static const char *ggml_blas_names[GGML_BLAS_TYPE_COUNT] = {
    [GGML_BLAS_TYPE_ACCELERATE] = "Accelerate",
    [GGML_BLAS_TYPE_CLBLAST] = "CLBlast",
    [GGML_BLAS_TYPE_CUBLAS] = "cuBLAS",
    [GGML_BLAS_TYPE_OPENBLAS] = "OpenBLAS",
};

const char *ggml_get_blas_names(void) { return (const char *)ggml_blas_names; }

int ggml_mulmat_read_tune_data(struct ggml_mulmat_tune *tune, FILE *fp) {
    int rc = fscanf(fp, "%d %s %s %s %d %d %d", &tune->version, tune->model,
                    tune->q_type_name, tune->blas_name, &tune->n_groups,
                    &tune->step_m, &tune->num_m);
    if (rc <= 0) {
        return rc;
    }

    for (int i = 0; i < 3; i++) {
        rc = fscanf(fp, "%1d", &tune->cpu_only_stages[i]);
        if (rc <= 0) {
            return rc;
        }
    }

    for (int i = 0; i < 3; i++) {
        rc = fscanf(fp, "%d", &tune->use_blas_stages[i]);
        if (rc <= 0) {
            return rc;
        }
    }

    tune->groups = malloc(sizeof(struct ggml_mulmat_tune_nk) * tune->n_groups);

    for (int i = 0; i < tune->n_groups; i++) {
        struct ggml_mulmat_tune_nk *s = &tune->groups[i];

        rc = fscanf(fp, "%d%d", &s->N, &s->K);
        if (rc <= 0) {
            return rc;
        }

        s->items = malloc(sizeof(struct ggml_mulmat_tune_m) * tune->num_m);

        for (int j = 0; j < tune->num_m; j++) {
            struct ggml_mulmat_tune_m *item = &s->items[j];
            rc = fscanf(fp, "%d %d %d %d %d %d %d", &item->M,
                        &item->cpu_only_time[0], &item->cpu_only_time[1],
                        &item->cpu_only_time[2], &item->use_blas_time[0],
                        &item->use_blas_time[1], &item->use_blas_time[2]);
            if (rc <= 0) {
                return rc;
            }
        }
    }

    return 0;
}

void ggml_mulmat_write_tune_data(const struct ggml_mulmat_tune *tune,
                                 FILE *fp) {
    fprintf(fp, "%d %s %s %s %d %d %d", tune->version, tune->model,
            tune->q_type_name, tune->blas_name, tune->n_groups, tune->step_m,
            tune->num_m);

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%2d", tune->cpu_only_stages[i]);
    }

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%2d", tune->use_blas_stages[i]);
    }

    fprintf(fp, "\n");

    for (int i = 0; i < tune->n_groups; i++) {
        struct ggml_mulmat_tune_nk *group = &tune->groups[i];

        fprintf(fp, "%d %d\n", group->N, group->K);

        for (int j = 0; j < tune->num_m; j++) {
            struct ggml_mulmat_tune_m *item = &group->items[j];
            fprintf(fp, "%3d", item->M);
            for (int k = 0; k < 3; k++) {
                if (tune->cpu_only_stages[k] > 0) {
                    fprintf(fp, "%9d", item->cpu_only_time[k]);
                } else {
                    fprintf(fp, " 0");
                }
            }
            for (int k = 0; k < 3; k++) {
                if (tune->use_blas_stages[k] > 0) {
                    fprintf(fp, "%9d", item->use_blas_time[k]);
                } else {
                    fprintf(fp, " 0");
                }
            }
            fprintf(fp, "\n");
        }
    }
}

// for given work load and number of threads, estimate total time of with or
// without blas. return -1 when unable to estimate.
int ggml_mulmat_estimate_time(const struct ggml_mulmat_tune *tune, int M, int N,
                              int K, int nth, bool cpu_only) {
    struct ggml_mulmat_tune_nk *group = NULL;
    for (int i = 0; i < tune->n_groups; i++) {
        if (tune->groups[i].N == N && tune->groups[i].K == K) {
            group = &tune->groups[i];
            break;
        }
    }

    if (group == NULL) {
        return -1;
    }

    if (M < tune->step_m || M > tune->step_m * tune->num_m) {
        return -1;
    }

    for (int i = 0; i < tune->num_m; i++) {
        struct ggml_mulmat_tune_m *item = &group->items[i];
        if (item->M == M) {
            int total = 0;
            for (int j = 0; j < 3; j++) {
                int sv = cpu_only ? tune->cpu_only_stages[j]
                                  : tune->use_blas_stages[j];

                if (sv > 0) {
                    int t = cpu_only ? item->cpu_only_time[j]
                                     : item->use_blas_time[j];
                    if (sv == GGML_TASK_FLAG_TN) {
                        t /= nth;
                    }
                    total += t;
                }
            }
            return total;
        }
    }

    for (int i = 0; i < tune->num_m - 1; i++) {
        struct ggml_mulmat_tune_m *prev = &group->items[i];
        struct ggml_mulmat_tune_m *next = &group->items[i + 1];
        // interpolate.
        if (M > prev->M && M < next->M) {
            double x = 1.0 * (M - prev->M) / (next->M - prev->M);
            int total = 0;
            for (int j = 0; j < 3; j++) {
                int sv = cpu_only ? tune->cpu_only_stages[j]
                                  : tune->use_blas_stages[j];

                if (sv > 0) {
                    int pv = cpu_only ? prev->cpu_only_time[j]
                                      : prev->use_blas_time[j];
                    int nv = cpu_only ? next->cpu_only_time[j]
                                      : next->use_blas_time[j];

                    double t = pv + (nv - pv) * x;
                    if (sv == GGML_TASK_FLAG_TN) {
                        t /= nth;
                    }
                    total += t;
                }
            }
            return (int)total;
        }
    }

    return -1;
}

int ggml_mulmat_tune_time_stats(
    const struct ggml_mulmat_tune *tune, const int M, const int N, const int K,
    const int nth, struct ggml_mulmat_tune_time_stats *time_stats) {
    if (M < tune->step_m) {
        return -1;
    } else if (M > tune->step_m * tune->num_m) {
        return -1;
    }

    struct ggml_mulmat_tune_nk *group = NULL;

    for (int i = 0; i < tune->n_groups; i++) {
        if (tune->groups[i].N == N && tune->groups[i].K == K) {
            group = &tune->groups[i];
            break;
        }
    }

    if (group == NULL) {
        return -1;
    }

    struct ggml_mulmat_tune_m *prev = NULL;
    struct ggml_mulmat_tune_m *next = NULL;

    for (int i = 0; i < tune->num_m; i++) {
        next = &group->items[i];
        if (next->M == M) {
            prev = next;
            break;
        }

        if (tune->num_m > 1) {
            prev = &group->items[i - 1];
            if (M > prev->M && M < next->M) {
                break;
            }
        }
    }

    // If this happens, the tune data should be incomplete.
    // TODO: we'd validate tune data before hands.
    if (prev == NULL || next == NULL) {
        return -1;
    }

    memset(time_stats, 0, sizeof(struct ggml_mulmat_tune_time_stats));

    // interpolate.

    double x = 0.0;
    if (prev != next) {
        x = 1.0 * (M - prev->M) / (next->M - prev->M);
    }

    for (int j = 0; j < 3; j++) {
        if (tune->cpu_only_stages[j] > 0) {
            int pv = prev->cpu_only_time[j];
            double t = pv;
            if (x > 0) {
                int nv = next->cpu_only_time[j];
                t += x * (nv - pv);
            }
            if (tune->cpu_only_stages[j] == GGML_TASK_FLAG_TN) {
                t /= nth;
            }
            time_stats->cpu_only_stages[j] = t;
            time_stats->cpu_only_total += t;
        }

        if (tune->use_blas_stages[j] > 0) {
            int pv = prev->use_blas_time[j];
            double t = pv;
            if (x > 0) {
                int nv = next->use_blas_time[j];
                t += x * (nv - pv);
            }
            if (tune->use_blas_stages[j] == GGML_TASK_FLAG_TN) {
                t /= nth;
            }
            time_stats->use_blas_stages[j] = t;
            time_stats->use_blas_total += t;
        }
    }

    return 0;
}

const char *ggml_get_blas_name(void) {
#if defined(GGML_USE_ACCELERATE)
    return ggml_blas_names[GGML_BLAS_TYPE_ACCELERATE];
#elif defined(GGML_USE_CLBLAST)
    return ggml_blas_names[GGML_BLAS_TYPE_CLBLAST];
#elif defined(GGML_USE_CUBLAS)
    return ggml_blas_names[GGML_BLAS_TYPE_CUBLAS];
#elif defined(GGML_USE_OPENBLAS)
    return ggml_blas_names[GGML_BLAS_TYPE_OPENBLAS];
#else
    return NULL;
#endif
}
