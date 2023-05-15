#include <string.h>

#include "examples/mulmat-device/mulmat-device.h"

void ggml_task_flag_set_blas(int32_t *flag, int8_t value) {
    *flag |= (value << 24);
}

int8_t ggml_task_flag_get_blas(int32_t flag) { return (flag >> 24) & 0xFF; }

void ggml_task_flag_set(int32_t *flag, int stage, int8_t value) {
    *flag |= (value << (8 * stage));
}

int8_t ggml_task_flag_get(int32_t flag, int stage) {
    return (flag >> (8 * stage)) & 0xFF;
}

static const char *ggml_blas_names[GGML_BLAS_TYPE_COUNT] = {
    [GGML_BLAS_TYPE_ACCELERATE] = "Accelerate",
    [GGML_BLAS_TYPE_CLBLAST] = "CLBlast",
    [GGML_BLAS_TYPE_CUBLAS] = "cuBLAS",
    [GGML_BLAS_TYPE_OPENBLAS] = "OpenBLAS",
};

const char *ggml_get_blas_names(void) { return (const char *)ggml_blas_names; }

int ggml_mulmat_read_bench_data(struct ggml_mulmat_bench *bench, FILE *fp) {
    int rc = fscanf(fp, "%d %s %s %d %d %d", &bench->version, bench->model,
                    bench->blas_name, &bench->n_groups, &bench->m_step,
                    &bench->num_m);
    if (rc <= 0) {
        return rc;
    }

    for (int i = 0; i < 3; i++) {
        rc = fscanf(fp, "%1d", &bench->cpu_only_stages[i]);
        if (rc <= 0) {
            return rc;
        }
    }

    for (int i = 0; i < 3; i++) {
        rc = fscanf(fp, "%d", &bench->use_blas_stages[i]);
        if (rc <= 0) {
            return rc;
        }
    }

    bench->groups =
        malloc(sizeof(struct ggml_mulmat_bench_nk) * bench->n_groups);

    for (int i = 0; i < bench->n_groups; i++) {
        struct ggml_mulmat_bench_nk *s = &bench->groups[i];

        rc = fscanf(fp, "%d%d", &s->N, &s->K);
        if (rc <= 0) {
            return rc;
        }

        s->items = malloc(sizeof(struct ggml_mulmat_bench_m) * bench->num_m);

        for (int j = 0; j < bench->num_m; j++) {
            struct ggml_mulmat_bench_m *item = &s->items[j];
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

void ggml_mulmat_write_bench_data(const struct ggml_mulmat_bench *bench,
                                  FILE *fp) {
    fprintf(fp, "%d %s %s %d %d %d", bench->version, bench->model,
            bench->blas_name, bench->n_groups, bench->m_step, bench->num_m);

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%2d", bench->cpu_only_stages[i]);
    }

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%2d", bench->use_blas_stages[i]);
    }

    fprintf(fp, "\n");

    for (int i = 0; i < bench->n_groups; i++) {
        struct ggml_mulmat_bench_nk *group = &bench->groups[i];

        fprintf(fp, "%d %d\n", group->N, group->K);

        for (int j = 0; j < bench->num_m; j++) {
            struct ggml_mulmat_bench_m *item = &group->items[j];
            fprintf(fp, "%3d", item->M);
            for (int k = 0; k < 3; k++) {
                if (bench->cpu_only_stages[k] > 0) {
                    fprintf(fp, "%9d", item->cpu_only_time[k]);
                } else {
                    fprintf(fp, " 0");
                }
            }
            for (int k = 0; k < 3; k++) {
                if (bench->use_blas_stages[k] > 0) {
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
int ggml_mulmat_estimate_time(const struct ggml_mulmat_bench *bench, int M,
                              int N, int K, int nth, bool cpu_only) {
    struct ggml_mulmat_bench_nk *group = NULL;
    for (int i = 0; i < bench->n_groups; i++) {
        if (bench->groups[i].N == N && bench->groups[i].K == K) {
            group = &bench->groups[i];
            break;
        }
    }

    if (group == NULL) {
        return -1;
    }

    if (M < bench->m_step || M > bench->m_step * bench->num_m) {
        return -1;
    }

    for (int i = 0; i < bench->num_m; i++) {
        struct ggml_mulmat_bench_m *item = &group->items[i];
        if (item->M == M) {
            int total = 0;
            for (int j = 0; j < 3; j++) {
                int sv = cpu_only ? bench->cpu_only_stages[j]
                                  : bench->use_blas_stages[j];

                if (sv > 0) {
                    int t = cpu_only ? item->cpu_only_time[j]
                                     : item->use_blas_time[j];
                    if (sv == GGML_TASK_FLAG_N_THREADS) {
                        t /= nth;
                    }
                    total += t;
                }
            }
            return total;
        }
    }

    for (int i = 0; i < bench->num_m - 1; i++) {
        struct ggml_mulmat_bench_m *prev = &group->items[i];
        struct ggml_mulmat_bench_m *next = &group->items[i + 1];
        // interpolate.
        if (M > prev->M && M < next->M) {
            double x = 1.0 * (M - prev->M) / (next->M - prev->M);
            int total = 0;
            for (int j = 0; j < 3; j++) {
                int sv = cpu_only ? bench->cpu_only_stages[j]
                                  : bench->use_blas_stages[j];

                if (sv > 0) {
                    int pv = cpu_only ? prev->cpu_only_time[j]
                                      : prev->use_blas_time[j];
                    int nv = cpu_only ? next->cpu_only_time[j]
                                      : next->use_blas_time[j];

                    double t = pv + (nv - pv) * x;
                    if (sv == GGML_TASK_FLAG_N_THREADS) {
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

int ggml_mulmat_bench_time_stats(const struct ggml_mulmat_bench *bench,
                                const int M, const int N, const int K,
                                const int nth,
                                struct ggml_mulmat_bench_time_stats *time_stats) {
    if (M < bench->m_step) {
        return -1;
    } else if (M > bench->m_step * bench->num_m) {
        return -1;
    }

    struct ggml_mulmat_bench_nk *group = NULL;

    for (int i = 0; i < bench->n_groups; i++) {
        if (bench->groups[i].N == N && bench->groups[i].K == K) {
            group = &bench->groups[i];
            break;
        }
    }

    if (group == NULL) {
        return -1;
    }

    struct ggml_mulmat_bench_m *prev = NULL;
    struct ggml_mulmat_bench_m *next = NULL;

    for (int i = 0; i < bench->num_m; i++) {
        next = &group->items[i];
        if (next->M == M) {
            prev = next;
            break;
        }

        if (bench->num_m > 1) {
            prev = &group->items[i - 1];
            if (M > prev->M && M < next->M) {
                break;
            }
        }
    }

    // If this happens, the bench data should be incomplete.
    // TODO: we'd validate bench data before hands.
    if (prev == NULL || next == NULL) {
        return -1;
    }

    memset(time_stats, 0, sizeof(struct ggml_mulmat_bench_time_stats));

    // interpolate.

    double x = 0.0;
    if (prev != next) {
        x = 1.0 * (M - prev->M) / (next->M - prev->M);
    }

    for (int j = 0; j < 3; j++) {
        if (bench->cpu_only_stages[j] > 0) {
            int pv = prev->cpu_only_time[j];
            double t = pv;
            if (x > 0) {
                int nv = next->cpu_only_time[j];
                t += x * (nv - pv);
            }
            if (bench->cpu_only_stages[j] == GGML_TASK_FLAG_N_THREADS) {
                t /= nth;
            }
            time_stats->cpu_only_stages[j] = t;
            time_stats->cpu_only_total += t;
        }

        if (bench->use_blas_stages[j] > 0) {
            int pv = prev->use_blas_time[j];
            double t = pv;
            if (x > 0) {
                int nv = next->use_blas_time[j];
                t += x * (nv - pv);
            }
            if (bench->use_blas_stages[j] == GGML_TASK_FLAG_N_THREADS) {
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
