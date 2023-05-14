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

#if defined(LLAMA_NO_ACCELERATE)
static const char * build_with_blas_name = ggml_blas_names[GGML_BLAS_TYPE_ACCELERATE];
#elif defined(LLAMA_CLBLAST)
static const char * build_with_blas_name = ggml_blas_names[GGML_BLAS_TYPE_CLBLAST];
#elif defined(LLAMA_CUBLAS)
static const char * build_with_blas_name = ggml_blas_names[GGML_BLAS_TYPE_CUBLAS];
#elif defined(LLAMA_OPENBLAS)
static const char * build_with_blas_name = ggml_blas_names[GGML_BLAS_TYPE_OPENBLAS];
#else
static const char * build_with_blas_name = "";
#endif

const char* ggml_get_blas_names(void) {
    return (const char *)ggml_blas_names;
}

int ggml_mulmat_read_bench_data(struct ggml_mulmat_bench *bench, FILE *fp) {
    int rc = fscanf(fp, "%d %s %s %d %d %d", &bench->version, bench->model,
                    bench->blas_name, &bench->n_groups, &bench->m_step,
                    &bench->num_m);
    if (rc <= 0) {
        return rc;
    }

    for (int i = 0; i < 3; i++) {
        rc = fscanf(fp, "%1d", &bench->cpu_stages[i]);
        if (rc <= 0) {
            return rc;
        }
    }

    for (int i = 0; i < 3; i++) {
        rc = fscanf(fp, "%d", &bench->gpu_stages[i]);
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
                        &item->cpu_time[0], &item->cpu_time[1],
                        &item->cpu_time[2], &item->gpu_time[0],
                        &item->gpu_time[1], &item->gpu_time[2]);
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
        fprintf(fp, "%2d", bench->cpu_stages[i]);
    }

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%2d", bench->gpu_stages[i]);
    }

    fprintf(fp, "\n");

    for (int i = 0; i < bench->n_groups; i++) {
        struct ggml_mulmat_bench_nk *group = &bench->groups[i];

        fprintf(fp, "%d %d\n", group->N, group->K);

        for (int j = 0; j < bench->num_m; j++) {
            struct ggml_mulmat_bench_m *item = &group->items[j];
            fprintf(fp, "%3d", item->M);
            for (int k = 0; k < 3; k++) {
                if (bench->cpu_stages[k] > 0) {
                    fprintf(fp, "%9d", item->cpu_time[k]);
                } else {
                    fprintf(fp, " 0");
                }
            }
            for (int k = 0; k < 3; k++) {
                if (bench->gpu_stages[k] > 0) {
                    fprintf(fp, "%9d", item->gpu_time[k]);
                } else {
                    fprintf(fp, " 0");
                }
            }
            fprintf(fp, "\n");
        }
    }
}

// for given work load and number of threads, estimate cpu or gpu time.
// return -1 when unable to estimate.
int ggml_mulmat_estimate_time(const struct ggml_mulmat_bench *bench, int M,
                              int N, int K, int nth, bool is_cpu) {
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
                int sv = is_cpu ? bench->cpu_stages[j] : bench->gpu_stages[j];

                if (sv > 0) {
                    int t = is_cpu ? item->cpu_time[j] : item->gpu_time[j];
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
                int sv = is_cpu ? bench->cpu_stages[j] : bench->gpu_stages[j];

                if (sv > 0) {
                    int pv = is_cpu ? prev->cpu_time[j] : prev->gpu_time[j];
                    int nv = is_cpu ? next->cpu_time[j] : next->gpu_time[j];

                    double t = pv + (nv - pv) * x;
                    if (sv & GGML_TASK_FLAG_N_THREADS) {
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

// TODO: check impl: cuBLAS, clBLAS, accelerate, openBLAS
bool ggml_mulmat_bench_use_blas(const struct ggml_mulmat_bench *b, const int M,
                                const int N, const int K, const int nth) {
    if (M < b->m_step) {
        return false;
    } else if (M > b->m_step * b->num_m) {
        return true;
    }

    int cpu_time = ggml_mulmat_estimate_time(b, M, N, K, nth, true /* cpu */);
    int gpu_time = ggml_mulmat_estimate_time(b, M, N, K, nth, false /* gpu */);

    if (cpu_time < 0 || cpu_time < 0) {
        return (M >= 32 && N >= 32 && K >= 32);
    }

    return (cpu_time > gpu_time);
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
