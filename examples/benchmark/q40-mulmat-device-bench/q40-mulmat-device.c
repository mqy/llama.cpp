#include "examples/benchmark/q40-mulmat-device-bench/q40-mulmat-device.h"

int ggml_mulmat_read_bench_data(struct ggml_mulmat_bench *bench, FILE *fp) {
    int rc = fscanf(fp, "%d %s %s %d %d %d", &bench->version, bench->model,
                    bench->gpu_impl, &bench->n_shapes, &bench->m_step,
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

    bench->shapes =
        malloc(sizeof(struct ggml_mulmat_bench_data_shape) * bench->n_shapes);

    for (int i = 0; i < bench->n_shapes; i++) {
        struct ggml_mulmat_bench_data_shape *s = &bench->shapes[i];

        rc = fscanf(fp, "%d%d", &s->N, &s->K);
        if (rc <= 0) {
            return rc;
        }

        s->items =
            malloc(sizeof(struct ggml_mulmat_bench_data_item) * bench->num_m);

        for (int j = 0; j < bench->num_m; j++) {
            struct ggml_mulmat_bench_data_item *item = &s->items[j];
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

void ggml_mulmat_write_bench_data(struct ggml_mulmat_bench *bench, FILE *fp) {
    fprintf(fp, "%d %s %s %d %d %d", bench->version, bench->model,
            bench->gpu_impl, bench->n_shapes, bench->m_step, bench->num_m);

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%2d", bench->cpu_stages[i]);
    }

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%2d", bench->gpu_stages[i]);
    }

    fprintf(fp, "\n");

    for (int i = 0; i < bench->n_shapes; i++) {
        struct ggml_mulmat_bench_data_shape *shape = &bench->shapes[i];

        fprintf(fp, "%d %d\n", shape->N, shape->K);

        for (int j = 0; j < bench->num_m; j++) {
            struct ggml_mulmat_bench_data_item *item = &shape->items[j];
            fprintf(fp, "%3d", item->M);
            for (int k = 0; k < 3; k++) {
                if (bench->cpu_stages[k] & COMPUTE_STAGE_FLAG_VALID) {
                    fprintf(fp, "%8d", item->cpu_time[k]);
                } else {
                    fprintf(fp, " 0");
                }
            }
            for (int k = 0; k < 3; k++) {
                if (bench->gpu_stages[k] & COMPUTE_STAGE_FLAG_VALID) {
                    fprintf(fp, "%7d", item->gpu_time[k]);
                } else {
                    fprintf(fp, " 0");
                }
            }
            fprintf(fp, "\n");
        }
    }
}

// for given work load and number of threads, estimate cpu or gpu time.
int ggml_mulmat_estimate_time(struct ggml_mulmat_bench *bench, int M, int N,
                              int K, int nth, bool is_cpu) {
    struct ggml_mulmat_bench_data_shape *shape = NULL;
    for (int i = 0; i < bench->n_shapes; i++) {
        if (bench->shapes[i].N == N && bench->shapes[i].K == K) {
            shape = &bench->shapes[i];
            break;
        }
    }

    if (shape == NULL) {
        return -1;
    }

    if (M < bench->m_step || M > bench->m_step * bench->num_m) {
        return -1;
    }

    for (int i = 0; i < bench->num_m; i++) {
        struct ggml_mulmat_bench_data_item *item = &shape->items[i];
        if (item->M == M) {
            int total = 0;
            for (int j = 0; j < 3; j++) {
                int sv = is_cpu ? bench->cpu_stages[j] : bench->gpu_stages[j];

                if (sv & COMPUTE_STAGE_FLAG_VALID) {
                    int t = is_cpu ? item->cpu_time[j] : item->gpu_time[j];
                    if (sv & COMPUTE_STAGE_FLAG_NEED_WORKER) {
                        t /= nth;
                    }
                    total += t;
                }
            }
            return total;
        }
    }

    for (int i = 0; i < bench->num_m - 1; i++) {
        struct ggml_mulmat_bench_data_item *prev = &shape->items[i];
        struct ggml_mulmat_bench_data_item *next = &shape->items[i + 1];
        // interpolate.
        if (M > prev->M && M < next->M) {
            double x = 1.0 * (M - prev->M) / (next->M - prev->M);
            int total = 0;
            for (int j = 0; j < 3; j++) {
                int sv = is_cpu ? bench->cpu_stages[j] : bench->gpu_stages[j];

                if (sv & COMPUTE_STAGE_FLAG_VALID) {
                    int pv = is_cpu ? prev->cpu_time[j] : prev->gpu_time[j];
                    int nv = is_cpu ? next->cpu_time[j] : next->gpu_time[j];

                    double t = pv + (nv - pv) * x;
                    if (sv & COMPUTE_STAGE_FLAG_NEED_WORKER) {
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

enum ggml_device_type ggml_mulmat_choose_device(struct ggml_mulmat_bench *b,
                                                int M, int N, int K, int nth) {
    if (M < b->m_step) {
        return GGML_DEVICE_CPU;
    } else if (M > b->m_step * b->num_m) {
        return GGML_DEVICE_GPU;
    }

    int cpu_time = ggml_mulmat_estimate_time(b, M, N, K, nth, true /* cpu */);
    int gpu_time = ggml_mulmat_estimate_time(b, M, N, K, nth, false /* gpu */);

    if (cpu_time < 0 || cpu_time < 0) {
        return (M < 32 || N < 32 || K < 32) ? GGML_DEVICE_CPU : GGML_DEVICE_GPU;
    }

    return (cpu_time < gpu_time) ? GGML_DEVICE_CPU : GGML_DEVICE_GPU;
}
