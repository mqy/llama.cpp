#include "examples/benchmark/q40-mulmat-device-bench/q40-mulmat-device.h"

int ggml_mulmat_read_bench_data(struct ggml_mulmat_bench_data *bd, FILE *fp) {
    int rc = fscanf(fp, "%d %s %s %d %d %d", &bd->version, bd->model,
                    bd->gpu_impl, &bd->n_shapes, &bd->m_step, &bd->num_m);
    if (rc <= 0) {
        return rc;
    }

    for (int i = 0; i < 3; i++) {
        rc = fscanf(fp, "%1d", &bd->cpu_stages[i]);
        if (rc <= 0) {
            return rc;
        }
    }

    for (int i = 0; i < 3; i++) {
        rc = fscanf(fp, "%d", &bd->gpu_stages[i]);
        if (rc <= 0) {
            return rc;
        }
    }

    bd->shapes = malloc(sizeof(struct ggml_mulmat_bench_data_shape) * bd->n_shapes);

    for (int i = 0; i < bd->n_shapes; i++) {
        struct ggml_mulmat_bench_data_shape *s = &bd->shapes[i];

        rc = fscanf(fp, "%d%d", &s->N, &s->K);
        if (rc <= 0) {
            return rc;
        }

        s->items = malloc(sizeof(struct ggml_mulmat_bench_data_item) * bd->num_m);

        for (int j = 0; j < bd->num_m; j++) {
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
}

void ggml_mulmat_write_bench_data(struct ggml_mulmat_bench_data *bd, FILE *fp) {
    fprintf(fp, "%d %s %s %d %d %d", bd->version, bd->model, bd->gpu_impl,
            bd->n_shapes, bd->m_step, bd->num_m);

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%2d", bd->cpu_stages[i]);
    }

    for (int i = 0; i < 3; i++) {
        fprintf(fp, "%2d", bd->gpu_stages[i]);
    }

    fprintf(fp, "\n");

    for (int i = 0; i < bd->n_shapes; i++) {
        struct ggml_mulmat_bench_data_shape *s = &bd->shapes[i];

        fprintf(fp, "%d %d\n", s->N, s->K);

        for (int j = 0; j < bd->num_m; j++) {
            struct ggml_mulmat_bench_data_item *item = &s->items[j];
            fprintf(fp, "%3d", item->M);
            for (int k = GGML_TASK_INIT; k <= GGML_TASK_FINALIZE; k++) {
                if (bd->cpu_stages[k] & 1) {
                    fprintf(fp, "%8d", item->cpu_time[k]);
                } else {
                    fprintf(fp, " 0");
                }
            }
            for (int k = GGML_TASK_INIT; k <= GGML_TASK_FINALIZE; k++) {
                if (bd->gpu_stages[k] & 1) {
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
int ggml_mulmat_estimate_time(struct ggml_mulmat_bench_data *bd, int M, int N, int K, int nth,
                  bool is_cpu) {
    struct ggml_mulmat_bench_data_shape *shape = NULL;
    for (int i = 0; i < bd->n_shapes; i++) {
        if (bd->shapes[i].N == N && bd->shapes[i].K == K) {
            shape = &bd->shapes[i];
            break;
        }
    }

    if (shape == NULL) {
        return -1;
    }

    if (M < bd->m_step || M > bd->m_step * bd->num_m) {
        return -1;
    }

    for (int i = 0; i < bd->num_m; i++) {
        struct ggml_mulmat_bench_data_item *item = &shape->items[i];
        if (item->M == M) {
            int total = 0;
            for (int j = 0; j < 3; j++) {
                int sv = is_cpu ? bd->cpu_stages[j] : bd->gpu_stages[j];

                if (sv & 1) {
                    int t = is_cpu ? item->cpu_time[j] : item->gpu_time[j];
                    if (sv & (1 << 1)) {
                        t /= nth;
                    }
                    total += t;
                }
            }
            return total;
        }
    }

    for (int i = 0; i < bd->num_m - 1; i++) {
        struct ggml_mulmat_bench_data_item *prev = &shape->items[i];
        struct ggml_mulmat_bench_data_item *next = &shape->items[i + 1];
        // interpolate.
        if (M > prev->M && M < next->M) {
            double x = 1.0 * (M - prev->M) / (next->M - prev->M);
            int total = 0;
            for (int j = 0; j < 3; j++) {
                int sv = is_cpu ? bd->cpu_stages[j] : bd->gpu_stages[j];

                if (sv & 1) {
                    int pv = is_cpu ? prev->cpu_time[j] : prev->gpu_time[j];
                    int nv = is_cpu ? next->cpu_time[j] : next->gpu_time[j];

                    double t = pv + (nv - pv) * x;
                    if (sv & (1 << 1)) {
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

enum ggml_device_type ggml_mulmat_choose_device(struct ggml_mulmat_bench_data *bd, int M, int N, int K,
                                    int nth) {
    if (M < bd->m_step) {
        return GGML_DEVICE_CPU;
    } else if (M > bd->m_step * bd->num_m) {
        return GGML_DEVICE_GPU;
    }

    int cpu_time = ggml_mulmat_estimate_time(bd, M, N, K, nth, true /* cpu */);
    int gpu_time = ggml_mulmat_estimate_time(bd, M, N, K, nth, false /* gpu */);

    if (cpu_time < 0 && cpu_time < 0) {
        return GGML_DEVICE_AUTO;
    }

    return (cpu_time < gpu_time) ? GGML_DEVICE_CPU : GGML_DEVICE_GPU;
}
