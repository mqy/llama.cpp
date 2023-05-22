#include <string.h>

#include "examples/mulmat-tune/mulmat-tune.h"
#include "ggml.h"

int ggml_mulmat_tune_setup_model(struct ggml_mulmat_tune *tune,
                                 const char *model_name) {
    size_t n = sizeof(tune->model);
    strncpy(tune->model, model_name, n);

    if (strcmp(model_name, "7B") == 0) {
        tune->n_shapes = 4;
        tune->shapes[0] = (struct ggml_mulmat_tune_shape){.N = 4096, .K = 4096};
        tune->shapes[1] =
            (struct ggml_mulmat_tune_shape){.N = 4094, .K = 11008};
        tune->shapes[2] =
            (struct ggml_mulmat_tune_shape){.N = 11008, .K = 4096};
        tune->shapes[3] =
            (struct ggml_mulmat_tune_shape){.N = 32000, .K = 4096};
    } else if (strcmp(model_name, "13B") == 0) {
        tune->n_shapes = 4;
        tune->shapes[0] = (struct ggml_mulmat_tune_shape){.N = 5120, .K = 5120};
        tune->shapes[1] =
            (struct ggml_mulmat_tune_shape){.N = 5120, .K = 13824};
        tune->shapes[2] =
            (struct ggml_mulmat_tune_shape){.N = 13824, .K = 5120};
        tune->shapes[3] =
            (struct ggml_mulmat_tune_shape){.N = 32000, .K = 5120};
    } else if (strcmp(model_name, "30B") == 0) {
        fprintf(stderr, "error: model %s is not supported\n", model_name);
        return -2;
    } else if (strcmp(model_name, "65B") == 0) {
        fprintf(stderr, "error: model %s  is not supported\n", model_name);
        return -2;
    } else {
        fprintf(stderr, "error: unknown model: %s\n", model_name);
        return -3;
    }

    return 0;
}

int ggml_mulmat_tune_validate(struct ggml_mulmat_tune *tune, const char *model,
                              int type) {
    enum ggml_backend gpu_backend = ggml_get_backend();
    if ((int)gpu_backend != tune->gpu_backend) {
        return -1;
    }

    if (model != NULL && !strcmp(model, tune->model)) {
        return -2;
    }

    if (type >= 0 && type != tune->type) {
        return -3;
    }

    // TODO: validate model_name, type
    return 0;
}

int ggml_mulmat_tune_read_data(struct ggml_mulmat_tune *tune, FILE *fp) {
    int rc =
        fscanf(fp, "%d %s %s %d %s %d %d %d %d", &tune->version, tune->model,
               tune->type_name, &tune->gpu_backend, tune->gpu_backend_name,
               &tune->n_shapes, &tune->m_step, &tune->m_num, &tune->n_profiles);
    if (rc <= 0) {
        return rc;
    }

    tune->items = malloc(sizeof(struct ggml_mulmat_tune_m) *
                         (tune->n_shapes * tune->m_num * tune->n_profiles));
    if (tune->items == NULL) {
        fprintf(stderr, "failed to allocate memory\n");
        return -2;
    }

    for (int ip = 0; ip < tune->n_profiles; ip++) {
        for (int j = 0; j < 3; j++) {
            int parallel, wait;
            struct ggml_task_conf *conf = &tune->conf[ip][j];
            rc = fscanf(fp, "%d %d %d", &conf->backend, &parallel, &wait);
            if (rc <= 0) {
                return rc;
            }
            conf->parallel = parallel ? true : false;
            conf->wait = wait ? true : false;
        }
    }

    for (int i_shapes = 0; i_shapes < tune->n_shapes; i_shapes++) {
        rc = fscanf(fp, "%d %d", &tune->shapes[i_shapes].N,
                    &tune->shapes[i_shapes].K);
        if (rc <= 0) {
            return rc;
        }

        for (int i_m = 0; i_m < tune->m_num; i_m++) {
            int M;
            for (int ip = 0; ip < tune->n_profiles; ip++) {
                if (ip == 0) {
                    rc = fscanf(fp, "%d", &M);
                    if (rc <= 0) {
                        return rc;
                    }
                }
                int item_index =
                    (i_shapes * tune->m_num + i_m) * tune->n_profiles + ip;
                struct ggml_mulmat_tune_m *item = &tune->items[item_index];
                item->M = M;
                rc = fscanf(fp, "%d %d %d", &item->stages_time[0],
                            &item->stages_time[1], &item->stages_time[2]);
                if (rc <= 0) {
                    return rc;
                }
            }
        }
    }

    return 0;
}

int ggml_mulmat_tune_write_data(const struct ggml_mulmat_tune *tune, FILE *fp) {
    int rc =
        fprintf(fp, "%d %s %s %d %s %d %d %d %d\n", tune->version, tune->model,
                tune->type_name, tune->gpu_backend, tune->gpu_backend_name,
                tune->n_shapes, tune->m_step, tune->m_num, tune->n_profiles);
    if (rc <= 0) {
        return rc;
    }

    for (int i = 0; i < tune->n_profiles; i++) {
        for (int j = 0; j < 3; j++) {
            rc = fprintf(fp, "%d %d %d", tune->conf[i][j].backend,
                         tune->conf[i][j].parallel ? 1 : 0,
                         tune->conf[i][j].wait ? 1 : 0);
            if (rc <= 0) {
                return rc;
            }
            if (j < 2) {
                rc = fprintf(fp, " ");
                if (rc <= 0) {
                    return rc;
                }
            }
        }
        rc = fprintf(fp, "\n");
        if (rc <= 0) {
            return rc;
        }
    }

    for (int i_shape = 0; i_shape < tune->n_shapes; i_shape++) {
        const struct ggml_mulmat_tune_shape *shape = &tune->shapes[i_shape];
        rc = fprintf(fp, "%d %d\n", shape->N, shape->K);
        if (rc <= 0) {
            return rc;
        }

        for (int i_m = 0; i_m < tune->m_num; i_m++) {
            for (int ip = 0; ip < tune->n_profiles; ip++) {
                int item_index =
                    (i_shape * tune->m_num + i_m) * tune->n_profiles + ip;
                struct ggml_mulmat_tune_m *item = &tune->items[item_index];
                if (ip == 0) {
                    rc = fprintf(fp, "%3d", item->M);
                    if (rc <= 0) {
                        return rc;
                    }
                }

                for (int k = 0; k < 3; k++) {
                    enum ggml_backend backend = tune->conf[ip][k].backend;
                    if (backend != GGML_BACKEND_UNKNOWN) {
                        rc = fprintf(fp, "%9d", item->stages_time[k]);
                        if (rc <= 0) {
                            return rc;
                        }
                    } else {
                        rc = fprintf(fp, " 0");
                        if (rc <= 0) {
                            return rc;
                        }
                    }
                }
            }
            rc = fprintf(fp, "\n");
            if (rc <= 0) {
                return rc;
            }
        }
    }

    return 0;
}

int ggml_mulmat_tune_time_stats(
    const struct ggml_mulmat_tune *tune, const int M, const int N, const int K,
    const int nth, struct ggml_mulmat_tune_time_stats *time_stats) {
    if (M < tune->m_step) {
        return -1;
    } else if (M > tune->m_step * tune->m_num) {
        return -1;
    }

    int shape_index = -1;
    for (int i = 0; i < tune->n_shapes; i++) {
        if (tune->shapes[i].N == N && tune->shapes[i].K == K) {
            shape_index = i;
            break;
        }
    }

    if (shape_index < 0) {
        return -1;
    }

    struct ggml_mulmat_tune_m *prev = NULL;
    struct ggml_mulmat_tune_m *next = NULL;

    for (int i = 0; i < tune->m_num; i++) {
        int next_index = (shape_index * tune->m_num + i) * tune->n_profiles;
        next = &tune->items[next_index];
        if (next->M == M) {
            prev = next;
            break;
        }

        if (tune->m_num > 1) {
            int prev_index = next_index - tune->n_profiles;
            prev = &tune->items[prev_index];
            if (M > prev->M && M < next->M) {
                break;
            }
        }
    }

    // If this happens, the tune data should be incomplete.
    if (prev == NULL || next == NULL) {
        return -1;
    }

    memset(time_stats, 0, sizeof(struct ggml_mulmat_tune_time_stats));
    time_stats->n_profiles = tune->n_profiles;

    // interpolate.

    double x = 0.0;
    if (prev != next) {
        x = 1.0 * (M - prev->M) / (next->M - prev->M);
    }

    for (int i = 0; i < tune->n_profiles; i++) {
        time_stats->profile_time[i].total_time = 0;
        for (int j = 0; j < 3; j++) {
            if (tune->conf[i][j].backend == GGML_BACKEND_UNKNOWN) {
                continue;
            }

            int pv = (prev + i)->stages_time[j];
            double t = pv;
            if (x > 0) {
                t += x * ((next + i)->stages_time[j] - pv);
            }
            if (tune->conf[i][j].parallel) {
                t /= nth;
            }
            time_stats->profile_time[i].stage_time[j] = t;
            time_stats->profile_time[i].total_time += t;
        }
    }

    return 0;
}

static const char *ggml_backend_names[] = {
    [GGML_BACKEND_UNKNOWN] = "UNKNOWN",
    [GGML_BACKEND_CPU] = "CPU",
    [GGML_BACKEND_CUDA] = "CUDA", // CUBLAS
    [GGML_BACKEND_ACCELERATE] = "ACCELERATE",
    [GGML_BACKEND_OPENBLAS] = "OPENBLAS",
    [GGML_BACKEND_CLBLAST] = "CLBLAST",
};

const char *ggml_get_backend_name(void) {
#if defined(GGML_USE_ACCELERATE)
    return ggml_backend_names[GGML_BACKEND_ACCELERATE];
#elif defined(GGML_USE_CLBLAST)
    return ggml_backend_names[GGML_BACKEND_CLBLAST];
#elif defined(GGML_USE_CUBLAS)
    return ggml_backend_names[GGML_BACKEND_CUBLAS];
#elif defined(GGML_USE_OPENBLAS)
    return ggml_backend_names[GGML_BACKEND_OPENBLAS];
#else
    return ggml_backend_names[GGML_BACKEND_CPU];
#endif
}

// See `enum ggml_backend` in `ggml.h`.
int ggml_get_backend(void) {
#if defined(GGML_USE_CUBLAS)
    return GGML_BACKEND_CUDA;
#elif defined(GGML_USE_ACCELERATE)
    return GGML_BACKEND_ACCELERATE;
#elif defined(GGML_USE_OPENBLAS)
    return GGML_BACKEND_OPENBLAS;
#elif defined(GGML_USE_CLBLAST)
    return GGML_BACKEND_CLBLAST;
#else
    return GGML_BACKEND_CPU;
#endif
}

void ggml_mulmat_tune_setup_task_conf(struct ggml_mulmat_tune *tune) {
    tune->n_profiles = 1;
    // cpu init + cpu compute
    tune->conf[0][0] = (struct ggml_task_conf){
        .backend = GGML_BACKEND_CPU,
    };
    tune->conf[0][1] = (struct ggml_task_conf){
        .backend = GGML_BACKEND_CPU,
        .parallel = true,
    };
#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)
    tune->n_profiles++;
    // cpu init + gpu compute
    tune->conf[1][0] = (struct ggml_task_conf){
        .backend = GGML_BACKEND_CPU,
        .parallel = true,
    };
    tune->conf[1][1] = (struct ggml_task_conf){
        .backend = tune->gpu_backend,
        .wait = true,
    };
#elif defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
    tune->n_profiles++;
    // gpu only: compute
    tune->conf[1][1] = (struct ggml_task_conf){
        .backend = tune->gpu_backend,
        .wait = true,
    };
#else
    abort();
#endif
}
