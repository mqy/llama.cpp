#include "examples/mulmat-tune/mulmat-tune.h"
#include "ggml.h"

#if defined GGML_USE_CLBLAST
#include "ggml-opencl.h"
#endif

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define BENCH_ASSERT_EQUAL(actual, expect, fmt, ...)                           \
    do {                                                                       \
        if (expect != actual) {                                                \
            fprintf(stderr,                                                    \
                    "test fail. line: %d: expect: %d, actual: %d. " fmt "\n",  \
                    __LINE__, expect, actual, __VA_ARGS__);                    \
        } else {                                                               \
            printf("test pass\n");                                             \
        }                                                                      \
    } while (0)

#define UNUSED(x) (void)(x)

static int tune_time_min(int *a, int len);
static void print_blas_build_tips(void);
static void progress(int i, int max);
static void print_envs_for_build(enum ggml_backend, char *buf, int buf_len);
static bool prompt_yes_no(const char *prompt);

static void cmd_tune(struct ggml_mulmat_tune *b, bool verbose);
static void cmd_analyze(struct ggml_mulmat_tune *b);
static void cmd_test(void);

static void test_select_profile(void);

static void usage(char *prog) {
    const char *usage_lines[] = {
        "usage: %s [bench ...] | [analyze FILE] | test | help\n",
        "\n",
        "bench [-m MODEL] [-t TYPE] [-f FILE] [-y]\n",
        "-model  MODEL  7B | 13B | 30B | 65B\n",
        "               default 7B\n",
        "-type   TYPE   Q4_0 |  Q4_1 | Q5_0 | Q5_1 | Q8_0 | ...\n",
        "               default Q4_0\n",
        "-m_step M_STEP the step of M, also as start value\n",
        "               suggest M_STEP %% 8 == 0\n",
        "               default 8\n",
        "-m_num  M_NUM  number of M, total M = M_STEP * M_NUM\n",
        "               default 16\n",
        "-file   FILE   data file to write\n",
        "               default stdout\n",
        "-y             always answer \"yes\" to all prompts\n",
    };

    int len = (int)(sizeof(usage_lines) / sizeof(char *));
    for (int i = 0; i < len; i++) {
        const char *line = usage_lines[i];
        if (i == 0) {
            fprintf(stderr, line, prog);
        } else {
            fprintf(stderr, "%s", line);
        }
    }
}

// main
int main(int argc, char **argv) {
    enum ggml_backend gpu_backend = ggml_get_backend();
    if (gpu_backend == GGML_BACKEND_CPU) {
        print_blas_build_tips();
        exit(1);
    }

    char *cmd = NULL;
    if (argc == 1) {
        cmd = "bench";
    } else {
        cmd = argv[1];
    }

    if (strcmp(cmd, "bench") == 0) {
        struct ggml_mulmat_tune tune = {
            .version = 1,
            .n_shapes = 0,
            .m_step = 8,
            .m_num = 16,
            .gpu_backend = gpu_backend,
        };

        {
            const char *name = ggml_get_backend_name();
            int n = sizeof(tune.gpu_backend_name);
            strncpy(tune.gpu_backend_name, name, n);
        }

        ggml_mulmat_tune_setup_task_conf(&tune);

        const char *arg_model = NULL;
        const char *arg_q_type = NULL;
        const char *arg_m_step = NULL;
        const char *arg_m_num = NULL;
        const char *arg_file = NULL;
        bool always_yes = false;

        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "-model") == 0) {
                if (i + 1 < argc) {
                    arg_model = argv[i + 1];
                    ++i;
                }
            } else if (strcmp(argv[i], "-q_type") == 0) {
                if (i + 1 < argc) {
                    arg_q_type = argv[i + 1];
                    ++i;
                }
            } else if (strcmp(argv[i], "-m_step") == 0) {
                if (i + 1 < argc) {
                    arg_m_step = argv[i + 1];
                    ++i;
                }
            } else if (strcmp(argv[i], "-m_num") == 0) {
                if (i + 1 < argc) {
                    arg_m_num = argv[i + 1];
                    ++i;
                }
            } else if (strcmp(argv[i], "-file") == 0) {
                if (i + 1 < argc) {
                    arg_file = argv[i + 1];
                    ++i;
                }
            } else if (strcmp(argv[i], "-y") == 0) {
                always_yes = true;
            } else {
                fprintf(stderr, "invalid arg: %s\n", argv[i]);
                usage(argv[0]);
                exit(1);
            }
        }

        FILE *fp = NULL;

        if (arg_file != NULL && !always_yes) {
            struct stat st;
            int rc = stat(arg_file, &st);
            UNUSED(st);
            if (rc == 0) { // prompt
                size_t len = strlen(arg_file) + 50;
                char *prompt = malloc(len);
                GGML_ASSERT(prompt);
                snprintf(prompt, len, "data file '%s' exists, override? (Y|n)",
                         arg_file);

                if (!prompt_yes_no(prompt)) {
                    printf("Aborted.\n");
                    exit(2);
                }
                free(prompt);
            }

            fp = fopen(arg_file, "w");
            GGML_ASSERT(fp);
        }

        if (arg_model == NULL) {
            arg_model = "7B";
        }
        ggml_mulmat_tune_setup_model(&tune, arg_model);

        if (arg_q_type == NULL) {
            arg_q_type = "Q4_0";
        }

        int n = sizeof(arg_q_type);
        strncpy(tune.type_name, arg_q_type, n);

        if (arg_m_step != NULL) {
            int v = atoi(arg_m_step);
            if (v <= 0) {
                fprintf(stderr, "invalid m_step: %s\n", arg_m_step);
                usage(argv[0]);
            }
            tune.m_step = v;
        }

        if (arg_m_num != NULL) {
            int v = atoi(arg_m_num);
            if (v <= 0) {
                fprintf(stderr, "invalid m_step: %s\n", arg_m_num);
                usage(argv[0]);
            }
            tune.m_num = v;
        }

        {
            size_t sz = sizeof(struct ggml_mulmat_tune_m) *
                        (tune.n_shapes * tune.m_num * tune.n_profiles);
            tune.items = malloc(sz);
            GGML_ASSERT(tune.items);
            memset(tune.items, 0, sz);
        }

#if defined GGML_USE_CLBLAST
        ggml_cl_init();
#endif

        printf("[BENCH] model: %s, type: %s, GPU backend: %s.\n\n", tune.model,
               tune.type_name, tune.gpu_backend_name);

        cmd_tune(&tune, true /* verbose */);

        int rc = ggml_mulmat_tune_write_data(&tune, fp == NULL ? stdout : fp);
        if (fp != NULL) {
            fclose(fp);
        }
        if (rc != 0) {
            printf("failed to write bench result to %s\n", arg_file);
            exit(1);
        }

        if (arg_file != NULL) {
            printf("result was written to %s\n", arg_file);
        }
    } else if (strcmp(cmd, "analyze") == 0) {
        if (argc < 3) {
            fprintf(stderr, "error: too few args\n");
            usage(argv[0]);
            exit(1);
        }

        struct ggml_mulmat_tune tune;

        char *data_file = argv[2];
        {
            struct stat st;
            int rc = stat(data_file, &st);
            UNUSED(st);
            if (rc != 0) {
                fprintf(stderr, "error: data file not exists: %s\n", data_file);
                exit(1);
            }
        }

        FILE *fp = fopen(data_file, "r");
        GGML_ASSERT(fp);
        int rc = ggml_mulmat_tune_read_data(&tune, fp);
        GGML_ASSERT(rc == 0);
        fclose(fp);

        cmd_analyze(&tune);
    } else if (strcmp(cmd, "test") == 0) {
        if (argc != 2) {
            fprintf(stderr, "error: invalid args\n");
            usage(argv[0]);
            exit(1);
        }
        cmd_test();
    } else if (strcmp(cmd, "help") == 0) {
        if (argc != 2) {
            fprintf(stderr, "error: invalid args\n");
            usage(argv[0]);
            exit(1);
        }
        usage(argv[0]);
    } else {
        fprintf(stderr, "error: unknown command: %s.\n", cmd);
        usage(argv[0]);
        exit(1);
    }

    return 0;
}

void cmd_tune(struct ggml_mulmat_tune *tune, bool verbose) {
    size_t wsize = 0;
    void *q_buf = NULL;
    void *wdata = NULL;

    // alloc q4_0_buf and wdata with max size.
    {
        int max_NxK = 0;
        for (int i = 0; i < tune->n_shapes; i++) {
            int sz = tune->shapes[i].N * tune->shapes[i].K;
            if (sz > max_NxK) {
                max_NxK = sz;
            }
        }

        size_t q_buf_size;
        if (strcmp(tune->type_name, "Q4_0") == 0) {
            q_buf_size = 2 * max_NxK * sizeof(int64_t);
        } else if (strcmp(tune->type_name, "Q4_1") == 0) {
            q_buf_size = 2 * max_NxK * sizeof(int64_t);
        } else {
            abort();
        }

        q_buf = malloc(q_buf_size);
        if (!q_buf) {
            fprintf(stderr,
                    "failed to allocate memory for q_buf, size: %zu MiB\n",
                    q_buf_size / 1024 / 1024);
            exit(1);
        }
        wsize = max_NxK * sizeof(float);
        wdata = malloc(wsize);
        if (!wdata) {
            fprintf(stderr,
                    "failed to allocate memory for wdata, size: %zu MiB\n",
                    wsize / 1024 / 1024);
            exit(1);
        }
    }

    for (int i_shape = 0; i_shape < tune->n_shapes; i_shape++) {
        int M;
        int N = tune->shapes[i_shape].N;
        int K = tune->shapes[i_shape].K;

        char progress_line[20];
        int line_len;

        for (int i_m = 0; i_m < tune->m_num; i_m++) {
            M = tune->m_step * (i_m + 1);

            if (verbose) {
                memset(progress_line, 0, sizeof(progress_line));
                snprintf(progress_line, sizeof(progress_line), "%d %d %d ", N,
                         K, M);
                printf("%s", progress_line);
                fflush(stdout);

                line_len = strlen(progress_line);
            }

            // TODO: not use ctx?

            struct ggml_context *ctx = NULL;
            {
                // TODO: the ctx_size is over estimated, fix it.
                size_t ctx_size = K * N * ggml_type_sizef(GGML_TYPE_F32) +
                                  K * sizeof(float) + 1024 * 1024 * 300;

                struct ggml_init_params init_params = {
                    .mem_size = ctx_size,
                    .mem_buffer = NULL,
                    .no_alloc = 0,
                };

                ctx = ggml_init(init_params);
                GGML_ASSERT(ctx);
            }

            struct ggml_tensor *src0 = NULL;
            struct ggml_tensor *src1 = NULL;
            struct ggml_tensor *dst = NULL;

            {
                // src0: K x N
                struct ggml_tensor *src0_f32 = ggml_new_tensor_2d(
                    ctx, GGML_TYPE_F32, (int64_t)K, (int64_t)N);
                ggml_set_f32(src0_f32, 0.1f);

                enum ggml_type q_type;
                if (strcmp(tune->type_name, "Q4_0") == 0) {
                    q_type = GGML_TYPE_Q4_0;
                } else if (strcmp(tune->type_name, "Q4_1") == 0) {
                    q_type = GGML_TYPE_Q4_1;
                } else {
                    abort();
                }

                src0 = ggml_new_tensor_2d(ctx, q_type, (int64_t)K, (int64_t)N);

                switch (q_type) {
                case GGML_TYPE_Q4_0:
                    ggml_quantize_q4_0((const float *)src0_f32->data,
                                       src0->data, N * K, K, (int64_t *)q_buf);
                    break;
                case GGML_TYPE_Q4_1:
                    ggml_quantize_q4_1((const float *)src0_f32->data,
                                       src0->data, N * K, K, (int64_t *)q_buf);
                    break;
                default:
                    // TODO
                    abort();
                }

                // src1: M x K
                src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)K,
                                          (int64_t)M);
                ggml_set_f32(src1, 0.5f);

                // dst: M x N
                dst = ggml_mul_mat(ctx, src0, src1);
            }

            for (int ip = 0; ip < tune->n_profiles; ip++) {
                struct ggml_task_conf *conf = tune->conf[ip];

                int item_index =
                    (i_shape * tune->m_num + i_m) * tune->n_profiles + ip;
                struct ggml_mulmat_tune_m *item = &tune->items[item_index];

                item->M = M;

                for (int stage = 0; stage < 3; stage++) {
                    item->stages_time[stage] = 0;
                    if (conf[stage].backend == GGML_BACKEND_UNKNOWN) {
                        continue;
                    }

                    // without memset, the first run may be significant slow.
                    memset(wdata, 0, wsize);

                    int stage_time[NUM_BENCH];
                    for (int i_bench = 0; i_bench < NUM_BENCH; i_bench++) {
                        int t0 = (int)ggml_time_us();

                        ggml_internal_compute_forward_mul_mat(
                            conf, stage, wsize, wdata, src0, src1, dst);

                        stage_time[i_bench] = (int)ggml_time_us() - t0;
                        if (verbose) {
                            progress(i_bench, NUM_BENCH);
                        }
                    }

                    item->stages_time[stage] =
                        tune_time_min(stage_time, NUM_BENCH);

                    if (verbose) {
                        line_len++;
                    }
                }
            }

            if (verbose) {
                // + 10: clear at most these additional chars that may be
                // unexpectedly pressed or pasted.
                line_len += 10;
                for (int j = 0; j < line_len; j++) {
                    printf("\b \b");
                }
                fflush(stdout);
            }

            ggml_free(ctx);
        }
    }

    free(wdata);
    free(q_buf);
}

static void print_blas_build_tips(void) {
    const char *make_target = "mulmat-tune";

    fprintf(stderr, "error: this program was not built with any BLAS. tips:\n");

    char buf[100];
    print_envs_for_build(GGML_BACKEND_ACCELERATE, buf, 100);
    fprintf(stderr, "* to build with Accelerate: make clean; %s make %s\n", buf,
            make_target);
    print_envs_for_build(GGML_BACKEND_OPENBLAS, buf, 100);
    fprintf(stderr, "* to build with openBLAS:   make clean; %s make %s\n", buf,
            make_target);
    print_envs_for_build(GGML_BACKEND_CUDA, buf, 100);
    fprintf(stderr, "* to build with cuBLAS:     make clean; %s make %s\n", buf,
            make_target);
    print_envs_for_build(GGML_BACKEND_CL, buf, 100);
    fprintf(stderr, "* to build with CLBLast:    make clean; %s make %s\n", buf,
            make_target);
}

static void print_envs_for_build(enum ggml_backend backend, char *buf,
                                 int buf_len) {
    memset(buf, 0, buf_len);
    const char *LLAMA_NO_ACCELERATE =
        backend == GGML_BACKEND_ACCELERATE ? " " : "1";
    const char *LLAMA_OPENBLAS = backend == GGML_BACKEND_OPENBLAS ? "1" : " ";
    const char *LLAMA_CUBLAS = backend == GGML_BACKEND_CUDA ? "1" : " ";
    const char *LLAMA_CLBLAST = backend == GGML_BACKEND_CL ? "1" : " ";

    snprintf(buf, buf_len,
             "LLAMA_NO_ACCELERATE=%s LLAMA_OPENBLAS=%s LLAMA_CUBLAS=%s "
             "LLAMA_CLBLAST=%s",
             LLAMA_NO_ACCELERATE, LLAMA_OPENBLAS, LLAMA_CUBLAS, LLAMA_CLBLAST);
}

static bool prompt_yes_no(const char *prompt) {
    char buf[2];
    while (true) {
        fprintf(stderr, "%s\n", prompt);
        buf[0] = 0;
        buf[1] = 0;
        int i = 0;
        int c = 0;

        while (c != '\n') {
            c = fgetc(stdin);
            buf[i % 2] = c;
            i++;
        }
        if (i == 1) {
            if (buf[0] == '\n') {
                return true;
            }
        } else if (i == 2) {
            if (buf[0] == 'Y' || buf[0] == 'y') {
                return true;
            } else if (buf[0] == 'N' || buf[0] == 'n') {
                return false;
            }
        }
    }
}

static void progress(int i, int n) {
    char tokens[4] = {'|', '/', '-', '\\'};
    if (i > 0) {
        printf("\b \b");
    }
    if (i + 1 < n) {
        putchar(tokens[i % 4]);
    } else {
        putchar('.');
    }
    fflush(stdout);
}

static int tune_time_min(int *a, int len) {
    int min = INT32_MAX;
    for (int i = 0; i < len; i++) {
        if (a[i] < min) {
            min = a[i];
        }
    }
    return min;
}

static void cmd_analyze(struct ggml_mulmat_tune *tune) {
    int m_num = tune->m_num;

    for (int i_shape = 0; i_shape < tune->n_shapes; i_shape++) {
        struct ggml_mulmat_tune_shape *shape = &tune->shapes[i_shape];

        const int nth_arr[] = {1, 2, 4, 6, 8};
        int nth_arr_len = (int)(sizeof(nth_arr) / sizeof(int));
        for (int i = 0; i < nth_arr_len; i++) {
            int nth = nth_arr[i];

            printf("N=%d,K=%d,nth=%d\n\n", shape->N, shape->K, nth);

            printf("#M      ");
            for (int i_m = 0; i_m < m_num; i_m++) {
                int item_index =
                    (i_shape * tune->m_num + i_m) * tune->n_profiles;
                printf(";%7d", tune->items[item_index].M);
            }
            printf("\n");

            for (int ip = 0; ip < tune->n_profiles; ip++) {
                struct ggml_task_conf *task_conf = tune->conf[ip];
                int *total_time = malloc(sizeof(int) * m_num);
                for (int k = 0; k < 3; k++) {
                    enum ggml_backend backend = task_conf[k].backend;
                    if (backend == GGML_BACKEND_UNKNOWN) {
                        continue;
                    }
                    char *backend_name =
                        backend == GGML_BACKEND_CPU ? "CPU" : "GPU";
                    printf("#%d_%d_%s", ip, k, backend_name);

                    for (int im = 0; im < m_num; im++) {
                        int item_index =
                            (i_shape * tune->m_num + im) * tune->n_profiles +
                            ip;
                        int stage_time = tune->items[item_index].stages_time[k];
                        if (task_conf[k].parallel) {
                            stage_time /= nth;
                        }
                        printf(";%7d", stage_time);
                        total_time[im] += stage_time;
                    }
                    printf("\n");
                }

                printf("#%d_total", ip);
                for (int im = 0; im < m_num; im++) {
                    printf(";%7d", total_time[im]);
                }
                printf("\n");
                free(total_time);
            }
            printf("\n");
        }
    }
}

static void cmd_test(void) {
    printf("\n=== test select profile\n\n");
    test_select_profile();
}

struct test_select_profile_data {
    int nth;
    int M;
    int profile_0_time;
    int profile_1_time;
};

static void test_select_profile(void) {
    struct ggml_mulmat_tune tune = {
        .version = 1,
        .type_name = "Q4_0",
        .gpu_backend = GGML_BACKEND_OPENBLAS,
        .gpu_backend_name = "OpenBLAS",
        .m_step = 2,
        .m_num = 2,
    };
    ggml_mulmat_tune_setup_model(&tune, "7B");
    ggml_mulmat_tune_setup_task_conf(&tune);

    size_t sz = sizeof(struct ggml_mulmat_tune_m) *
                (tune.n_shapes * tune.m_num * tune.n_profiles);
    tune.items = malloc(sz);
    GGML_ASSERT(tune.items);
    memset(tune.items, 0, sz);

    // shape 0, profile 0
    tune.items[0] =
        (struct ggml_mulmat_tune_m){.M = 2, .stages_time = {2, 4, 0}};
    tune.items[1] =
        (struct ggml_mulmat_tune_m){.M = 2, .stages_time = {4, 4, 0}};
    // shape 0, profile 1
    tune.items[2] =
        (struct ggml_mulmat_tune_m){.M = 4, .stages_time = {4, 8, 0}};
    tune.items[3] =
        (struct ggml_mulmat_tune_m){.M = 4, .stages_time = {4, 4, 0}};

    const int N = tune.shapes[0].N;
    const int K = tune.shapes[0].K;

    // When M out of range.
    {
        const int M_arr[2] = {tune.items[0].M - 1, tune.items[2].M + 1};
        int n = (int)(sizeof(M_arr) / sizeof(int));

        for (int i = 1; i <= 8; i++) {
            int nth = i;
            for (int j = 0; j < n; j++) {
                struct ggml_mulmat_tune_time_stats time_stats;
                int rc = ggml_mulmat_tune_time_stats(&tune, M_arr[j], N, K, nth,
                                                     &time_stats);
                BENCH_ASSERT_EQUAL(rc, -1, "#(i: %d, i: %d)", i, j);
            }
        }
    }

    // When M in range.
    {
        const struct test_select_profile_data test_data[] = {
            {
                .nth = 1,
                .M = 2,
                .profile_0_time = 6,
                .profile_1_time = 8,
            },
            {
                .nth = 1,
                .M = 4,
                .profile_0_time = 12,
                .profile_1_time = 8,
            },
            {
                .nth = 1,
                .M = 3,
                .profile_0_time = 9,
                .profile_1_time = 8,
            },
            {
                .nth = 2,
                .M = 2,
                .profile_0_time = 4,
                .profile_1_time = 6,
            },
            {
                .nth = 2,
                .M = 4,
                .profile_0_time = 8,
                .profile_1_time = 6,
            },
            {
                .nth = 2,
                .M = 3,
                .profile_0_time = 6,
                .profile_1_time = 6,
            }};

        int n =
            (int)(sizeof(test_data) / sizeof(struct test_select_profile_data));

        for (int i = 0; i < n; i++) {
            const struct test_select_profile_data *e = &test_data[i];
            struct ggml_mulmat_tune_time_stats time_stats;
            int rc = ggml_mulmat_tune_time_stats(&tune, e->M, N, K, e->nth,
                                                 &time_stats);
            BENCH_ASSERT_EQUAL(rc, 0, "#(i: %d)", i);
            BENCH_ASSERT_EQUAL(time_stats.profile_time[0].total_time,
                               e->profile_0_time, "#(i: %d)", i);
            BENCH_ASSERT_EQUAL(time_stats.profile_time[1].total_time,
                               e->profile_1_time, "#(i: %d)", i);
        }
    }
}
