#include "examples/mulmat-tune/mulmat-tune.h"
#include "ggml.h"

#if defined GGML_USE_CLBLAST
#include "ggml-opencl.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <inttypes.h>
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
static void print_envs_for_build(enum ggml_blas_type, char *buf, int buf_len);
static bool prompt_yes_no(const char *prompt);

static void cmd_tune(struct ggml_mulmat_tune *b);
static void cmd_analyze(struct ggml_mulmat_tune *b);
static void cmd_test(void);

static void test__estimate_time(void);
static void test__choose_device(void);

static void usage(char *prog) {
    const char *usage_lines[] = {
        "usage: %s [tune ...] | [analyze FILE] | test | help\n\n",
        "tune [-m MODEL] [-t TYPE] [-f FILE] [-y]\n",
        "-model  MODEL   7B | 13B | 30B | 64B\n",
        "                default 7B\n",
        "-q_type TYPE    Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1\n",
        "                default Q4_0\n",
        "-step_m STEP_M  the step of M, also as start value\n",
        "                suggest STEP_M %% 8 == 0\n",
        "                default 8\n",
        "-num_m  NUM_M   number of M, total M = STEP_M * NUM_M\n",
        "                default 16\n",
        "-file   FILE    data file to write\n",
        "                default stdout\n",
        "-y              always answer \"yes\" to all prompts\n",
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
    if (!ggml_cpu_has_blas()) {
        print_blas_build_tips();
        exit(1);
    }

    char *cmd = NULL;
    if (argc == 1) {
        cmd = "tune";
    } else {
        cmd = argv[1];
    }

    if (strcmp(cmd, "tune") == 0) {
        struct ggml_mulmat_tune tune = {
            .version = 1,
            .n_groups = 0,
            .step_m = 8,
            .num_m = 16,
            .cpu_only_stages = {GGML_TASK_FLAG_T1, GGML_TASK_FLAG_TN, 0},
#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)
            .use_blas_stages = {GGML_TASK_FLAG_TN, GGML_TASK_FLAG_T1_WAIT, 0},
#elif defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
            .use_blas_stages = {0, GGML_TASK_FLAG_T1, 0},
#endif
            .groups = NULL,
        };

        const char *arg_model = NULL;
        const char *arg_q_type = NULL;
        const char *arg_step_m = NULL;
        const char *arg_num_m = NULL;
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
            } else if (strcmp(argv[i], "-step_m") == 0) {
                if (i + 1 < argc) {
                    arg_step_m = argv[i + 1];
                    ++i;
                }
            } else if (strcmp(argv[i], "-num_m") == 0) {
                if (i + 1 < argc) {
                    arg_num_m = argv[i + 1];
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
                fprintf(stderr, "[%s]: invalid arg: %s\n", cmd, argv[i]);
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
                snprintf(prompt, len,
                         "[%s]: data file '%s' exists, override? (Y|n)", cmd,
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

        if (strcmp(arg_model, "7B") == 0) {
            tune.n_groups = 4;
            tune.groups =
                malloc(tune.n_groups * sizeof(struct ggml_mulmat_tune_nk));
            GGML_ASSERT(tune.groups);
            tune.groups[0] = (struct ggml_mulmat_tune_nk){
                .N = 4096, .K = 4096, .items = NULL};
            tune.groups[1] = (struct ggml_mulmat_tune_nk){
                .N = 4096, .K = 11008, .items = NULL};
            tune.groups[2] = (struct ggml_mulmat_tune_nk){
                .N = 11008, .K = 4096, .items = NULL};
            tune.groups[3] = (struct ggml_mulmat_tune_nk){
                .N = 32000, .K = 4096, .items = NULL};
        } else if (strcmp(arg_model, "13B") == 0) {
            tune.n_groups = 4;
            tune.groups =
                malloc(tune.n_groups * sizeof(struct ggml_mulmat_tune_nk));
            GGML_ASSERT(tune.groups);
            tune.groups[0] = (struct ggml_mulmat_tune_nk){
                .N = 5120, .K = 5120, .items = NULL};
            tune.groups[1] = (struct ggml_mulmat_tune_nk){
                .N = 5120, .K = 13824, .items = NULL};
            tune.groups[2] = (struct ggml_mulmat_tune_nk){
                .N = 13824, .K = 5120, .items = NULL};
            tune.groups[3] = (struct ggml_mulmat_tune_nk){
                .N = 32000, .K = 5120, .items = NULL};
        } else if (strcmp(arg_model, "30B") == 0) {
            // TODO
            abort();
        } else if (strcmp(arg_model, "30B") == 0) {
            // TODO
            abort();
        } else {
            fprintf(stderr, "[%s]: error: unknown model: %s\n", cmd, arg_model);
            usage(argv[0]);
            exit(1);
        }

        enum ggml_type q_type;
        if (arg_q_type == NULL) {
            arg_q_type = "Q4_0";
            q_type = GGML_TYPE_Q4_0;
        }

        if (strcmp(arg_q_type, "Q4_0") == 0) {
            q_type = GGML_TYPE_Q4_0;
        } else if (strcmp(arg_q_type, "Q4_1") == 0) {
            q_type = GGML_TYPE_Q4_1;
        } else if (strcmp(arg_q_type, "Q5_0") == 0) {
            q_type = GGML_TYPE_Q5_0;
        } else if (strcmp(arg_q_type, "Q5_1") == 0) {
            q_type = GGML_TYPE_Q5_1;
        } else if (strcmp(arg_q_type, "Q8_0") == 0) {
            q_type = GGML_TYPE_Q8_0;
        } else if (strcmp(arg_q_type, "Q8_1") == 0) {
            q_type = GGML_TYPE_Q8_1;
        } else {
            fprintf(stderr, "[%s]: error: unsupported q_type: %s\n", cmd,
                    arg_q_type);
            usage(argv[0]);
            exit(1);
        }

        // tune.q_type, tune.arg_q_type
        {
            tune.q_type = q_type;
            size_t n = sizeof(tune.q_type_name);
            GGML_ASSERT(n > strlen(arg_q_type));
            strncpy(tune.q_type_name, arg_q_type, n);
        }

        // tune.model
        {
            size_t n = sizeof(tune.model);
            GGML_ASSERT(n > strlen(arg_model));
            strncpy(tune.model, arg_model, n);
        }

        // tune.model
        {
            if (arg_step_m != NULL) {
                int step_m = atoi(arg_step_m);
                if (step_m <= 0) {
                    fprintf(stderr, "[%s]: error: invalid step_m: %s\n", cmd,
                            arg_step_m);
                    usage(argv[0]);
                    exit(1);
                }
                tune.step_m = step_m;
            }
            if (arg_num_m != NULL) {
                int num_m = atoi(arg_num_m);
                if (num_m <= 0) {
                    fprintf(stderr, "[%s]: error: invalid num_m: %s\n", cmd,
                            arg_num_m);
                    usage(argv[0]);
                    exit(1);
                }
                tune.num_m = num_m;
            }
        }

        // tune.model
        {
            size_t n = sizeof(tune.model);
            GGML_ASSERT(n > strlen(arg_model));
            strncpy(tune.model, arg_model, n);
        }

        // tune.blas_name
        {
            const char *blas_name = ggml_get_blas_name();
            GGML_ASSERT(blas_name);

            size_t n = sizeof(tune.blas_name);
            GGML_ASSERT(n > strlen(blas_name));
            strncpy(tune.blas_name, blas_name, n);
        }

#if defined GGML_USE_CLBLAST
        ggml_cl_init();
#endif

        printf("[BENCH] model name: %s, q_type: %s, blas: %s.\n", tune.model,
               tune.q_type_name, tune.blas_name);

        cmd_tune(&tune);

        ggml_mulmat_write_tune_data(&tune, fp == NULL ? stdout : fp);
        if (fp != NULL) {
            fclose(fp);
        }

        if (arg_file != NULL) {
            printf("[%s]: result was written to %s\n", cmd, arg_file);
        }
    } else if (strcmp(cmd, "analyze") == 0) {
        if (argc < 3) {
            fprintf(stderr, "[%s]: error: too few args\n", cmd);
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
                fprintf(stderr, "[%s]: error: data file not exists: %s\n", cmd,
                        data_file);
                exit(1);
            }
        }

        FILE *fp = fopen(data_file, "r");
        GGML_ASSERT(fp);
        int rc = ggml_mulmat_read_tune_data(&tune, fp);
        GGML_ASSERT(rc == 0);
        fclose(fp);

        cmd_analyze(&tune);
    } else if (strcmp(cmd, "test") == 0) {
        if (argc != 2) {
            fprintf(stderr, "[%s]: error: invalid args\n", cmd);
            usage(argv[0]);
            exit(1);
        }
        cmd_test();
    } else if (strcmp(cmd, "help") == 0) {
        if (argc != 2) {
            fprintf(stderr, "[%s]: error: invalid args\n", cmd);
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

void cmd_tune(struct ggml_mulmat_tune *tune) {
    size_t wsize = 0;
    void *q_buf = NULL;
    void *wdata = NULL;

    // alloc q4_0_buf and wdata with max size.
    {
        int max_NxK = 0;
        for (int i = 0; i < tune->n_groups; i++) {
            int sz = tune->groups[i].N * tune->groups[i].K;
            if (sz > max_NxK) {
                max_NxK = sz;
            }
        }

        size_t q_buf_size;
        switch (tune->q_type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            q_buf_size = 2 * max_NxK * sizeof(int64_t);
            break;
        default:
            // TODO
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

    for (int i = 0; i < tune->n_groups; i++) {
        struct ggml_mulmat_tune_nk *group = &tune->groups[i];
        int M;
        int N = group->N;
        int K = group->K;

        {
            size_t sz = sizeof(struct ggml_mulmat_tune_m) * tune->num_m;
            group->items = malloc(sz);
            GGML_ASSERT(group->items);
            memset(group->items, 0, sz);
        }

        char progress_line[20];

        for (int im = 0; im < tune->num_m; im++) {
            M = tune->step_m * (im + 1);

            memset(progress_line, 0, sizeof(progress_line));
            snprintf(progress_line, sizeof(progress_line), "%d %d %d ", N, K,
                     M);
            printf("%s", progress_line);
            fflush(stdout);

            int line_len = strlen(progress_line);

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

                src0 = ggml_new_tensor_2d(ctx, tune->q_type, (int64_t)K,
                                          (int64_t)N);

                switch (tune->q_type) {
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

            struct ggml_mulmat_tune_m *tune_item = &group->items[im];
            tune_item->M = M;

            ggml_task_flag_set_blas(&dst->task_flag, 0);
            for (int stage = 0; stage < 3; stage++) {
                if (tune->cpu_only_stages[stage] > 0) {
                    // without this, the first run may be significant slow.
                    memset(wdata, 0, wsize);

                    for (int nb = 0; nb < NUM_BENCH; nb++) {
                        int t0 = (int)ggml_time_us();
                        ggml_internal_compute_forward_mul_mat_q_f32_for_fine_tune(
                            stage, wsize, wdata, src0, src1, dst);
                        tune_item->cpu_only_records[stage][nb] =
                            (int)ggml_time_us() - t0;
                        progress(nb, NUM_BENCH);
                    }
                    line_len++;
                }
            }

            ggml_task_flag_set_blas(&dst->task_flag, 1);
            for (int stage = 0; stage < 3; stage++) {
                if (tune->use_blas_stages[stage] > 0) {
                    for (int nb = 0; nb < NUM_BENCH; nb++) {
                        int t0 = (int)ggml_time_us();
                        ggml_internal_compute_forward_mul_mat_q_f32_for_fine_tune(
                            stage, wsize, wdata, src0, src1, dst);
                        tune_item->use_blas_records[stage][nb] =
                            (int)ggml_time_us() - t0;
                        progress(nb, NUM_BENCH);
                    }
                    line_len++;
                }
            }

            line_len += 20; // + 20: clear at most these additional chars that
                            // user unexpectedly pressed or pasted.
            for (int j = 0; j < line_len; j++) {
                printf("\b \b");
            }
            fflush(stdout);

            ggml_free(ctx);
        }
    }

    free(wdata);
    free(q_buf);

    // collect stat records.
    for (int i = 0; i < tune->n_groups; i++) {
        for (int j = 0; j < tune->num_m; j++) {
            struct ggml_mulmat_tune_m *item = &tune->groups[i].items[j];
            for (int stage = 0; stage < 3; stage++) {
                if (tune->cpu_only_stages[stage] > 0) {
                    item->cpu_only_time[stage] =
                        tune_time_min(item->cpu_only_records[stage], NUM_BENCH);
                }
                if (tune->use_blas_stages[stage] > 0) {
                    item->use_blas_time[stage] =
                        tune_time_min(item->use_blas_records[stage], NUM_BENCH);
                }
            }
        }
    }
}

static void print_blas_build_tips(void) {
    const char *make_target = "mulmat-tune";

    fprintf(stderr, "error: this program was not built with any BLAS. tips:\n");

    char buf[100];
    print_envs_for_build(GGML_BLAS_TYPE_ACCELERATE, buf, 100);
    fprintf(stderr, "* to build with Accelerate: make clean; %s make %s\n", buf,
            make_target);
    print_envs_for_build(GGML_BLAS_TYPE_OPENBLAS, buf, 100);
    fprintf(stderr, "* to build with openBLAS:   make clean; %s make %s\n", buf,
            make_target);
    print_envs_for_build(GGML_BLAS_TYPE_CUBLAS, buf, 100);
    fprintf(stderr, "* to build with cuBLAS:     make clean; %s make %s\n", buf,
            make_target);
    print_envs_for_build(GGML_BLAS_TYPE_CLBLAST, buf, 100);
    fprintf(stderr, "* to build with CLBLast:    make clean; %s make %s\n", buf,
            make_target);
}

static void print_envs_for_build(enum ggml_blas_type blas, char *buf,
                                 int buf_len) {
    memset(buf, 0, buf_len);
    const char *LLAMA_NO_ACCELERATE =
        blas == GGML_BLAS_TYPE_ACCELERATE ? " " : "1";
    const char *LLAMA_OPENBLAS = blas == GGML_BLAS_TYPE_OPENBLAS ? "1" : " ";
    const char *LLAMA_CUBLAS = blas == GGML_BLAS_TYPE_CUBLAS ? "1" : " ";
    const char *LLAMA_CLBLAST = blas == GGML_BLAS_TYPE_CLBLAST ? "1" : " ";

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

// TODO: write as column-wise CSV format.
static void cmd_analyze(struct ggml_mulmat_tune *tune) {
    printf("== gpu compute stage for all NK groups ==\n\n");
    {
        int num_m = tune->num_m;

        printf("#M");
        for (int i = 0; i < num_m; i++) {
            printf(";%3d", tune->groups[0].items[i].M);
        }
        printf("\n");

        // Nothing but for pretty align.
        size_t buf_slot_size = 24;
        char *buf = malloc(buf_slot_size * tune->n_groups);

        size_t max_nxk_len = 0;
        for (int i = 0; i < tune->n_groups; i++) {
            struct ggml_mulmat_tune_nk *group = &tune->groups[i];
            size_t offset = i * buf_slot_size;
            snprintf(&buf[offset], buf_slot_size, "NxK=%dx%d", group->N,
                     group->K);
            size_t len = strlen(&buf[offset]);
            if (len > max_nxk_len) {
                max_nxk_len = len;
            }
        }

        for (int i = 0; i < tune->n_groups; i++) {
            struct ggml_mulmat_tune_nk *group = &tune->groups[i];

            size_t offset = i * buf_slot_size;
            printf("%s", &buf[offset]);
            for (int j = 0; j < (int)(max_nxk_len - strlen(&buf[offset]));
                 j++) {
                printf(" ");
            }

            for (int j = 0; j < num_m; j++) {
                printf(";%8.3f", group->items[j].use_blas_time[1] / 1000.0);
            }
            printf("\n");
        }

        free(buf);
    }

    printf("\n== details for each NK group ==\n\n");
    {
        for (int i = 0; i < tune->n_groups; i++) {
            struct ggml_mulmat_tune_nk *group = &tune->groups[i];
            printf("#M@%dx%d", group->N, group->K);

            for (int j = 0; j < tune->num_m; j++) {
                printf(";%3d", group->items[j].M);
            }
            printf("\n");

            for (int j = 0; j < 3; j++) {
                if (tune->cpu_only_stages[j] > 0) {
                    printf("cpu_only_%d", j);
                    for (int k = 0; k < tune->num_m; k++) {
                        printf(";%8.3f",
                               group->items[k].cpu_only_time[j] / 1000.0);
                    }
                    printf("\n");
                }
            }

            for (int j = 0; j < 3; j++) {
                if (tune->use_blas_stages[j] > 0) {
                    printf("use_blas_%d", j);
                    for (int k = 0; k < tune->num_m; k++) {
                        printf(";%8.3f",
                               group->items[k].use_blas_time[j] / 1000.0);
                    }
                    printf("\n");
                }
            }

            printf("\n");
        }
    }

    printf("== n_threads affects ==\n\n");
    {
        const int nth_list[5] = {1, 2, 4, 6, 8};
        int num_nth = (int)(sizeof(nth_list) / sizeof(int));

        for (int i = 0; i < tune->n_groups; i++) {
            if (i > 0) {
                printf("\n");
            }
            struct ggml_mulmat_tune_nk *group = &tune->groups[i];
            printf("#M@%dx%d", group->N, group->K);

            for (int j = 0; j < tune->num_m; j++) {
                printf(";%3d", group->items[j].M);
            }
            printf("\n");

            for (int k = 0; k < num_nth; k++) {
                int nth = nth_list[k];

                printf("cpu_nth_%d", nth);
                for (int j = 0; j < tune->num_m; j++) {
                    double total = 0.0;
                    for (int stage = 0; stage < 3; stage++) {
                        if (tune->cpu_only_stages[stage] > 0) {
                            int t = group->items[j].cpu_only_time[stage];
                            if (tune->cpu_only_stages[stage] & ((1 << 1))) {
                                t /= nth;
                            }
                            total += t / 1000.0;
                        }
                    }
                    printf(";%8.3f", total);
                }
                printf("\n");

                printf("gpu_nth_%d", nth);
                for (int j = 0; j < tune->num_m; j++) {
                    double total = 0.0;
                    for (int stage = 0; stage < 3; stage++) {
                        if (tune->use_blas_stages[stage] > 0) {
                            int t = group->items[j].use_blas_time[stage];
                            if (tune->use_blas_stages[stage] ==
                                GGML_TASK_FLAG_TN) {
                                t /= nth;
                            }
                            total += t / 1000.0;
                        }
                    }
                    printf(";%8.3f", total);
                }
                printf("\n");
            }
        }
    }
}

static void cmd_test(void) {
    printf("=== test estimate_time\n\n");
    test__estimate_time();

    printf("\n=== test choose_device\n\n");
    test__choose_device();
}

struct test__estimate_time_data {
    int nth;
    int M;
    int expected_cpu_time;
    int expected_gpu_time;
};

static void test__estimate_time(void) {
    struct ggml_mulmat_tune tune = {
        .version = 1,
        .model = "7B",
        .blas_name = "OpenBLAS",
        .n_groups = 1,
        .step_m = 8,
        .num_m = 2,
        .cpu_only_stages = {GGML_TASK_FLAG_T1, GGML_TASK_FLAG_TN, 0},
        .use_blas_stages = {GGML_TASK_FLAG_TN, GGML_TASK_FLAG_T1_WAIT, 0},
    };
    tune.groups = malloc(sizeof(struct ggml_mulmat_tune_nk) * tune.n_groups);
    tune.groups[0] = (struct ggml_mulmat_tune_nk){
        .N = 4096,
        .K = 4096,
    };
    tune.groups[0].items =
        malloc(sizeof(struct ggml_mulmat_tune_m) * tune.num_m);
    tune.groups[0].items[0] = (struct ggml_mulmat_tune_m){
        .M = 8,
        .cpu_only_time = {10, 20, 0},
        .use_blas_time = {30, 40, 0},
    };
    tune.groups[0].items[1] = (struct ggml_mulmat_tune_m){
        .M = 16,
        .cpu_only_time = {50, 60, 0},
        .use_blas_time = {70, 80, 0},
    };

    const int N = tune.groups[0].N;
    const int K = tune.groups[0].K;

    const int nth = 1;

    // Test exact M equals.

    for (int i = 0; i < 2; i++) {
        bool is_cpu = (i == 0);
        for (int j = 0; j < tune.num_m; j++) {
            struct ggml_mulmat_tune_m *item = &tune.groups[0].items[j];
            int M = item->M;

            int t = (i == 0)
                        ? ggml_mulmat_estimate_time(&tune, M, N, K, nth, true)
                        : ggml_mulmat_estimate_time(&tune, M, N, K, nth, false);
            if (is_cpu) {
                BENCH_ASSERT_EQUAL(
                    t, item->cpu_only_time[0] + item->cpu_only_time[1],
                    "#(i: %d, j: %d)", i, j);
            } else {
                BENCH_ASSERT_EQUAL(
                    t, item->use_blas_time[0] + item->use_blas_time[1],
                    "#(i: %d, j: %d)", i, j);
            }
        }
    }

    // Test M out of range
    {
        const int M_arr[2] = {tune.groups[0].items[0].M - 1,
                              tune.groups[0].items[1].M + 1};
        int n = (int)(sizeof(M_arr) / sizeof(int));

        for (int i = 0; i < 2; i++) {
            bool is_cpu = (i == 0);
            for (int j = 0; j < n; j++) {
                int t = ggml_mulmat_estimate_time(&tune, M_arr[j], N, K, nth,
                                                  is_cpu);
                BENCH_ASSERT_EQUAL(t, -1, "#(i: %d, j: %d)", i, j);
            }
        }
    }

    // Test M in range
    {
        const struct test__estimate_time_data test_data[] = {
            {
                .nth = 1,
                .M = 12,
                .expected_cpu_time = 70,
                .expected_gpu_time = 110,
            },
            {
                .nth = 1,
                .M = 14,
                .expected_cpu_time = 90,
                .expected_gpu_time = 130,
            },
        };

        int n =
            (int)(sizeof(test_data) / sizeof(struct test__estimate_time_data));

        for (int i = 0; i < 2; i++) {
            bool is_cpu = (i == 0);
            for (int j = 0; j < n; j++) {
                int t = ggml_mulmat_estimate_time(&tune, test_data[j].M, N, K,
                                                  test_data[j].nth, is_cpu);
                if (is_cpu) {
                    BENCH_ASSERT_EQUAL(t, test_data[j].expected_cpu_time,
                                       "#(i: %d, j: %d)", i, j);
                } else {
                    BENCH_ASSERT_EQUAL(t, test_data[j].expected_gpu_time,
                                       "#(i: %d, j: %d)", i, j);
                }
            }
        }
    }
}

struct test__choose_device_data {
    int nth;
    int M;
    int cpu_only_time;
    int use_blas_time;
};

static void test__choose_device(void) {
    struct ggml_mulmat_tune tune = {
        .version = 1,
        .model = "7B",
        .blas_name = "OPENBLAS",
        .n_groups = 1,
        .step_m = 8,
        .num_m = 2,
        .cpu_only_stages = {GGML_TASK_FLAG_T1, GGML_TASK_FLAG_TN, 0},
        .use_blas_stages = {GGML_TASK_FLAG_TN, GGML_TASK_FLAG_T1, 0},
    };
    tune.groups = malloc(sizeof(struct ggml_mulmat_tune_nk) * tune.n_groups);
    tune.groups[0] = (struct ggml_mulmat_tune_nk){
        .N = 4096,
        .K = 4096,
    };
    tune.groups[0].items =
        malloc(sizeof(struct ggml_mulmat_tune_m) * tune.num_m);
    tune.groups[0].items[0] = (struct ggml_mulmat_tune_m){
        .M = 8,
        .cpu_only_time = {2, 4, 0},
        .use_blas_time = {4, 4, 0},
    };
    tune.groups[0].items[1] = (struct ggml_mulmat_tune_m){
        .M = 16,
        .cpu_only_time = {4, 8, 0},
        .use_blas_time = {4, 4, 0},
    };

    const int N = tune.groups[0].N;
    const int K = tune.groups[0].K;

    // When M out of range.
    {
        const int M_arr[2] = {tune.groups[0].items[0].M - 1,
                              tune.groups[0].items[1].M + 1};
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
        const struct test__choose_device_data test_data[] = {
            {
                .nth = 1,
                .M = 8,
                .cpu_only_time = 6,
                .use_blas_time = 8,
            },
            {
                .nth = 1,
                .M = 12,
                .cpu_only_time = 9,
                .use_blas_time = 8,
            },
            {
                .nth = 1,
                .M = 16,
                .cpu_only_time = 12,
                .use_blas_time = 8,
            },
            {
                .nth = 2,
                .M = 8,
                .cpu_only_time = 4,
                .use_blas_time = 6,
            },
            {
                .nth = 2,
                .M = 12,
                .cpu_only_time = 6,
                .use_blas_time = 6,
            },
            {
                .nth = 2,
                .M = 16,
                .cpu_only_time = 8,
                .use_blas_time = 6,
            }};

        int n =
            (int)(sizeof(test_data) / sizeof(struct test__choose_device_data));

        for (int i = 0; i < n; i++) {
            const struct test__choose_device_data *e = &test_data[i];
            struct ggml_mulmat_tune_time_stats time_stats;
            int rc = ggml_mulmat_tune_time_stats(&tune, e->M, N, K, e->nth,
                                                 &time_stats);
            BENCH_ASSERT_EQUAL(rc, 0, "#(i: %d)", i);
            BENCH_ASSERT_EQUAL(time_stats.cpu_only_total, e->cpu_only_time,
                               "#(i: %d)", i);
            BENCH_ASSERT_EQUAL(time_stats.use_blas_total, e->use_blas_time,
                               "#(i: %d)", i);
        }
    }
}
