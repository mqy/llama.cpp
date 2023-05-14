#include "examples/mulmat-device/mulmat-device.h"
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

//
// TODO: support all quantization types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0.
//       add quant type to bench output file.
//

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

static int bench_time_min(int *a, int len);
static void print_blas_build_tips(void);
static void progress(int i, int max);
static void print_envs_for_build(enum ggml_blas_type, char *buf, int buf_len);
static bool prompt_yes_no(const char *prompt);

static void cmd_bench(struct ggml_mulmat_bench *b);
static void cmd_analyze(struct ggml_mulmat_bench *b);
static void cmd_test(void);

static void test__estimate_time(void);
static void test__choose_device(void);

static void usage(char *prog) {
    const char *usage_lines[] = {
        "usage:\n",
        "* %s bench   <model> [data-file] [-y]\n",
        "  model: 7B or 13B.\n",
        "  data-file: the data file to write to, write to stdout if absent.\n",
        "  -y always answer \"yes\" to all prompts.\n",
        "* %s analyze <data-file>\n",
        "* %s test\n",
        "* %s help\n",
    };

    int len = (int)(sizeof(usage_lines) / sizeof(char *));
    for (int i = 0; i < len; i++) {
        const char *line = usage_lines[i];
        if (line[0] == '*') {
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

    if (argc < 2) {
        fprintf(stderr, "error: need sub command");
        usage(argv[0]);
        exit(1);
    }

    char *cmd = argv[1];

    if (strcmp(cmd, "bench") == 0) {
        if (argc < 3) {
            fprintf(stderr, "[%s]: too few args", cmd);
            usage(argv[0]);
            exit(1);
        }

        struct ggml_mulmat_bench bench = {
            .version = 1,
            .n_groups = 0,
            .m_step = 8,
            .num_m = 16,
            .cpu_stages = {GGML_TASK_FLAG_1_THREAD, GGML_TASK_FLAG_N_THREADS,
                           0},
#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)
            .gpu_stages = {GGML_TASK_FLAG_N_THREADS,
                           GGML_TASK_FLAG_1_THREAD__WAIT, 0},
#elif defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
            .gpu_stages = {0, GGML_TASK_FLAG_1_THREAD, 0},
#endif
            .groups = NULL,
        };

        if (false) {
            // for larger M range.
            bench.m_step = 16;
            bench.num_m = 32;
        }

        const char *model = argv[2];

        const char *data_file = NULL;
        FILE *fp = NULL;

        bool always_yes = false;

        if (argc == 4 || argc == 5) {
            for (int i = 3; i < argc; i++) {
                if (strcmp(argv[i], "-y") == 0) {
                    always_yes = true;
                } else {
                    data_file = argv[i];
                }
            }
        }

        if (data_file != NULL && !always_yes) {
            struct stat st;
            int rc = stat(data_file, &st);
            UNUSED(st);
            if (rc == 0) { // prompt
                size_t len = strlen(data_file) + 50;
                char *prompt = malloc(len);
                GGML_ASSERT(prompt);
                snprintf(prompt, len,
                         "[%s]: data file '%s' exists, override? (Y|n)", cmd,
                         data_file);

                if (!prompt_yes_no(prompt)) {
                    printf("Aborted.\n");
                    exit(2);
                }
                free(prompt);
            }

            fp = fopen(data_file, "w");
            GGML_ASSERT(fp);
        }

        if (strcmp(model, "7B") == 0) {
            bench.n_groups = 4;
            bench.groups =
                malloc(bench.n_groups * sizeof(struct ggml_mulmat_bench_nk));
            GGML_ASSERT(bench.groups);
            bench.groups[0] = (struct ggml_mulmat_bench_nk){
                .N = 4096, .K = 4096, .items = NULL};
            bench.groups[1] = (struct ggml_mulmat_bench_nk){
                .N = 4096, .K = 11008, .items = NULL};
            bench.groups[2] = (struct ggml_mulmat_bench_nk){
                .N = 11008, .K = 4096, .items = NULL};
            bench.groups[3] = (struct ggml_mulmat_bench_nk){
                .N = 32000, .K = 4096, .items = NULL};
        } else if (strcmp(model, "13B") == 0) {
            bench.n_groups = 4;
            bench.groups =
                malloc(bench.n_groups * sizeof(struct ggml_mulmat_bench_nk));
            GGML_ASSERT(bench.groups);
            bench.groups[0] = (struct ggml_mulmat_bench_nk){
                .N = 5120, .K = 5120, .items = NULL};
            bench.groups[1] = (struct ggml_mulmat_bench_nk){
                .N = 5120, .K = 13824, .items = NULL};
            bench.groups[2] = (struct ggml_mulmat_bench_nk){
                .N = 13824, .K = 5120, .items = NULL};
            bench.groups[3] = (struct ggml_mulmat_bench_nk){
                .N = 32000, .K = 5120, .items = NULL};
        } else {
            // TODO: support 30B and 65B.
            fprintf(stderr, "[%s]: error: unsupported model: %s", cmd, model);
            usage(argv[0]);
            exit(1);
        }

        // bench.model
        {
            size_t n = sizeof(bench.model);
            GGML_ASSERT(n > strlen(model));
            strncpy(bench.model, model, n);
        }

        // bench.blas_name
        {
            const char *blas_name = ggml_get_blas_name();
            GGML_ASSERT(blas_name);

            size_t n = sizeof(bench.blas_name);
            GGML_ASSERT(n > strlen(blas_name));
            strncpy(bench.blas_name, blas_name, n);
        }

#if defined GGML_USE_CLBLAST
        ggml_cl_init();
#endif

        if (always_yes) {
            printf("Bench for %s ...\n", bench.blas_name);
        } else {
            char buf[64];
            snprintf(buf, 64, "Will bench for %s, are you sure? (Y|n)",
                     bench.blas_name);

            if (!prompt_yes_no(buf)) {
                printf("Aborted.\n");
                exit(2);
            }
        }

        cmd_bench(&bench);

        ggml_mulmat_write_bench_data(&bench, fp == NULL ? stdout : fp);
        if (fp != NULL) {
            fclose(fp);
        }

        if (data_file != NULL) {
            printf("[%s]: result was written to %s\n", cmd, data_file);
        }
    } else if (strcmp(cmd, "analyze") == 0) {
        if (argc < 3) {
            fprintf(stderr, "[%s]: error: too few args", cmd);
            usage(argv[0]);
            exit(1);
        }

        struct ggml_mulmat_bench bench;

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
        int rc = ggml_mulmat_read_bench_data(&bench, fp);
        GGML_ASSERT(rc == 0);
        fclose(fp);

        cmd_analyze(&bench);
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

void cmd_bench(struct ggml_mulmat_bench *bench) {
    size_t wsize = 0;
    void *q4_0_buf = NULL;
    void *wdata = NULL;

    // alloc q4_0_buf and wdata with max size.
    {
        int max_NxK = 0;
        for (int i = 0; i < bench->n_groups; i++) {
            int sz = bench->groups[i].N * bench->groups[i].K;
            if (sz > max_NxK) {
                max_NxK = sz;
            }
        }

        size_t q4_0_buf_size = 2 * max_NxK * sizeof(int64_t);
        q4_0_buf = malloc(q4_0_buf_size);
        if (!q4_0_buf) {
            fprintf(stderr,
                    "failed to allocate memory for q4_0_buf, size: %zu MiB\n",
                    q4_0_buf_size / 1024 / 1024);
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

    for (int i = 0; i < bench->n_groups; i++) {
        struct ggml_mulmat_bench_nk *group = &bench->groups[i];
        int M;
        int N = group->N;
        int K = group->K;

        {
            size_t sz = sizeof(struct ggml_mulmat_bench_m) * bench->num_m;
            group->items = malloc(sz);
            GGML_ASSERT(group->items);
            memset(group->items, 0, sz);
        }

        char progress_line[20];

        for (int im = 0; im < bench->num_m; im++) {
            M = bench->m_step * (im + 1);

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

                src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, (int64_t)K,
                                          (int64_t)N);
                ggml_quantize_q4_0((const float *)src0_f32->data, src0->data,
                                   N * K, K, (int64_t *)q4_0_buf);

                // src1: M x K
                src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)K,
                                          (int64_t)M);
                ggml_set_f32(src1, 0.5f);

                // dst: M x N
                dst = ggml_mul_mat(ctx, src0, src1);
            }

            struct ggml_mulmat_bench_m *bench_item = &group->items[im];
            bench_item->M = M;

            ggml_task_flag_set_blas(&dst->task_flag, 0);
            for (int stage = 0; stage < 3; stage++) {
                if (bench->cpu_stages[stage] > 0) {
                    // without this, the first run may be significant slow.
                    memset(wdata, 0, wsize);

                    for (int nb = 0; nb < NUM_BENCH; nb++) {
                        int t0 = (int)ggml_time_us();
                        ggml_internal_compute_forward_mul_mat_q_f32_for_bench(
                            stage, wsize, wdata, src0, src1, dst);
                        bench_item->cpu_records[stage][nb] =
                            (int)ggml_time_us() - t0;
                        progress(nb, NUM_BENCH);
                    }
                    line_len++;
                }
            }

            ggml_task_flag_set_blas(&dst->task_flag, 1);
            for (int stage = 0; stage < 3; stage++) {
                if (bench->gpu_stages[stage] > 0) {
                    for (int nb = 0; nb < NUM_BENCH; nb++) {
                        int t0 = (int)ggml_time_us();
                        ggml_internal_compute_forward_mul_mat_q_f32_for_bench(
                            stage, wsize, wdata, src0, src1, dst);
                        bench_item->gpu_records[stage][nb] =
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
    free(q4_0_buf);

    // collect stat records.
    for (int i = 0; i < bench->n_groups; i++) {
        for (int j = 0; j < bench->num_m; j++) {
            struct ggml_mulmat_bench_m *item = &bench->groups[i].items[j];
            for (int stage = 0; stage < 3; stage++) {
                if (bench->cpu_stages[stage] > 0) {
                    item->cpu_time[stage] =
                        bench_time_min(item->cpu_records[stage], NUM_BENCH);
                }
                if (bench->gpu_stages[stage] > 0) {
                    item->gpu_time[stage] =
                        bench_time_min(item->gpu_records[stage], NUM_BENCH);
                }
            }
        }
    }
}

static void print_blas_build_tips(void) {
    const char *make_target = "mulmat-device-bench";

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

static int bench_time_min(int *a, int len) {
    // bubble sort `a`.
    for (int i = 0; i < len - 1; i++) {
        for (int j = i + 1; j < len; j++) {
            if (a[j] < a[i]) {
                int temp = a[j];
                a[j] = a[i];
                a[i] = temp;
            }
        }
    }

    int min = 0;
    for (int i = 0; i < len; i++) {
        if (a[i] < min) {
            min = a[i];
        }
    }
    return min;
}

// TODO: write as column-wise CSV format.
static void cmd_analyze(struct ggml_mulmat_bench *bench) {
    printf("== gpu compute stage for all NK groups ==\n\n");
    {
        int num_m = bench->num_m;

        printf("#M");
        for (int i = 0; i < num_m; i++) {
            printf(";%3d", bench->groups[0].items[i].M);
        }
        printf("\n");

        // Nothing but for pretty align.
        size_t buf_slot_size = 24;
        char *buf = malloc(buf_slot_size * bench->n_groups);

        size_t max_nxk_len = 0;
        for (int i = 0; i < bench->n_groups; i++) {
            struct ggml_mulmat_bench_nk *group = &bench->groups[i];
            size_t offset = i * buf_slot_size;
            snprintf(&buf[offset], buf_slot_size, "NxK=%dx%d", group->N,
                     group->K);
            size_t len = strlen(&buf[offset]);
            if (len > max_nxk_len) {
                max_nxk_len = len;
            }
        }

        for (int i = 0; i < bench->n_groups; i++) {
            struct ggml_mulmat_bench_nk *group = &bench->groups[i];

            size_t offset = i * buf_slot_size;
            printf("%s", &buf[offset]);
            for (int j = 0; j < (int)(max_nxk_len - strlen(&buf[offset]));
                 j++) {
                printf(" ");
            }

            for (int j = 0; j < num_m; j++) {
                printf(";%8.3f", group->items[j].gpu_time[1] / 1000.0);
            }
            printf("\n");
        }

        free(buf);
    }

    printf("\n== details for each NK group ==\n\n");
    {
        for (int i = 0; i < bench->n_groups; i++) {
            struct ggml_mulmat_bench_nk *group = &bench->groups[i];
            printf("#M@%dx%d", group->N, group->K);

            for (int j = 0; j < bench->num_m; j++) {
                printf(";%3d", group->items[j].M);
            }
            printf("\n");

            for (int j = 0; j < 3; j++) {
                if (bench->cpu_stages[j] > 0) {
                    printf("cpu_%d", j);
                    for (int k = 0; k < bench->num_m; k++) {
                        printf(";%8.3f", group->items[k].cpu_time[j] / 1000.0);
                    }
                    printf("\n");
                }
            }

            for (int j = 0; j < 3; j++) {
                if (bench->gpu_stages[j] > 0) {
                    printf("gpu_%d", j);
                    for (int k = 0; k < bench->num_m; k++) {
                        printf(";%8.3f", group->items[k].gpu_time[j] / 1000.0);
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

        for (int i = 0; i < bench->n_groups; i++) {
            if (i > 0) {
                printf("\n");
            }
            struct ggml_mulmat_bench_nk *group = &bench->groups[i];
            printf("#M@%dx%d", group->N, group->K);

            for (int j = 0; j < bench->num_m; j++) {
                printf(";%3d", group->items[j].M);
            }
            printf("\n");

            for (int k = 0; k < num_nth; k++) {
                int nth = nth_list[k];

                printf("cpu_nth_%d", nth);
                for (int j = 0; j < bench->num_m; j++) {
                    double total = 0.0;
                    for (int stage = 0; stage < 3; stage++) {
                        if (bench->cpu_stages[stage] > 0) {
                            int t = group->items[j].cpu_time[stage];
                            if (bench->cpu_stages[stage] & ((1 << 1))) {
                                t /= nth;
                            }
                            total += t / 1000.0;
                        }
                    }
                    printf(";%8.3f", total);
                }
                printf("\n");

                printf("gpu_nth_%d", nth);
                for (int j = 0; j < bench->num_m; j++) {
                    double total = 0.0;
                    for (int stage = 0; stage < 3; stage++) {
                        if (bench->gpu_stages[stage] > 0) {
                            int t = group->items[j].gpu_time[stage];
                            if (bench->gpu_stages[stage] ==
                                GGML_TASK_FLAG_N_THREADS) {
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
    struct ggml_mulmat_bench bench = {
        .version = 1,
        .model = "7B",
        .blas_name = "OpenBLAS",
        .n_groups = 1,
        .m_step = 8,
        .num_m = 2,
        .cpu_stages = {GGML_TASK_FLAG_1_THREAD, GGML_TASK_FLAG_N_THREADS, 0},
        .gpu_stages = {GGML_TASK_FLAG_N_THREADS, GGML_TASK_FLAG_1_THREAD__WAIT,
                       0},
    };
    bench.groups = malloc(sizeof(struct ggml_mulmat_bench_nk) * bench.n_groups);
    bench.groups[0] = (struct ggml_mulmat_bench_nk){
        .N = 4096,
        .K = 4096,
    };
    bench.groups[0].items =
        malloc(sizeof(struct ggml_mulmat_bench_m) * bench.num_m);
    bench.groups[0].items[0] = (struct ggml_mulmat_bench_m){
        .M = 8,
        .cpu_time = {10, 20, 0},
        .gpu_time = {30, 40, 0},
    };
    bench.groups[0].items[1] = (struct ggml_mulmat_bench_m){
        .M = 16,
        .cpu_time = {50, 60, 0},
        .gpu_time = {70, 80, 0},
    };

    const int N = bench.groups[0].N;
    const int K = bench.groups[0].K;

    const int nth = 1;

    // Test exact M equals.

    for (int i = 0; i < 2; i++) {
        bool is_cpu = (i == 0);
        for (int j = 0; j < bench.num_m; j++) {
            struct ggml_mulmat_bench_m *item = &bench.groups[0].items[j];
            int M = item->M;

            int t =
                (i == 0)
                    ? ggml_mulmat_estimate_time(&bench, M, N, K, nth, true)
                    : ggml_mulmat_estimate_time(&bench, M, N, K, nth, false);
            if (is_cpu) {
                BENCH_ASSERT_EQUAL(t, item->cpu_time[0] + item->cpu_time[1],
                                   "#(i: %d, j: %d)", i, j);
            } else {
                BENCH_ASSERT_EQUAL(t, item->gpu_time[0] + item->gpu_time[1],
                                   "#(i: %d, j: %d)", i, j);
            }
        }
    }

    // Test M out of range
    {
        const int M_arr[2] = {bench.groups[0].items[0].M - 1,
                              bench.groups[0].items[1].M + 1};
        int n = (int)(sizeof(M_arr) / sizeof(int));

        for (int i = 0; i < 2; i++) {
            bool is_cpu = (i == 0);
            for (int j = 0; j < n; j++) {
                int t = ggml_mulmat_estimate_time(&bench, M_arr[j], N, K, nth,
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
                int t = ggml_mulmat_estimate_time(&bench, test_data[j].M, N, K,
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
    bool use_blas;
};

static void test__choose_device(void) {
    struct ggml_mulmat_bench bench = {
        .version = 1,
        .model = "7B",
        .blas_name = "OPENBLAS",
        .n_groups = 1,
        .m_step = 8,
        .num_m = 2,
        .cpu_stages = {GGML_TASK_FLAG_1_THREAD, GGML_TASK_FLAG_N_THREADS, 0},
        .gpu_stages = {GGML_TASK_FLAG_N_THREADS, GGML_TASK_FLAG_1_THREAD, 0},
    };
    bench.groups = malloc(sizeof(struct ggml_mulmat_bench_nk) * bench.n_groups);
    bench.groups[0] = (struct ggml_mulmat_bench_nk){
        .N = 4096,
        .K = 4096,
    };
    bench.groups[0].items =
        malloc(sizeof(struct ggml_mulmat_bench_m) * bench.num_m);
    bench.groups[0].items[0] = (struct ggml_mulmat_bench_m){
        .M = 8,
        .cpu_time = {10, 100, 0},
        .gpu_time = {100, 200, 0},
    };
    bench.groups[0].items[1] = (struct ggml_mulmat_bench_m){
        .M = 16,
        .cpu_time = {20, 300, 0},
        .gpu_time = {100, 100, 0},
    };

    const int N = bench.groups[0].N;
    const int K = bench.groups[0].K;

    // When M out of range.
    {
        const int M_arr[2] = {bench.groups[0].items[0].M - 1,
                              bench.groups[0].items[1].M + 1};
        int n = (int)(sizeof(M_arr) / sizeof(int));

        for (int i = 1; i <= 8; i++) {
            int nth = i;
            for (int j = 0; j < n; j++) {
                bool use_blas =
                    ggml_mulmat_bench_use_blas(&bench, M_arr[j], N, K, nth);
                if (j == 0) {
                    BENCH_ASSERT_EQUAL(use_blas, false, "#(i: %d, i: %d)", i,
                                       j);
                } else {
                    BENCH_ASSERT_EQUAL(use_blas, true, "#(i: %d, i: %d)", i, j);
                }
            }
        }
    }

    // When M in range.
    {
        const struct test__choose_device_data test_data[] = {
            {
                .nth = 1,
                .M = 8,
                .use_blas = false,
            },
            {
                .nth = 1,
                .M = 12,
                .use_blas = false,
            },
            {
                .nth = 1,
                .M = 16,
                .use_blas = true,
            },
            {
                .nth = 2,
                .M = 8,
                .use_blas = false,
            },
            {
                .nth = 2,
                .M = 12,
                .use_blas = false,
            },
            {
                .nth = 2,
                .M = 16,
                .use_blas = true,
            },
            {
                .nth = 4,
                .M = 8,
                .use_blas = false,
            },
            {
                .nth = 4,
                .M = 12,
                .use_blas = false,
            },
            {
                .nth = 4,
                .M = 16,
                .use_blas = false,
            },
        };

        int n =
            (int)(sizeof(test_data) / sizeof(struct test__choose_device_data));

        for (int i = 0; i < n; i++) {
            const struct test__choose_device_data *e = &test_data[i];
            bool us_blas =
                ggml_mulmat_bench_use_blas(&bench, e->M, N, K, e->nth);
            BENCH_ASSERT_EQUAL(us_blas, e->use_blas, "#(i: %d)", i);
        }
    }
}
