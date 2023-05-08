#include "examples/benchmark/q40-mulmat-device-bench/q40-mulmat-device.h"
#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <inttypes.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define BENCH_ASSERT(x)                                                        \
    do {                                                                       \
        if (!(x)) {                                                            \
            fprintf(stderr, "BENCH_ASSERT: %s:%d: %s\n", __FILE__, __LINE__,   \
                    #x);                                                       \
            abort();                                                           \
        }                                                                      \
    } while (0)

#define BENCH_ASSERT_INT_EQUAL(actual, expect, fmt, ...)                       \
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

static int64_t time_us(void);
static int bench_time_avg(int *a, int len);
static void print_build_blas_tip(void);
static void progress(int i, int max);
static void envs_for_gpu_feature(int feature, char *buf, int buf_len);
static bool prompt_yes_no(const char *prompt);

static void cmd_bench(struct ggml_mulmat_bench_data *bd);
static void cmd_analyze(struct ggml_mulmat_bench_data *bd);
static void cmd_test(void);

static void test__estimate_time(void);
static void test__choose_device(void);

static void usage(char *prog) {
    const char *usage_lines[7] = {
        "usage:\n",
        "* %s bench   <model> [data-file [-y]]\n",
        "  model: 7B or 13B.\n",
        "  data-file: the file to write bench result to.\n",
        "  -y always answer \"yes\" to overriding existing data file.\n",
        "* %s analyze <data-file>\n",
        "* %s test\n",
    };

    for (int i = 0; i < 7; i++) {
        const char *line = usage_lines[i];
        if (line[0] == '*') {
            fprintf(stderr, line, prog);
        } else {
            fprintf(stderr, "%s", line);
        }
    }
}

static struct model_nk_shape {
    int N;
    int K;
};

// main
int main(int argc, char **argv) {
    printf("\n");

    if (!ggml_cpu_has_blas()) {
        print_build_blas_tip();
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

        struct ggml_mulmat_bench_data bd = {
            .version = 1,
            .m_step = 8,
            .num_m = 11,
            .cpu_stages = {1, (2 | 1), 0},
#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)
            .gpu_stages = {(2 | 1), 1, 0},
#elif defined(GGML_USE_CUBLAS)
            .gpu_stages = {0, 1, 0},
#endif
            .shapes = NULL,
        };

        if (false) {
            // for M from 16 through 512.
            bd.m_step = 32;
            bd.num_m = 16;
        }

        const char *model = argv[2];

        const char *data_file = NULL;
        FILE *fp = NULL;

        if (argc == 4 || argc == 5) {
            data_file = argv[3];
            bool overriding_check = true;

            if (argc == 5) {
                if (strcmp(argv[4], "-y") != 0) {
                    fprintf(
                        stderr,
                        "[%s]: error: the last arg is expected to be \"-y\".\n",
                        cmd);
                    usage(argv[0]);
                    exit(1);
                }
                overriding_check = false;
            }

            if (overriding_check) {
                struct stat st;
                int rc = stat(data_file, &st);
                UNUSED(st);
                if (rc == 0) { // prompt
                    size_t len = strlen(data_file) + 50;
                    char *prompt = malloc(len);
                    BENCH_ASSERT(prompt);
                    snprintf(prompt, len,
                             "[%s]: data file '%s' exists, override? (Y|n)",
                             cmd, data_file);

                    if (!prompt_yes_no(prompt)) {
                        printf("Aborted.\n");
                        exit(2);
                    }
                    free(prompt);
                }
            }

            fp = fopen(data_file, "w");
            BENCH_ASSERT(fp);
        }

        if (strcmp(model, "7B") == 0) {
            bd.n_shapes = 3,
            bd.shapes = malloc(bd.n_shapes * sizeof(struct model_nk_shape));
            BENCH_ASSERT(bd.shapes);
            bd.shapes = (struct ggml_mulmat_bench_data_shape[]){
                {.N = 4096, .K = 4096},
                {.N = 4096, .K = 11008},
                {.N = 11008, .K = 4096},
            };
        } else if (strcmp(model, "13B") == 0) {
            bd.n_shapes = 3,
            bd.shapes = malloc(bd.n_shapes * sizeof(struct model_nk_shape));
            BENCH_ASSERT(bd.shapes);
            bd.shapes = (struct ggml_mulmat_bench_data_shape[]){
                {.N = 5120, .K = 5120},
                {.N = 5120, .K = 13824},
                {.N = 13824, .K = 5120},
            };
        } else {
            // TODO: support 30B and 65B.
            fprintf(stderr, "[%s]: error: unsupported model: %s", cmd, model);
            usage(argv[0]);
            exit(1);
        }

        size_t n = sizeof(bd.model);
        memset(bd.model, 0, n);
        strncpy(bd.model, model, n);

        n = sizeof(bd.gpu_impl);
        memset(bd.gpu_impl, 0, n);

#if defined(GGML_USE_ACCELERATE)
        strncpy(bd.gpu_impl, "ACCELERATE", n);
#elif defined(GGML_USE_OPENBLAS)
        strncpy(bd.gpu_impl, "OPENBLAS", n);
#elif defined(GGML_USE_CUBLAS)
        strncpy(bd.gpu_impl, "CUBLAS", n);
#endif

        cmd_bench(&bd);

        ggml_mulmat_write_bench_data(&bd, fp == NULL ? stdout : fp);
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

        struct ggml_mulmat_bench_data bd;

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
        BENCH_ASSERT(fp);
        int rc = ggml_mulmat_read_bench_data(&bd, fp);
        BENCH_ASSERT(rc > 0);
        fclose(fp);

        cmd_analyze(&bd);
    } else if (strcmp(cmd, "test") == 0) {
        if (argc != 2) {
            fprintf(stderr, "[%s]: error: invalid args\n", cmd);
            usage(argv[0]);
            exit(1);
        }
        cmd_test();
    } else {
        fprintf(stderr, "error: unknown command: %s.\n", cmd);
        usage(argv[0]);
        exit(1);
    }

    printf("\n[%s]: done!\n", cmd);

    return 0;
}

void cmd_bench(struct ggml_mulmat_bench_data *bd) {
    size_t wdata_size = 0;
    void *q4_0_buf = NULL;
    void *wdata = NULL;

    // alloc q4_0_buf and wdata with max size.
    {
        size_t max_NxK = 0;
        for (int i = 0; i < bd->n_shapes; i++) {
            size_t sz = bd->shapes[i].N * bd->shapes[i].K;
            if (sz > max_NxK) {
                max_NxK = sz;
            }
        }

        q4_0_buf = malloc(max_NxK * sizeof(int64_t));
        if (!q4_0_buf) {
            fprintf(stderr, "failed to allocate memory\n");
            exit(1);
        }
        wdata_size = max_NxK * sizeof(float);
        wdata = malloc(wdata_size);
        if (!wdata) {
            fprintf(stderr, "failed to allocate memory\n");
            exit(1);
        }
    }

    for (int i = 0; i < bd->n_shapes; i++) {
        int M;
        int N = bd->shapes[i].N;
        int K = bd->shapes[i].K;

        struct ggml_mulmat_bench_data_shape *bench_shape = &bd->shapes[i];
        {
            size_t sz = sizeof(struct ggml_mulmat_bench_data_item) * bd->num_m;
            bench_shape->items = malloc(sz);
            BENCH_ASSERT(bench_shape->items);
            memset(bench_shape->items, 0, sz);
        }

        for (int im = 0; im < bd->num_m; im++) {
            M = bd->m_step * (im + 1);
            int line_len = 16;
            printf("%d %d %d ", N, K, M);
            fflush(stdout);

            struct ggml_context *ctx = NULL;
            {
                // The ctx_size is over estimated.
                size_t ctx_size = K * N * ggml_type_sizef(GGML_TYPE_F32) +
                                  K * sizeof(float) + 1024 * 1024 * 300;

                struct ggml_init_params init_params = {
                    .mem_size = ctx_size,
                    .mem_buffer = NULL,
                    .no_alloc = 0,
                };

                ctx = ggml_init(init_params);
                BENCH_ASSERT(ctx);
            }

            struct ggml_tensor *src0 = NULL;
            struct ggml_tensor *src1 = NULL;
            struct ggml_tensor *dst = NULL;

            {
                // src0: K x N
                struct ggml_tensor *src0_f32 =
                    ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
                ggml_set_f32(src0_f32, 0.1f);

                src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
                ggml_quantize_q4_0((const float *)src0_f32->data, src0->data, N,
                                   K, (int64_t *)q4_0_buf);

                // src1: M x K
                src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
                ggml_set_f32(src1, 0.5f);

                // dst: M x N
                dst = ggml_mul_mat(ctx, src0, src1);
            }

            struct ggml_mulmat_bench_data_item *bench_item = &bench_shape->items[im];
            bench_item->M = M;

            struct ggml_compute_params compute_params = {
                .n_threads = 1,
                .ith = 0,
                .nth = 1,
                .wsize = wdata_size,
                .wdata = wdata,
            };

            dst->sched.device = GGML_DEVICE_CPU;
            for (int stage = GGML_TASK_INIT; stage <= GGML_TASK_FINALIZE;
                 stage++) {
                if (bd->cpu_stages[stage] & 1) {
                    // without this, the first run may be significant slow.
                    memset(wdata, 0, wdata_size);

                    compute_params.type = stage;
                    for (int nb = 0; nb < NUM_BENCH; nb++) {
                        int t0 = (int)time_us();
                        ggml_compute_forward_mul_mat_q_f32(&compute_params,
                                                           src0, src1, dst);
                        bench_item->cpu_records[stage][nb] =
                            (int)time_us() - t0;
                        progress(nb, NUM_BENCH);
                    }
                    line_len++;
                }
            }

            dst->sched.device = GGML_DEVICE_GPU;
            for (int stage = GGML_TASK_INIT; stage <= GGML_TASK_FINALIZE;
                 stage++) {
                if (bd->gpu_stages[stage] & 1) {
                    compute_params.type = stage;
                    for (int nb = 0; nb < NUM_BENCH; nb++) {
                        int t0 = (int)time_us();
                        ggml_compute_forward_mul_mat_q_f32(&compute_params,
                                                           src0, src1, dst);
                        bench_item->gpu_records[stage][nb] =
                            (int)time_us() - t0;
                        progress(nb, NUM_BENCH);
                    }
                    line_len++;
                }
            }

            for (int j = 0; j < line_len; j++) {
                printf("\b \b");
            }
            fflush(stdout);

            ggml_free(ctx);
        }
    }

    free(wdata);
    free(q4_0_buf);

    // stats -> avg.
    for (int i = 0; i < bd->n_shapes; i++) {
        for (int j = 0; j < bd->num_m; j++) {
            struct ggml_mulmat_bench_data_item *item = &bd->shapes[i].items[j];
            for (int stage = GGML_TASK_INIT; stage <= GGML_TASK_FINALIZE;
                 stage++) {
                if (bd->cpu_stages[stage] > 0) {
                    item->cpu_time[stage] =
                        bench_time_avg(item->cpu_records[stage], NUM_BENCH);
                }
                if (bd->gpu_stages[stage] > 0) {
                    item->gpu_time[stage] =
                        bench_time_avg(item->gpu_records[stage], NUM_BENCH);
                }
            }
        }
    }
}

static int64_t time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}

static void print_build_blas_tip(void) {
    const char *make_target = "q40-mulmat-device-bench";

    fprintf(stderr,
            "error: this program was not built with any GPU feature. tips:\n");

    char buf[100];
    envs_for_gpu_feature(1, buf, 100);
    fprintf(stderr, "* to build with accelerate: make clean; %s make %s\n", buf,
            make_target);
    envs_for_gpu_feature(2, buf, 100);
    fprintf(stderr, "* to build with openBLAS:   make clean; %s make %s\n", buf,
            make_target);
    envs_for_gpu_feature(3, buf, 100);
    fprintf(stderr, "* to build with cuBLAS:     make clean; %s make %s\n", buf,
            make_target);
}

// feature: 1: apple accelerate, 2: openBLAS, 3: cuBLAS
static void envs_for_gpu_feature(int feature, char *buf, int buf_len) {
    memset(buf, 0, buf_len);
    const char *LLAMA_NO_ACCELERATE = feature == 1 ? " " : "1";
    const char *LLAMA_OPENBLAS = feature == 2 ? "1" : " ";
    const char *LLAMA_CUBLAS = feature == 3 ? "1" : " ";
    snprintf(buf, buf_len,
             "LLAMA_NO_ACCELERATE=%s LLAMA_OPENBLAS=%s LLAMA_CUBLAS=%s",
             LLAMA_NO_ACCELERATE, LLAMA_OPENBLAS, LLAMA_CUBLAS);
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

static int bench_time_avg(int *a, int len) {
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

    int total = 0;
    // throw away min and max
    for (int i = 1; i < len - 1; i++) {
        total += a[i];
    }
    return total / (len - 2);
}

// TODO: write as column wise CSV format.
static void cmd_analyze(struct ggml_mulmat_bench_data *bd) {
    printf("\n== gpu compute stage for all shapes ==\n\n");
    {
        int num_m = bd->num_m;

        printf("#M");
        for (int i = 0; i < num_m; i++) {
            printf(";%3d", bd->shapes[0].items[i].M);
        }
        printf("\n");

        // Nothing but for pretty align.
        size_t buf_slot_size = 24;
        char *buf = malloc(buf_slot_size * bd->n_shapes);

        size_t max_nxk_len = 0;
        for (int i = 0; i < bd->n_shapes; i++) {
            struct ggml_mulmat_bench_data_shape *s = &bd->shapes[i];
            size_t offset = i * buf_slot_size;
            snprintf(&buf[offset], buf_slot_size, "NxK=%dx%d", s->N, s->K);
            size_t len = strlen(&buf[offset]);
            if (len > max_nxk_len) {
                max_nxk_len = len;
            }
        }

        for (int i = 0; i < bd->n_shapes; i++) {
            struct ggml_mulmat_bench_data_shape *s = &bd->shapes[i];

            size_t offset = i * buf_slot_size;
            printf("%s", &buf[offset]);
            for (int j = 0; j < (int)(max_nxk_len - strlen(&buf[offset]));
                 j++) {
                printf(" ");
            }

            for (int j = 0; j < num_m; j++) {
                printf(";%8.3f", s->items[j].gpu_time[1] / 1000.0);
            }
            printf("\n");
        }

        free(buf);
    }

    printf("\n== details for each shape ==\n\n");
    {
        for (int i = 0; i < bd->n_shapes; i++) {
            struct ggml_mulmat_bench_data_shape *s = &bd->shapes[i];
            printf("#M@%dx%d", s->N, s->K);

            for (int j = 0; j < bd->num_m; j++) {
                printf(";%3d", s->items[j].M);
            }
            printf("\n");

            for (int j = GGML_TASK_INIT; j <= GGML_TASK_FINALIZE; j++) {
                if (bd->cpu_stages[j] & 1) {
                    printf("cpu_%d", j);
                    for (int k = 0; k < bd->num_m; k++) {
                        printf(";%8.3f", s->items[k].cpu_time[j] / 1000.0);
                    }
                    printf("\n");
                }
            }

            for (int j = GGML_TASK_INIT; j <= GGML_TASK_FINALIZE; j++) {
                if (bd->gpu_stages[j] & 1) {
                    printf("gpu_%d", j);
                    for (int k = 0; k < bd->num_m; k++) {
                        printf(";%8.3f", s->items[k].gpu_time[j] / 1000.0);
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

        for (int i = 0; i < bd->n_shapes; i++) {
            if (i > 0) {
                printf("\n");
            }
            struct ggml_mulmat_bench_data_shape *shape = &bd->shapes[i];
            printf("#M@%dx%d", shape->N, shape->K);

            for (int j = 0; j < bd->num_m; j++) {
                printf(";%3d", shape->items[j].M);
            }
            printf("\n");

            for (int k = 0; k < num_nth; k++) {
                int nth = nth_list[k];

                printf("cpu_nth_%d", nth);
                for (int j = 0; j < bd->num_m; j++) {
                    double total = 0.0;
                    for (int stage = GGML_TASK_INIT;
                         stage <= GGML_TASK_FINALIZE; stage++) {
                        if (bd->cpu_stages[stage] & 1) {
                            int t = shape->items[j].cpu_time[stage];
                            if (bd->cpu_stages[stage] & ((1 << 1))) {
                                t /= nth;
                            }
                            total += t / 1000.0;
                        }
                    }
                    printf(";%8.3f", total);
                }
                printf("\n");

                printf("gpu_nth_%d", nth);
                for (int j = 0; j < bd->num_m; j++) {
                    double total = 0.0;
                    for (int stage = GGML_TASK_INIT;
                         stage <= GGML_TASK_FINALIZE; stage++) {
                        if (bd->gpu_stages[stage] & 1) {
                            int t = shape->items[j].gpu_time[stage];
                            if (bd->gpu_stages[stage] & (1 << 1)) {
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
    struct ggml_mulmat_bench_data bd = {
        .version = 1,
        .model = "7B",
        .gpu_impl = "OPENBLAS",
        .n_shapes = 1,
        .m_step = 8,
        .num_m = 2,
        .cpu_stages = {1, (2 | 1), 0},
        .gpu_stages = {(2 | 1), 1, 0},
    };
    bd.shapes = malloc(sizeof(struct ggml_mulmat_bench_data_shape) * bd.n_shapes);
    bd.shapes[0] = (struct ggml_mulmat_bench_data_shape){
        .N = 4096,
        .K = 4096,
    };
    bd.shapes[0].items = malloc(sizeof(struct ggml_mulmat_bench_data_item) * bd.num_m);
    bd.shapes[0].items[0] = (struct ggml_mulmat_bench_data_item){
        .M = 8,
        .cpu_time = {10, 20, 0},
        .gpu_time = {30, 40, 0},
    };
    bd.shapes[0].items[1] = (struct ggml_mulmat_bench_data_item){
        .M = 16,
        .cpu_time = {50, 60, 0},
        .gpu_time = {70, 80, 0},
    };

    const int N = bd.shapes[0].N;
    const int K = bd.shapes[0].K;

    const int nth = 1;

    // Test exact M equals.

    for (int i = 0; i < 2; i++) {
        bool is_cpu = (i == 0);
        for (int j = 0; j < bd.num_m; j++) {
            struct ggml_mulmat_bench_data_item *item = &bd.shapes[0].items[j];
            int M = item->M;

            int t = (i == 0) ? ggml_mulmat_estimate_time(&bd, M, N, K, nth, true)
                             : ggml_mulmat_estimate_time(&bd, M, N, K, nth, false);
            if (is_cpu) {
                BENCH_ASSERT_INT_EQUAL(t, item->cpu_time[0] + item->cpu_time[1],
                                       "#(i: %d, j: %d)", i, j);
            } else {
                BENCH_ASSERT_INT_EQUAL(t, item->gpu_time[0] + item->gpu_time[1],
                                       "#(i: %d, j: %d)", i, j);
            }
        }
    }

    // Test M out of range
    {
        const int M_arr[2] = {bd.shapes[0].items[0].M - 1,
                              bd.shapes[0].items[1].M + 1};
        int n = (int)(sizeof(M_arr) / sizeof(int));

        for (int i = 0; i < 2; i++) {
            bool is_cpu = (i == 0);
            for (int j = 0; j < n; j++) {
                int t = ggml_mulmat_estimate_time(&bd, M_arr[j], N, K, nth, is_cpu);
                BENCH_ASSERT_INT_EQUAL(t, -1, "#(i: %d, j: %d)", i, j);
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
                int t = ggml_mulmat_estimate_time(&bd, test_data[j].M, N, K,
                                      test_data[j].nth, is_cpu);
                if (is_cpu) {
                    BENCH_ASSERT_INT_EQUAL(t, test_data[j].expected_cpu_time,
                                           "#(i: %d, j: %d)", i, j);
                } else {
                    BENCH_ASSERT_INT_EQUAL(t, test_data[j].expected_gpu_time,
                                           "#(i: %d, j: %d)", i, j);
                }
            }
        }
    }
}

struct test__choose_device_data {
    int nth;
    int M;
    enum ggml_device_type expected_device;
};

static void test__choose_device(void) {
    struct ggml_mulmat_bench_data bd = {
        .version = 1,
        .model = "7B",
        .gpu_impl = "OPENBLAS",
        .n_shapes = 1,
        .m_step = 8,
        .num_m = 2,
        .cpu_stages = {1, (2 | 1), 0},
        .gpu_stages = {(2 | 1), 1, 0},
    };
    bd.shapes = malloc(sizeof(struct ggml_mulmat_bench_data_shape) * bd.n_shapes);
    bd.shapes[0] = (struct ggml_mulmat_bench_data_shape){
        .N = 4096,
        .K = 4096,
    };
    bd.shapes[0].items = malloc(sizeof(struct ggml_mulmat_bench_data_item) * bd.num_m);
    bd.shapes[0].items[0] = (struct ggml_mulmat_bench_data_item){
        .M = 8,
        .cpu_time = {10, 100, 0},
        .gpu_time = {100, 200, 0},
    };
    bd.shapes[0].items[1] = (struct ggml_mulmat_bench_data_item){
        .M = 16,
        .cpu_time = {20, 300, 0},
        .gpu_time = {100, 100, 0},
    };

    const int N = bd.shapes[0].N;
    const int K = bd.shapes[0].K;

    // When M out of range.
    {
        const int M_arr[2] = {bd.shapes[0].items[0].M - 1,
                              bd.shapes[0].items[1].M + 1};
        int n = (int)(sizeof(M_arr) / sizeof(int));

        for (int i = 1; i <= 8; i++) {
            int nth = i;
            for (int j = 0; j < n; j++) {
                enum ggml_device_type device =
                    ggml_mulmat_choose_device(&bd, M_arr[j], N, K, nth);
                if (j == 0) {
                    BENCH_ASSERT_INT_EQUAL(device, GGML_DEVICE_CPU,
                                           "#(i: %d, i: %d)", i, j);
                } else {
                    BENCH_ASSERT_INT_EQUAL(device, GGML_DEVICE_GPU,
                                           "#(i: %d, i: %d)", i, j);
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
                .expected_device = GGML_DEVICE_CPU,
            },
            {
                .nth = 1,
                .M = 12,
                .expected_device = GGML_DEVICE_CPU,
            },
            {
                .nth = 1,
                .M = 16,
                .expected_device = GGML_DEVICE_GPU,
            },
            {
                .nth = 2,
                .M = 8,
                .expected_device = GGML_DEVICE_CPU,
            },
            {
                .nth = 2,
                .M = 12,
                .expected_device = GGML_DEVICE_CPU,
            },
            {
                .nth = 2,
                .M = 16,
                .expected_device = GGML_DEVICE_GPU,
            },
            {
                .nth = 4,
                .M = 8,
                .expected_device = GGML_DEVICE_CPU,
            },
            {
                .nth = 4,
                .M = 12,
                .expected_device = GGML_DEVICE_CPU,
            },
            {
                .nth = 4,
                .M = 16,
                .expected_device = GGML_DEVICE_CPU,
            },
        };

        int n =
            (int)(sizeof(test_data) / sizeof(struct test__choose_device_data));

        for (int i = 0; i < n; i++) {
            const struct test__choose_device_data *e = &test_data[i];
            enum ggml_device_type device =
                ggml_mulmat_choose_device(&bd, e->M, N, K, e->nth);
            BENCH_ASSERT_INT_EQUAL(device, e->expected_device, "#(i: %d)", i);
        }
    }
}
