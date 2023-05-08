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

#define UNUSED(x) (void)(x)

#define NUM_BENCH 5

struct bench_data_item {
    int M;

    int cpu_time[3];
    int gpu_time[3];

    int cpu_records[3][NUM_BENCH];
    int gpu_records[3][NUM_BENCH];
};

struct bench_data_shape {
    int N;
    int K;

    struct bench_data_item *items;
};

// top bench data to write/read to/from file.
struct bench_data {
    int version;

    char model[4];     // 7B | 13B
    char gpu_impl[20]; // ACCELERATE, OPENBLAS, CUBLAS
    int n_shapes;
    int m_step;
    int num_m;

    // bit 0: valid, bit 1: can parallel
    // TODO: define macros.
    int cpu_stages[3];
    int gpu_stages[3];

    struct bench_data_shape *shapes;
};

struct model_nk_shape {
    int N;
    int K;
};

static int64_t time_us(void);
static void util__print_build_blas_tip(void);
static bool util__prompt_yes_no(const char *prompt);
static void util__progress(int i, int max);
static void util__envs_for_gpu_feature(int feature, char *buf, int buf_len);

static void write_bench_data(struct bench_data *bd, FILE *file);
static void read_bench_data(struct bench_data *bd, FILE *file);

static int bench_time_avg(int *a, int len);
static int estimate_time(struct bench_data *bd, int M, int N, int K, int nth,
                         bool is_cpu);
static enum ggml_device_type choose_device(struct bench_data *bd, int M, int N,
                                           int K, int nth);

static void cmd_bench(struct bench_data *bd);
static void cmd_analyze(struct bench_data *bd);
static void cmd_test(void);

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

// main
int main(int argc, char **argv) {
    printf("\n");

    if (!ggml_cpu_has_blas()) {
        util__print_build_blas_tip();
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

        struct bench_data bd = {
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

                    if (!util__prompt_yes_no(prompt)) {
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
            bd.shapes = (struct bench_data_shape[]){
                {.N = 4096, .K = 4096},
                {.N = 4096, .K = 11008},
                {.N = 11008, .K = 4096},
            };
        } else if (strcmp(model, "13B") == 0) {
            bd.n_shapes = 3,
            bd.shapes = malloc(bd.n_shapes * sizeof(struct model_nk_shape));
            BENCH_ASSERT(bd.shapes);
            bd.shapes = (struct bench_data_shape[]){
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

        write_bench_data(&bd, fp == NULL ? stdout : fp);
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

        struct bench_data bd;

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
        read_bench_data(&bd, fp);
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

void cmd_bench(struct bench_data *bd) {
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

        struct bench_data_shape *bench_shape = &bd->shapes[i];
        {
            size_t sz = sizeof(struct bench_data_item) * bd->num_m;
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

            struct bench_data_item *bench_item = &bench_shape->items[im];
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
                        util__progress(nb, NUM_BENCH);
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
                        util__progress(nb, NUM_BENCH);
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
            struct bench_data_item *item = &bd->shapes[i].items[j];
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

// for given work load and number of threads, estimate cpu or gpu time.
static int estimate_time(struct bench_data *bd, int M, int N, int K, int nth,
                         bool is_cpu) {
    struct bench_data_shape *shape = NULL;
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
        struct bench_data_item *item = &shape->items[i];
        if (item->M == M) {
            int total = 0;
            for (int j = GGML_TASK_INIT; j <= GGML_TASK_FINALIZE; j++) {
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
        struct bench_data_item *prev = &shape->items[i];
        struct bench_data_item *next = &shape->items[i + 1];
        // interpolate.
        if (M > prev->M && M < next->M) {
            double x = 1.0 * (M - prev->M) / (next->M - prev->M);
            int total = 0;
            for (int j = GGML_TASK_INIT; j <= GGML_TASK_FINALIZE; j++) {
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

static enum ggml_device_type choose_device(struct bench_data *bd, int M, int N,
                                           int K, int nth) {
    if (M < bd->m_step) {
        return GGML_DEVICE_CPU;
    } else if (M > bd->m_step * bd->num_m) {
        return GGML_DEVICE_GPU;
    }

    int cpu_time = estimate_time(bd, M, N, K, nth, true /* cpu */);
    int gpu_time = estimate_time(bd, M, N, K, nth, false /* gpu */);

    if (cpu_time < 0 && cpu_time < 0) {
        return GGML_DEVICE_AUTO;
    }

    return (cpu_time < gpu_time) ? GGML_DEVICE_CPU : GGML_DEVICE_GPU;
}

static int64_t time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}

static void util__print_build_blas_tip(void) {
    const char *make_target = "q40-mulmat-device-bench";

    fprintf(stderr,
            "error: this program was not built with any GPU feature. tips:\n");

    char buf[100];
    util__envs_for_gpu_feature(1, buf, 100);
    fprintf(stderr, "* to build with accelerate: make clean; %s make %s\n", buf,
            make_target);
    util__envs_for_gpu_feature(2, buf, 100);
    fprintf(stderr, "* to build with openBLAS:   make clean; %s make %s\n", buf,
            make_target);
    util__envs_for_gpu_feature(3, buf, 100);
    fprintf(stderr, "* to build with cuBLAS:     make clean; %s make %s\n", buf,
            make_target);
}

// feature: 1: apple accelerate, 2: openBLAS, 3: cuBLAS
static void util__envs_for_gpu_feature(int feature, char *buf, int buf_len) {
    memset(buf, 0, buf_len);
    const char *LLAMA_NO_ACCELERATE = feature == 1 ? " " : "1";
    const char *LLAMA_OPENBLAS = feature == 2 ? "1" : " ";
    const char *LLAMA_CUBLAS = feature == 3 ? "1" : " ";
    snprintf(buf, buf_len,
             "LLAMA_NO_ACCELERATE=%s LLAMA_OPENBLAS=%s LLAMA_CUBLAS=%s",
             LLAMA_NO_ACCELERATE, LLAMA_OPENBLAS, LLAMA_CUBLAS);
}

static bool util__prompt_yes_no(const char *prompt) {
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

static void util__progress(int i, int n) {
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

static void write_bench_data(struct bench_data *bd, FILE *fp) {
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
        struct bench_data_shape *s = &bd->shapes[i];

        fprintf(fp, "%d %d\n", s->N, s->K);

        for (int j = 0; j < bd->num_m; j++) {
            struct bench_data_item *item = &s->items[j];
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

static void read_bench_data(struct bench_data *bd, FILE *fp) {
    int rc = fscanf(fp, "%d %s %s %d %d %d", &bd->version, bd->model,
                    bd->gpu_impl, &bd->n_shapes, &bd->m_step, &bd->num_m);
    BENCH_ASSERT(rc > 0);

    for (int i = 0; i < 3; i++) {
        rc = fscanf(fp, "%1d", &bd->cpu_stages[i]);
        BENCH_ASSERT(rc > 0);
    }

    for (int i = 0; i < 3; i++) {
        rc = fscanf(fp, "%d", &bd->gpu_stages[i]);
        BENCH_ASSERT(rc > 0);
    }

    bd->shapes = malloc(sizeof(struct bench_data_shape) * bd->n_shapes);

    for (int i = 0; i < bd->n_shapes; i++) {
        struct bench_data_shape *s = &bd->shapes[i];

        rc = fscanf(fp, "%d%d", &s->N, &s->K);
        BENCH_ASSERT(rc > 0);

        s->items = malloc(sizeof(struct bench_data_item) * bd->num_m);

        for (int j = 0; j < bd->num_m; j++) {
            struct bench_data_item *item = &s->items[j];
            rc = fscanf(fp, "%d %d %d %d %d %d %d", &item->M,
                        &item->cpu_time[0], &item->cpu_time[1],
                        &item->cpu_time[2], &item->gpu_time[0],
                        &item->gpu_time[1], &item->gpu_time[2]);
            BENCH_ASSERT(rc > 0);
        }
    }
}

static void cmd_analyze(struct bench_data *bd) {
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
            struct bench_data_shape *s = &bd->shapes[i];
            size_t offset = i * buf_slot_size;
            snprintf(&buf[offset], buf_slot_size, "NxK=%dx%d", s->N, s->K);
            size_t len = strlen(&buf[offset]);
            if (len > max_nxk_len) {
                max_nxk_len = len;
            }
        }

        for (int i = 0; i < bd->n_shapes; i++) {
            struct bench_data_shape *s = &bd->shapes[i];

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
            struct bench_data_shape *s = &bd->shapes[i];
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
        for (int i = 0; i < bd->n_shapes; i++) {
            if (i > 0) {
                printf("\n");
            }
            struct bench_data_shape *s = &bd->shapes[i];
            printf("#M@%dx%d", s->N, s->K);

            for (int j = 0; j < bd->num_m; j++) {
                printf(";%3d", s->items[j].M);
            }
            printf("\n");

            for (int k = 0; k < 5; k++) {
                int nth = nth_list[k];

                printf("cpu_nth_%d", nth);
                for (int j = 0; j < bd->num_m; j++) {
                    double total = 0.0;
                    for (int stage = GGML_TASK_INIT;
                         stage <= GGML_TASK_FINALIZE; stage++) {
                        if (bd->cpu_stages[stage] & 1) {
                            int t = s->items[j].cpu_time[stage];
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
                            int t = s->items[j].gpu_time[stage];
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
    struct bench_data bd = {
        .version = 1,
        .model = "7B",
        .gpu_impl = "OPENBLAS",
        .n_shapes = 1,
        .m_step = 8,
        .num_m = 2,
        .cpu_stages = {1, (2 | 1), 0},
        .gpu_stages = {(2 | 1), 1, 0},
    };
    bd.shapes = malloc(sizeof(struct bench_data_shape) * bd.n_shapes);
    bd.shapes[0] = (struct bench_data_shape){
        .N = 4096,
        .K = 4096,
    };
    bd.shapes[0].items = malloc(sizeof(struct bench_data_item) * bd.num_m);
    bd.shapes[0].items[0] = (struct bench_data_item){
        .M = 8,
        .cpu_time = {10, 20, 0},
        .gpu_time = {30, 40, 0},
    };
    bd.shapes[0].items[1] = (struct bench_data_item){
        .M = 16,
        .cpu_time = {50, 60, 0},
        .gpu_time = {70, 80, 0},
    };

    const double error_bound = 0.01;

    const int nth = 1;
    const int N = bd.shapes[0].N;
    const int K = bd.shapes[0].K;
    const int num_m = bd.num_m;
    const int m_step = bd.m_step;

    BENCH_ASSERT(num_m >= 2);
    BENCH_ASSERT(m_step % 2 == 0);

    const int m_max = m_step * num_m;

    const int Ms[4] = {m_step, m_step + m_step / 2, m_step * 2, m_max + 1};

    int T[4];

    for (int i = 0; i < 2; i++) {
        printf("\nestimate %s time\n", i == 0 ? "CPU" : "GPU");

        for (int j = 0; j < 4; j++) {
            int M = Ms[j];
            T[j] = (i == 0) ? estimate_time(&bd, M, N, K, nth, true)
                            : estimate_time(&bd, M, N, K, nth, false);
            printf("M: %3d, N: %5d, K: %5d, nth: %d, time: %7d\n", M, N, K, nth,
                   T[j]);
        }

        int sum = T[0] + T[2];
        double diff = sum - 2 * T[1];
        if (diff < 0) {
            diff = -diff;
        }
        BENCH_ASSERT((diff / sum) < error_bound);
        BENCH_ASSERT(T[3] == -1);
    }
}
