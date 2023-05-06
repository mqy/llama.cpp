#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <inttypes.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(GGML_USE_OPENBLAS)
#include <cblas.h>
#endif

// For single thread:
// - It's almost true that: cpu is fast when M < 16; gpu is fast when M > 64.
// - Most of the time, when M < 32 cpu is fast.
// But when n_threads > 1, when M in range [8, 88], either cpu or gpu wins.

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

struct bench_params {
    char *model;
    const struct model_nk_shape *shapes;
    int n_shapes;
    int m_step;
    int num_m;
};

struct bench_data_stats {
    int cpu_init[NUM_BENCH];
    int cpu_comp[NUM_BENCH];

    int gpu_init[NUM_BENCH];
    int gpu_comp[NUM_BENCH];
};

struct bench_data_item {
    int M;

    // avg.
    int cpu_init_avg;
    int cpu_comp_avg;

    int gpu_init_avg;
    int gpu_comp_avg;

    struct bench_data_stats stats;
};

struct bench_data_shape {
    int N;
    int K;

    // M range: m_step * [1..num_m]
    int m_step;

    int num_m;
    struct bench_data_item *items;
};

// top bench data to write/read to/from file.
struct bench_data {
    char model[4]; // 7B | 13B
    int n_shapes;
    struct bench_data_shape *shapes;
};

struct model_nk_shape {
    int N;
    int K;
};

const struct model_nk_shape model_nk_shape_7b[] = {
    {4096, 4096},
    {4096, 11008},
    {11008, 4096},
};

const struct model_nk_shape model_nk_shape_13b[] = {
    {5120, 5120},
    {5120, 13824},
    {13824, 5120},
};

// enum ggml_device_type {
//     GGML_DEVICE_CPU = 1,
//     GGML_DEVICE_GPU = 2,
// }

// copied from ggml.c
enum ggml_task_type {
    GGML_TASK_INIT = 0,
    GGML_TASK_COMPUTE,
    GGML_TASK_FINALIZE,
};

// copied from ggml.c
struct ggml_compute_params {
    enum ggml_task_type type;
    int n_threads;

    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void *wdata;

    // add device
    enum ggml_device_type device;
};

static int64_t time_us(void);
static bool util__yes_no(const char *prompt);
static void util__progress(int i, int max);

static int bench_time_avg(int *a, int len);
static void write_bench_data(struct bench_data *bd, FILE *file);
static void read_bench_data(struct bench_data *bd, FILE *file);

static int estimate_time(struct bench_data *bd, int M, int N, int K, int nth,
                         bool is_cpu);

static void mock__ggml_compute_forward_mul_mat_q_f32(
    const struct ggml_compute_params *params, const struct ggml_tensor *src0,
    const struct ggml_tensor *src1, struct ggml_tensor *dst, int M, int N,
    int K);

static void cmd_bench(const struct bench_params *params, struct bench_data *bd);
static void cmd_analyze(struct bench_data *bd);
static void cmd_test(struct bench_data *bd);

static void usage(char *prog) {
    fprintf(stderr,
            "usage: %s <command args ...>\n"
            "\t%s bench   <model> [data-file], where: model is 7B or 13B, the "
            "optional data-file is used to write bench result to\n"
            "\t%s analyze <data-file>,         where: data-file is used to "
            "read bench result from\n"
            "\t%s test    <data-file>,         where: data-file is used to "
            "read bench result from\n",
            prog, prog, prog, prog);
}

// main
int main(int argc, char **argv) {
    printf("\n");

#if !(defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS))
    fprintf(stderr, "GGML_USE_ACCELERATE or GGML_USE_OPENBLAS: undefined\n"
                    "build with accelerate: LLAMA_NO_ACCELERATE=  "
                    "LLAMA_OPENBLAS=   make q4_0-mulmat-bench"
                    "build with openblas:   LLAMA_NO_ACCELERATE=1 "
                    "LLAMA_OPENBLAS=1  make q4_0-mulmat-bench");
    exit(1);
#endif

    if (argc < 2) {
        fprintf(stderr, "need sub command");
        usage(argv[0]);
        exit(1);
    }

    char *cmd = argv[1];

    if (strcmp(cmd, "bench") == 0) {
        if (argc < 3) {
            fprintf(stderr, "bench: too few args");
            usage(argv[0]);
            exit(1);
        }

        struct bench_params params = {
            .n_shapes = 3,
            .m_step = 8,
            .num_m = 11,
            .shapes = NULL,
        };

        if (false) {
            // for M from 16 through 512.
            params.m_step = 32;
            params.num_m = 16;
        }

        params.model = argv[2];

        const char *data_file = NULL;
        FILE *fp = NULL;

        if (argc == 4) {
            data_file = argv[3];

            {
                struct stat st;
                int rc = stat(data_file, &st);
                UNUSED(st);
                if (rc == 0) { // prompt
                    size_t len = strlen(data_file) + 40;
                    char *prompt = malloc(len);
                    BENCH_ASSERT(prompt);
                    snprintf(prompt, len,
                             "%s: data file '%s' exists, override? (Y|n)", cmd,
                             data_file);

                    if (!util__yes_no(prompt)) {
                        printf("Aborted.\n");
                        exit(2);
                    }
                    free(prompt);
                }
            }

            fp = fopen(data_file, "w");
            BENCH_ASSERT(fp);
        }

        if (strcmp(params.model, "7B") == 0) {
            params.shapes = model_nk_shape_7b;
        } else if (strcmp(params.model, "13B") == 0) {
            params.shapes = model_nk_shape_13b;
        } else {
            fprintf(stderr, "%s: unsupported model: %s", cmd, params.model);
            usage(argv[0]);
            exit(1);
        }

        struct bench_data bd;
        cmd_bench(&params, &bd);

        printf("\n");
        write_bench_data(&bd, fp == NULL ? stdout : fp);
        if (fp != NULL) {
            fclose(fp);
        }

        if (data_file != NULL) {
            printf("%s: result was written to %s\n", cmd, data_file);
        }
        printf("\n%s: done!\n", cmd);
    } else if (strcmp(cmd, "analyze") == 0) {
        if (argc < 3) {
            fprintf(stderr, "%s: too few args", cmd);
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
                fprintf(stderr, "%s: data file not exists: %s\n", cmd,
                        data_file);
                exit(1);
            }
        }

        FILE *fp = fopen(data_file, "r");
        BENCH_ASSERT(fp);
        read_bench_data(&bd, fp);
        fclose(fp);

        cmd_analyze(&bd);
        printf("\n%s: done!\n", cmd);
    } else if (strcmp(cmd, "test") == 0) {
        if (argc < 3) {
            fprintf(stderr, "%s: too few args", cmd);
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
                fprintf(stderr, "%s: data file not exists: %s\n", cmd,
                        data_file);
                exit(1);
            }
        }

        FILE *fp = fopen(data_file, "r");
        BENCH_ASSERT(fp);
        read_bench_data(&bd, fp);
        fclose(fp);

        cmd_test(&bd);
        printf("\n%s: done!\n", cmd);
    } else {
        fprintf(stderr, "unknown command: %s", cmd);
        usage(argv[0]);
        exit(1);
    }

    return 0;
}

void cmd_bench(const struct bench_params *params, struct bench_data *bd) {
    size_t wdata_size = 0;
    void *q4_0_buf = NULL;
    void *wdata = NULL;

    // alloc q4_0_buf and wdata with max size.
    {
        size_t max_NxK = 0;
        for (int i = 0; i < params->n_shapes; i++) {
            size_t sz = params->shapes[i].N * params->shapes[i].K;
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

    {
        BENCH_ASSERT(params->model);
        memset(bd->model, 0, 4);
        strncpy(bd->model, params->model, sizeof(bd->model) - 1);
        bd->n_shapes = params->n_shapes;
        bd->shapes = NULL;

        size_t sz = sizeof(struct bench_data_shape) * params->n_shapes;
        bd->shapes = malloc(sz);
        BENCH_ASSERT(bd->shapes);
        memset(bd->shapes, 0, sz);
    }

    for (int i = 0; i < params->n_shapes; i++) {
        int N = params->shapes[i].N;
        int K = params->shapes[i].K;

        struct bench_data_shape *bench_shape = &bd->shapes[i];
        {
            bench_shape->N = N;
            bench_shape->K = K;
            bench_shape->m_step = params->m_step;
            bench_shape->num_m = params->num_m;

            size_t sz = sizeof(struct bench_data_item) * bench_shape->num_m;
            bench_shape->items = malloc(sz);
            BENCH_ASSERT(bench_shape->items);
            memset(bench_shape->items, 0, sz);
        }

        int M;
        for (int im = 0; im < bench_shape->num_m; im++) {
            M = params->m_step * (im + 1);
            printf("%5d %5d %3d ", N, K, M);
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
                .wsize = wdata_size,
                .wdata = wdata,
            };

            {
                compute_params.device = GGML_DEVICE_CPU;

                compute_params.type = GGML_TASK_INIT;
                memset(wdata, 0, wdata_size);

                for (int nb = 0; nb < NUM_BENCH; nb++) {
                    int t0 = (int)time_us();
                    mock__ggml_compute_forward_mul_mat_q_f32(
                        &compute_params, src0, src1, dst, M, N, K);
                    bench_item->stats.cpu_init[nb] = (int)time_us() - t0;
                    util__progress(nb, NUM_BENCH);
                }

                compute_params.type = GGML_TASK_COMPUTE;
                memset(wdata, 0, wdata_size);

                for (int nb = 0; nb < NUM_BENCH; nb++) {
                    int t0 = (int)time_us();
                    mock__ggml_compute_forward_mul_mat_q_f32(
                        &compute_params, src0, src1, dst, M, N, K);
                    bench_item->stats.cpu_comp[nb] = (int)time_us() - t0;
                    util__progress(nb, NUM_BENCH);
                }
            }

            {
                compute_params.device = GGML_DEVICE_GPU;

                compute_params.type = GGML_TASK_INIT;

                for (int nb = 0; nb < NUM_BENCH; nb++) {
                    int t0 = (int)time_us();
                    mock__ggml_compute_forward_mul_mat_q_f32(
                        &compute_params, src0, src1, dst, M, N, K);
                    bench_item->stats.gpu_init[nb] = (int)time_us() - t0;
                    util__progress(nb, NUM_BENCH);
                }

                compute_params.type = GGML_TASK_COMPUTE;
                // gpu comp (single thread).
                for (int nb = 0; nb < NUM_BENCH; nb++) {
                    int t0 = (int)time_us();
                    mock__ggml_compute_forward_mul_mat_q_f32(
                        &compute_params, src0, src1, dst, M, N, K);
                    bench_item->stats.gpu_comp[nb] = (int)time_us() - t0;
                    util__progress(nb, NUM_BENCH);
                }
            }
            printf("\n");

            ggml_free(ctx);
        }
    }

    free(wdata);
    free(q4_0_buf);

    // stats -> avg.
    for (int i = 0; i < params->n_shapes; i++) {
        for (int j = 0; j < bd->shapes[i].num_m; j++) {
            struct bench_data_item *item = &bd->shapes[i].items[j];
            item->cpu_init_avg =
                bench_time_avg(item->stats.cpu_init, NUM_BENCH);
            item->cpu_comp_avg =
                bench_time_avg(item->stats.cpu_comp, NUM_BENCH);
            item->gpu_init_avg =
                bench_time_avg(item->stats.gpu_init, NUM_BENCH);
            item->gpu_comp_avg =
                bench_time_avg(item->stats.gpu_comp, NUM_BENCH);
        }
    }
}

static void mock__ggml_compute_forward_mul_mat_q_f32(
    const struct ggml_compute_params *params, const struct ggml_tensor *src0,
    const struct ggml_tensor *src1, struct ggml_tensor *dst, int M, int N,
    int K) {

    quantize_fns_t funcs = ggml_internal_get_quantize_fn(GGML_TYPE_Q4_0);
    dequantize_row_q_t dequantize_row_q = funcs.dequantize_row_q;
    quantize_row_q_t quantize_row_q = funcs.quantize_row_q;
    vec_dot_q_t vec_dot_q = funcs.vec_dot_q;

    if (params->device == GGML_DEVICE_CPU) {
        if (params->type == GGML_TASK_INIT) {
            for (int m = 0; m < M; m++) {
                quantize_row_q((float *)((char *)src1->data + m * K),
                               (char *)params->wdata + m * src1->nb[1], K);
            }
        } else if (params->type == GGML_TASK_COMPUTE) {
            for (int m = 0; m < M; m++) {
                float *src0_row = (float *)src0->data + m * N;
                for (int n = 0; n < N; n++) {
                    vec_dot_q(K, (float *)dst->data + m * N, src0_row,
                              (float *)params->wdata + m * K + n);
                }
            }
        } else {
            abort();
        }
    } else if (params->device == GGML_DEVICE_GPU) {
        if (params->type == GGML_TASK_INIT) {
            for (int n = 0; n < N; n++) {
                dequantize_row_q((const float *)src0->data + n * K,
                                 (float *)params->wdata + n * K, K);
            }
        } else if (params->type == GGML_TASK_COMPUTE) {
            const int lda = K;
            const int ldb = K;
            const int ldc = N;

            const float alpha = 1.0f;
            const float beta = 0.0f;

            const float *A = (float *)src1->data;
            const float *B = (float *)params->wdata;
            float *C = (float *)dst->data;

#if (defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS))
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha,
                        A, lda, B, ldb, beta, C, ldc);
#endif
        } else {
            abort();
        }
    } else {
        abort();
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

    if (M < shape->m_step || M > shape->m_step * shape->num_m) {
        return -1;
    }

    if (is_cpu) {
        for (int i = 0; i < shape->num_m; i++) {
            struct bench_data_item *item = &shape->items[i];
            if (item->M == M) {
                return item->cpu_init_avg + item->cpu_comp_avg / nth;
            }
        }

        for (int i = 0; i < shape->num_m - 1; i++) {
            struct bench_data_item *prev = &shape->items[i];
            struct bench_data_item *next = &shape->items[i + 1];
            // interpolate.
            if (M > prev->M && M < next->M) {
                double x = 1.0 * (M - prev->M) / (next->M - prev->M);
                double init = prev->cpu_init_avg +
                              (next->cpu_init_avg - prev->cpu_init_avg) * x;
                double comp =
                    prev->cpu_comp_avg +
                    (next->cpu_comp_avg - prev->cpu_comp_avg) * x / nth;
                return (int)(init + comp);
            }
        }
    } else {
        for (int i = 0; i < shape->num_m; i++) {
            struct bench_data_item *item = &shape->items[i];
            if (item->M == M) {
                return item->gpu_init_avg / nth + item->gpu_comp_avg;
            }
        }

        for (int i = 0; i < shape->num_m - 1; i++) {
            struct bench_data_item *prev = &shape->items[i];
            struct bench_data_item *next = &shape->items[i + 1];

            // interpolate.
            if (M > prev->M && M < next->M) {
                double x = 1.0 * (M - prev->M) / (next->M - prev->M);
                double init =
                    prev->gpu_init_avg +
                    (next->gpu_init_avg - prev->gpu_init_avg) * x / nth;
                double comp = prev->gpu_comp_avg +
                              (next->gpu_comp_avg - prev->gpu_comp_avg) * x;
                return (int)(init + comp);
            }
        }
    }

    return -1;
}

static int64_t time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}

static bool util__yes_no(const char *prompt) {
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
        putchar('\b');
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
    fprintf(fp, "%s %d\n", bd->model, bd->n_shapes);

    for (int i = 0; i < bd->n_shapes; i++) {
        struct bench_data_shape *s = &bd->shapes[i];

        fprintf(fp, "%d %d %d %d\n", s->N, s->K, s->m_step, s->num_m);

        for (int j = 0; j < s->num_m; j++) {
            struct bench_data_item *item = &s->items[j];
            fprintf(fp, "%3d %7d %7d %7d %7d\n", item->M, item->cpu_init_avg,
                    item->cpu_comp_avg, item->gpu_init_avg, item->gpu_comp_avg);
        }
    }
}

static void read_bench_data(struct bench_data *bd, FILE *fp) {
    int rc = fscanf(fp, "%s %d", bd->model, &bd->n_shapes);
    BENCH_ASSERT(rc > 0);

    bd->shapes = malloc(sizeof(struct bench_data_shape) * bd->n_shapes);

    for (int i = 0; i < bd->n_shapes; i++) {
        struct bench_data_shape *s = &bd->shapes[i];

        rc = fscanf(fp, "%d%d%d%d", &s->N, &s->K, &s->m_step, &s->num_m);
        BENCH_ASSERT(rc > 0);

        s->items = malloc(sizeof(struct bench_data_item) * s->num_m);

        for (int j = 0; j < s->num_m; j++) {
            struct bench_data_item *item = &s->items[j];
            rc = fscanf(fp, "%d%d%d%d%d", &item->M, &item->cpu_init_avg,
                        &item->cpu_comp_avg, &item->gpu_init_avg,
                        &item->gpu_comp_avg);
            BENCH_ASSERT(rc > 0);
        }
    }
}

static void cmd_analyze(struct bench_data *bd) {
    printf("\ngpu_comp for all shapes:\n\n");
    {
        int num_m = bd->shapes[0].num_m;

        printf("#M");
        for (int i = 0; i < num_m; i++) {
            printf(";%3d", bd->shapes[0].items[i].M);
        }
        printf("\n");

        // Nothing but for pretty align.
        size_t buf_slot_size = 24;
        char * buf = malloc(buf_slot_size * bd->n_shapes);

        size_t max_nxk_len = 0;
        for (int i = 0; i < bd->n_shapes; i++) {
            struct bench_data_shape *s = &bd->shapes[i];
            size_t offset = i*buf_slot_size;
            snprintf(&buf[offset], buf_slot_size, "NxK=%dx%d", s->N, s->K);
            size_t len = strlen(&buf[offset]);
            if (len > max_nxk_len) {
                max_nxk_len = len;
            }
        }

        for (int i = 0; i < bd->n_shapes; i++) {
            struct bench_data_shape *s = &bd->shapes[i];

            size_t offset = i*buf_slot_size;
            printf("%s", &buf[offset]);
            for (int j = 0; j < (int)(max_nxk_len - strlen(&buf[offset])); j++) {
                printf(" ");
            }

            for (int j = 0; j < num_m; j++) {
                printf(";%8.3f", s->items[j].gpu_comp_avg / 1000.0);
            }
            printf("\n");
        }

        free(buf);
    }

    printf("\ndetails for each shape: \n\n");
    {
        for (int i = 0; i < bd->n_shapes; i++) {
            struct bench_data_shape *s = &bd->shapes[i];
            printf("#M@NxK=%dx%d", s->N, s->K);

            for (int j = 0; j < s->num_m; j++) {
                printf(";%3d", s->items[j].M);
            }
            printf("\n");

            printf("cpu_init");
            for (int j = 0; j < s->num_m; j++) {
                printf(";%8.3f", s->items[j].cpu_init_avg / 1000.0);
            }
            printf("\n");

            printf("cpu_comp");
            for (int j = 0; j < s->num_m; j++) {
                printf(";%8.3f", s->items[j].cpu_comp_avg / 1000.0);
            }
            printf("\n");

            printf("gpu_init");
            for (int j = 0; j < s->num_m; j++) {
                printf(";%8.3f", s->items[j].gpu_init_avg / 1000.0);
            }
            printf("\n");

            printf("gpu_comp");
            for (int j = 0; j < s->num_m; j++) {
                printf(";%8.3f", s->items[j].gpu_comp_avg / 1000.0);
            }
            printf("\n\n");
        }
    }

    printf("n_threads affects: \n\n");
    {
        const int nth_list[5] = {1, 2, 4, 6, 8};
        for (int i = 0; i < bd->n_shapes; i++) {
            if (i > 0) {
                printf("\n");
            }
            struct bench_data_shape *s = &bd->shapes[i];
            printf("#M@NxK=%dx%d", s->N, s->K);

            for (int j = 0; j < s->num_m; j++) {
                printf(";%3d", s->items[j].M);
            }
            printf("\n");

            for (int k = 0; k < 5; k++) {
                int nth = nth_list[k];

                printf("cpu_nth_%d", nth);
                for (int j = 0; j < s->num_m; j++) {
                    printf(";%8.3f", (s->items[j].cpu_init_avg +
                                      s->items[j].cpu_comp_avg / nth) /
                                         1000.0);
                }
                printf("\n");

                printf("gpu_nth_%d", nth);
                for (int j = 0; j < s->num_m; j++) {
                    printf(";%8.3f", (s->items[j].gpu_init_avg / nth +
                                      s->items[j].gpu_comp_avg) /
                                         1000.0);
                }
                printf("\n");
            }
        }
    }
}

static void cmd_test(struct bench_data *bd) {
    const struct bench_data_shape *shape = &bd->shapes[0];

    const double error_bound = 0.01;

    // These can be read from data file.
    const int nth = 1;
    const int N = shape->N;
    const int K = shape->K;
    const int num_m = shape->num_m;
    const int m_step = shape->m_step;

    BENCH_ASSERT(num_m > 2);
    BENCH_ASSERT(m_step % 2 == 0);

    const int m_max = m_step * num_m;

    const int Ms[4] = {m_step, m_step + m_step / 2, m_step * 2, m_max + 1};

    int T[4];

    for (int i = 0; i < 2; i++) {
        printf("\nestimate %s time\n", i == 0 ? "CPU" : "GPU");

        for (int j = 0; j < 4; j++) {
            int M = Ms[j];
            T[j] = (i == 0) ? estimate_time(bd, M, N, K, nth, true)
                            : estimate_time(bd, M, N, K, nth, false);
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
