#include "ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Build Apple Accelerate:
//     LLAMA_NO_ACCELERATE=1 LLAMA_OPENBLAS= make q4_0-mulmat-bench

// Build OPENBLAS:
//     LLAMA_NO_ACCELERATE= LLAMA_OPENBLAS=1 make q4_0-mulmat-bench

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(GGML_USE_OPENBLAS)
#include <cblas.h>
#endif

// typedef long long int64_t;

// For single thread:
// - It's almost true that: cpu is fast when M < 16; gpu is fast when M > 64.
// - Most of the time, when M < 32 cpu is fast.
// But when n_threads > 1, when M in range [8, 80], either cpu or gpu.
#define NUM_M 11
#define M_STEP 8

#define NUM_NK_PAIRS 3

#define NUM_BENCH 5

#define BENCH_ASSERT(x)                                                        \
    do {                                                                       \
        if (!(x)) {                                                            \
            fprintf(stderr, "BENCH_ASSERT: %s:%d: %s\n", __FILE__, __LINE__,   \
                    #x);                                                       \
            abort();                                                           \
        }                                                                      \
    } while (0)

#define UNUSED(x) (void)(x)

// TODO: remove bench, bench_m, bench_nk
struct bench_data_stats {
    int64_t cpu_init[NUM_BENCH];
    int64_t cpu_comp[NUM_BENCH];

    int64_t gpu_init[NUM_BENCH];
    int64_t gpu_comp[NUM_BENCH];
};

// sub data record to write/read to/from file.
struct bench_data_item {
    int32_t M;

    // avg.
    int64_t cpu_init;
    int64_t cpu_comp;
    int64_t gpu_init;
    int64_t gpu_comp;

    struct bench_data_stats stats;
};

// sub data record to write/read to/from file.
struct bench_data_shape {
    int32_t N;
    int32_t K;

    // M range: m_step * [1..num_m]
    int32_t m_step;

    int32_t num_m;
    struct bench_data_item *items;
};

// top bench data to write/read to/from file.
struct bench_data {
    char model[4];
    int32_t n_shapes;
    struct bench_data_shape *shapes;
};

struct nk_pair {
    int32_t N;
    int32_t K;
};

const struct nk_pair nk_pairs_7b[] = {
    {4096, 4096},
    {4096, 11008},
    {11008, 4096},
};

const struct nk_pair nk_pairs_13b[] = {
    {5120, 5120},
    {5120, 13824},
    {13824, 5120},
};

static int64_t time_us(void);
static int64_t bench_time_avg(int64_t *a, int len);
static void write_bench_data_csv(struct bench_data *bd, FILE *file);
static void write_bench_data(struct bench_data *bd, FILE *file);
static void read_bench_data(struct bench_data *bd, FILE *file);

static int64_t estimate_time(struct bench_data *bd, int M, int N, int K,
                                 int nth, bool is_cpu);

static void test_compare_bench_data(struct bench_data *bd,
                                    struct bench_data *bd2);
static void test_estimate_xpu_time(void);

static void usage(char *prog) {
    fprintf(stderr, "usage: %s <model>, where models can be 7B or 13B\n", prog);
}

// main
int main(int argc, char **argv) {
#if !(defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS))
    fprintf(stderr, "GGML_USE_ACCELERATE or GGML_USE_OPENBLAS: undefined\n");
    exit(1);
#endif

    // if (true) {
    //     test_estimate_xpu_time();
    //     exit(0);
    // }

    if (argc != 2) {
        usage(argv[0]);
        exit(1);
    }

    const struct nk_pair *nk_pairs = NULL;

    char *model = argv[1];

    if (strcmp(model, "7B") == 0) {
        nk_pairs = nk_pairs_7b;
    } else if (strcmp(model, "13B") == 0) {
        nk_pairs = nk_pairs_13b;
    } else {
        usage(argv[0]);
        exit(1);
    }

    quantize_fns_t funcs = ggml_internal_get_quantize_fn(GGML_TYPE_Q4_0);
    dequantize_row_q_t dequantize_row_q = funcs.dequantize_row_q;
    quantize_row_q_t quantize_row_q = funcs.quantize_row_q;
    vec_dot_q_t vec_dot_q = funcs.vec_dot_q;

    void *q4_0_buf = NULL;
    size_t wdata_size = 0;
    void *wdata = NULL;
    {
        size_t max_NxK = 0;
        for (int i = 0; i < NUM_NK_PAIRS; i++) {
            size_t sz = nk_pairs[i].N * nk_pairs[i].K;
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

    struct bench_data bd = {
        .n_shapes = NUM_NK_PAIRS,
        .shapes = NULL,
    };
    strncpy(bd.model, model, strlen(model));

    size_t sz = sizeof(struct bench_data_shape) * NUM_NK_PAIRS;
    bd.shapes = malloc(sz);
    memset(bd.shapes, 0, sz);

    for (int i = 0; i < NUM_NK_PAIRS; i++) {
        int N = nk_pairs[i].N;
        int K = nk_pairs[i].K;

        const int lda = K;
        const int ldb = K;
        const int ldc = N;

        struct bench_data_shape *bench_shape = &bd.shapes[i];
        bench_shape->N = N;
        bench_shape->K = K;
        bench_shape->m_step = M_STEP;
        bench_shape->num_m = NUM_M;

        sz = sizeof(struct bench_data_item) * NUM_M;
        bench_shape->items = malloc(sz);
        memset(bench_shape->items, 0, sz);

        int32_t M;
        for (int im = 0; im < NUM_M; im++) {
            M = M_STEP * (im + 1);
            struct bench_data_item *bench_item = &bench_shape->items[im];
            bench_item->M = M;

            size_t ctx_size = K * N * ggml_type_sizef(GGML_TYPE_F32) +
                              K * sizeof(float) + 1024 * 1024 * 300;

            struct ggml_init_params params = {
                .mem_size = ctx_size,
                .mem_buffer = NULL,
                .no_alloc = 0,
            };

            struct ggml_context *ctx = ggml_init(params);
            if (!ctx) {
                fprintf(stderr, "Error: ggml_init() returned empty ctx\n");
                return -1;
            }

            // src0: K x N
            struct ggml_tensor *src0_f32 =
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            ggml_set_f32(src0_f32, 0.1f);

            struct ggml_tensor *src0 =
                ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
            ggml_quantize_q4_0((const float *)src0_f32->data, src0->data, N, K,
                               (int64_t *)q4_0_buf);

            // src1: M x K
            struct ggml_tensor *src1 =
                ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
            ggml_set_f32(src1, 0.5f);

            memset(wdata, 0, wdata_size);

            size_t sizeC = M * N * sizeof(float);
            void *C = malloc(sizeC);
            memset(C, 0, sizeC);

            // cpu init (very fast: several us)
            for (int nb = 0; nb < NUM_BENCH; nb++) {
                int64_t t0 = time_us();
                for (int64_t m = 0; m < M; m++) {
                    quantize_row_q((float *)((char *)src1->data + m * K),
                                   (char *)wdata + m * src1->nb[1], K);
                }
                bench_item->stats.cpu_init[nb] = time_us() - t0;
            }

            memset(wdata, 0, wdata_size);

            // cpu comp (support multi-threads)
            for (int nb = 0; nb < NUM_BENCH; nb++) {
                int64_t t0 = time_us();
                for (int m = 0; m < M; m++) {
                    float *src0_row = (float *)src0->data + m * N;
                    for (int n = 0; n < N; n++) {
                        vec_dot_q(K, (float *)C + m * N, src0_row,
                                  (float *)wdata + m * K + n);
                    }
                }
                bench_item->stats.cpu_comp[nb] = time_us() - t0;
            }

            // gpu init (support multi-threads)
            for (int nb = 0; nb < NUM_BENCH; nb++) {
                int64_t t0 = time_us();
                for (int n = 0; n < N; n++) {
                    dequantize_row_q((const float *)src0->data + n * K,
                                     (float *)wdata + n * K, K);
                }
                bench_item->stats.gpu_init[nb] = time_us() - t0;
            }

            const float alpha = 1.0f;
            const float beta = 0.0f;
            const float *A = (float *)src1->data;
            const float *B = (float *)wdata;

            // gpu comp (single thread).
            for (int nb = 0; nb < NUM_BENCH; nb++) {
                int64_t t0 = time_us();
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K,
                            alpha, A, lda, B, ldb, beta, (float *)C, ldc);
                bench_item->stats.gpu_comp[nb] = time_us() - t0;
            }

            ggml_free(ctx);
            free(C);
        }
    }

    free(wdata);
    free(q4_0_buf);

    // stats -> avg.
    for (int i = 0; i < NUM_NK_PAIRS; i++) {
        for (int j = 0; j < NUM_M; j++) {
            struct bench_data_item *item = &bd.shapes[i].items[j];
            item->cpu_init = bench_time_avg(item->stats.cpu_init, NUM_BENCH);
            item->cpu_comp = bench_time_avg(item->stats.cpu_comp, NUM_BENCH);
            item->gpu_init = bench_time_avg(item->stats.gpu_init, NUM_BENCH);
            item->gpu_comp = bench_time_avg(item->stats.gpu_comp, NUM_BENCH);
        }
    }

    // TODO: free bd and bd2.

    printf("\n====== print as csv: \n\n");

    write_bench_data_csv(&bd, stdout);

    printf("\n====== print as txt: \n\n");
    write_bench_data(&bd, stdout);

    if (true) {
        char file_name[64];
        snprintf(file_name, sizeof(file_name), "q4_0-mulmat-bench.%s.txt",
                 model);

        printf("\n====== write to txt file: %s", file_name);
        FILE *fp = fopen(file_name, "w");
        write_bench_data(&bd, fp);
        fclose(fp);

        struct bench_data bd2;

        printf("\n====== read from txt file: %s", file_name);
        fp = fopen(file_name, "r");
        read_bench_data(&bd2, fp);
        fclose(fp);
        write_bench_data(&bd2, stdout);

        // compare bd and bd2
        test_compare_bench_data(&bd, &bd2);
    }

    return 0;
}

// for given work load and number of threads, estimate gpu time.
static int64_t estimate_time(struct bench_data *bd, int M, int N, int K,
                                 int nth, bool is_cpu) {
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
                return item->cpu_init + item->cpu_comp / nth;
            }
        }

        for (int i = 0; i < shape->num_m - 1; i++) {
            struct bench_data_item *prev = &shape->items[i];
            struct bench_data_item *next = &shape->items[i + 1];
            // interpolate.
            if (M > prev->M && M < next->M) {
                double x = 1.0 * (M - prev->M) / (next->M - prev->M);
                double init =
                    prev->cpu_init + (next->cpu_init - prev->cpu_init) * x;
                double comp =
                    prev->cpu_comp + (next->cpu_comp - prev->cpu_comp) * x / nth;
                return (int64_t)(init + comp);
            }
        }
    } else {
        for (int i = 0; i < shape->num_m; i++) {
            struct bench_data_item *item = &shape->items[i];
            if (item->M == M) {
                return item->gpu_init / nth + item->gpu_comp;
            }
        }

        for (int i = 0; i < shape->num_m - 1; i++) {
            struct bench_data_item *prev = &shape->items[i];
            struct bench_data_item *next = &shape->items[i + 1];

            // interpolate.
            if (M > prev->M && M < next->M) {
                double x = 1.0 * (M - prev->M) / (next->M - prev->M);
                double init =
                    prev->gpu_init + (next->gpu_init - prev->gpu_init) * x / nth;
                double comp =
                    prev->gpu_comp + (next->gpu_comp - prev->gpu_comp) * x;
                return (int64_t)(init + comp);
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

static int64_t bench_time_avg(int64_t *a, int len) {
    // bubble sort `a`.
    for (int i = 0; i < len - 1; i++) {
        for (int j = i + 1; j < len; j++) {
            if (a[j] < a[i]) {
                int64_t temp = a[j];
                a[j] = a[i];
                a[i] = temp;
            }
        }
    }

    int64_t total = 0;
    // throw away min and max
    for (int i = 1; i < len - 1; i++) {
        total += a[i];
    }
    return total / (len - 2);
}

// print benchmark of gpu_comp as csv: for plotting lines.
static void write_bench_data_csv(struct bench_data *bd, FILE *fp) {
    int num_m = bd->shapes[0].num_m;
    BENCH_ASSERT(num_m == NUM_M);

    fprintf(fp, "M");
    for (int i = 0; i < num_m; i++) {
        fprintf(fp, ";%2d", bd->shapes[0].items[i].M);
    }
    fprintf(fp, "\n");

    for (int i = 0; i < bd->n_shapes; i++) {
        struct bench_data_shape *s = &bd->shapes[i];
        fprintf(fp, "NxK=%dx%d", s->N, s->K);
        for (int j = 0; j < num_m; j++) {
            fprintf(fp, ";%6.3f", s->items[j].gpu_comp / 1000.0);
        }
        fprintf(fp, "\n");
    }
}

// print benchmark as binary file: for load by llama.
static void write_bench_data(struct bench_data *bd, FILE *fp) {
    fprintf(fp, "%s %d\n", bd->model, bd->n_shapes);

    for (int i = 0; i < bd->n_shapes; i++) {
        struct bench_data_shape *s = &bd->shapes[i];

        fprintf(fp, "%d %d %d %d\n", s->N, s->K, s->m_step, s->num_m);

        for (int j = 0; j < s->num_m; j++) {
            struct bench_data_item *item = &s->items[j];
            fprintf(fp, "%2d %6lld %6lld %6lld %6lld\n", item->M,
                    item->cpu_init, item->cpu_comp, item->gpu_init,
                    item->gpu_comp);
        }
    }
}

static void read_bench_data(struct bench_data *bd, FILE *fp) {
    fscanf(fp, "%s %d", bd->model, &bd->n_shapes);
    bd->shapes = malloc(sizeof(struct bench_data_shape) * bd->n_shapes);

    for (int i = 0; i < bd->n_shapes; i++) {
        struct bench_data_shape *s = &bd->shapes[i];

        fscanf(fp, "%d", &s->N);
        fscanf(fp, "%d", &s->K);
        fscanf(fp, "%d", &s->m_step);
        fscanf(fp, "%d", &s->num_m);

        s->items = malloc(sizeof(struct bench_data_item) * s->num_m);

        for (int j = 0; j < s->num_m; j++) {
            struct bench_data_item *item = &s->items[j];
            fscanf(fp, "%d", &item->M);
            fscanf(fp, "%lld", &item->cpu_init);
            fscanf(fp, "%lld", &item->cpu_comp);
            fscanf(fp, "%lld", &item->gpu_init);
            fscanf(fp, "%lld", &item->gpu_comp);
        }
    }
}

static void test_compare_bench_data(struct bench_data *bd,
                                    struct bench_data *bd2) {
    BENCH_ASSERT(bd->n_shapes == bd2->n_shapes);

    for (int i = 0; i < bd->n_shapes; i++) {
        struct bench_data_shape *s1 = &bd->shapes[i];
        struct bench_data_shape *s2 = &bd2->shapes[i];
        BENCH_ASSERT(s1->N == s2->N);
        BENCH_ASSERT(s1->K == s2->K);

        BENCH_ASSERT(s1->num_m == s2->num_m);

        for (int j = 0; j < s1->num_m; j++) {
            struct bench_data_item *item1 = &s1->items[j];
            struct bench_data_item *item2 = &s2->items[j];

            BENCH_ASSERT(item1->M == item2->M);
            BENCH_ASSERT(item1->cpu_init == item2->cpu_init);
            BENCH_ASSERT(item1->cpu_comp == item2->cpu_comp);
            BENCH_ASSERT(item1->gpu_init == item2->gpu_init);
            BENCH_ASSERT(item1->gpu_comp == item2->gpu_comp);
        }
    }
}

static void test_estimate_xpu_time(void) {
    struct bench_data bd;

    const char *data_file = "q4_0-mulmat-bench.7B.txt";

    FILE *fp = fopen(data_file, "r");
    BENCH_ASSERT(fp);

    read_bench_data(&bd, fp);
    fclose(fp);

    const double error_bound = 0.01;

    // These can be read from data file.
    const int nth = 1;
    const int N = bd.shapes[0].N;
    const int K = bd.shapes[0].K;
    const int num_m = bd.shapes[0].num_m;
    const int m_step = bd.shapes[0].m_step;

    BENCH_ASSERT(num_m > 2);
    BENCH_ASSERT(m_step % 2 == 0);

    const int m_max = m_step * num_m;

    const int num_tests = 4;

    const int Ms[num_tests] = {m_step, m_step + m_step / 2,
                               m_step * 2, m_max + 1};

    int64_t T[num_tests];

    for (int i = 0; i < 2; i++) {
        printf("\n====== test: estimate %s time\n", i == 0 ? "CPU" : "GPU");

        for (int j = 0; j < num_tests; j++) {
            int M = Ms[j];
            T[j] = (i == 0) ? estimate_time(&bd, M, N, K, nth, true)
                            : estimate_time(&bd, M, N, K, nth, false);
            printf("M: %2d, N: %5d, K: %5d, nth: %d, time: %6lld\n", M, N, K,
                   nth, T[j]);
        }

        int64_t sum = T[0] + T[2];
        double diff = sum - 2 * T[1];
        if (diff < 0) {
            diff = -diff;
        }
        BENCH_ASSERT((diff / sum) < error_bound);
        BENCH_ASSERT(T[3] == -1);
    }
}
