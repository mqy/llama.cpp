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
#define M_MAX 96
#define M_STEP 8

#define NUM_M (M_MAX / M_STEP + 1)

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

struct bench_nk {
    int32_t N;
    int32_t K;

    int64_t cpu_init[NUM_BENCH];
    int64_t cpu_comp[NUM_BENCH];

    int64_t gpu_init[NUM_BENCH];
    int64_t gpu_comp[NUM_BENCH];
};

struct bench_m {
    int32_t M;
    struct bench_nk bnk[NUM_NK_PAIRS];
};

struct bench {
    struct bench_m m[NUM_M];
};

// TODO: remove bench, bench_m, bench_nk
struct bench_dat_stat {
    int64_t cpu_init[NUM_BENCH];
    int64_t cpu_comp[NUM_BENCH];

    int64_t gpu_init[NUM_BENCH];
    int64_t gpu_comp[NUM_BENCH];
};

// sub data record to write/read to/from file.
struct bench_dat_item {
    int32_t M;

    // avg.
    int64_t cpu_init;
    int64_t cpu_comp;
    int64_t gpu_init;
    int64_t gpu_comp;

    struct bench_dat_stat stat;
};

// sub data record to write/read to/from file.
struct bench_dat_shape {
    int32_t N;
    int32_t K;

    int32_t n_items;
    struct bench_dat_item *items;
};

// top bench data to write/read to/from file.
struct bench_dat {
    int32_t n_shapes;
    struct bench_dat_shape *shapes;
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
static int64_t get_bench_time(int64_t *a, int len);
static void write_benches_csv(struct bench_dat *bd, FILE *file);
static void write_benches_txt(struct bench_dat *bd, FILE *file);
static void write_benches_dat(struct bench_dat *bd, FILE *file);
static void load_benches_dat(struct bench_dat *bd, FILE *file);
static int64_t estimate_cpu_time(struct bench_dat *bd, int M, int N, int K,
                                 int nth);
static int64_t estimate_gpu_time_blas(struct bench_dat *bd, int M, int N,
                                          int K, int nth);
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

    test_estimate_xpu_time();
    exit(0);

    if (argc != 2) {
        usage(argv[0]);
        exit(1);
    }

    const struct nk_pair *nk_pairs = NULL;

    char *model = argv[1];

    if (strcmp(model, "7B") == 0 || strcmp(model, "7b") == 0) {
        nk_pairs = nk_pairs_7b;
    } else if (strcmp(model, "13B") == 0 || strcmp(model, "13b") == 0) {
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

    const float alpha = 1.0f;
    const float beta = 0.0f;

    struct bench bench;

    int32_t M;
    for (int im = 0; im < NUM_M; im++) {
        if (im > 0) {
            M = M_STEP * im;
        } else {
            M = 1;
        }

        struct bench_m *bm = &bench.m[im];
        bm->M = M;

        for (int i = 0; i < NUM_NK_PAIRS; i++) {
            int N = nk_pairs[i].N;
            int K = nk_pairs[i].K;

            const int lda = K;
            const int ldb = K;
            const int ldc = N;

            struct bench_nk *bnk = &bm->bnk[i];
            bnk->N = N;
            bnk->K = K;

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

            size_t sizeC = M_MAX * N * sizeof(float);
            void *C = malloc(sizeC);
            memset(C, 0, sizeC);

            // cpu init (very fast: several us)
            for (int nb = 0; nb < NUM_BENCH; nb++) {
                int64_t t0 = time_us();
                for (int64_t m = 0; m < M; m++) {
                    quantize_row_q((float *)((char *)src1->data + m * K),
                                   (char *)wdata + m * src1->nb[1], K);
                }
                bnk->cpu_init[nb] = time_us() - t0;
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
                bnk->cpu_comp[nb] = time_us() - t0;
            }

            // gpu init (support multi-threads)
            for (int nb = 0; nb < NUM_BENCH; nb++) {
                int64_t t0 = time_us();
                for (int n = 0; n < N; n++) {
                    dequantize_row_q((const float *)src0->data + n * K,
                                     (float *)wdata + n * K, K);
                }
                bnk->gpu_init[nb] = time_us() - t0;
            }

            const float *A = (float *)src1->data;
            const float *B = (float *)wdata;

            // gpu comp (single thread).
            for (int nb = 0; nb < NUM_BENCH; nb++) {
                int64_t t0 = time_us();
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K,
                            alpha, A, lda, B, ldb, beta, (float *)C, ldc);
                bnk->gpu_comp[nb] = time_us() - t0;
            }

            ggml_free(ctx);
            free(C);
        }
    }

    free(wdata);
    free(q4_0_buf);

    struct bench_dat bd = {.n_shapes = NUM_NK_PAIRS, .shapes = NULL};
    size_t sz = sizeof(struct bench_dat_shape) * NUM_NK_PAIRS;
    bd.shapes = malloc(sz);
    memset(bd.shapes, 0, sz);

    // convert `bench` to `bd`.
    for (int i = 0; i < NUM_NK_PAIRS; i++) {
        bd.shapes[i].N = bench.m[0].bnk[i].N;
        bd.shapes[i].K = bench.m[0].bnk[i].K;

        bd.shapes[i].n_items = NUM_M;
        sz = sizeof(struct bench_dat_item) * NUM_M;
        bd.shapes[i].items = malloc(sz);
        memset(bd.shapes[i].items, 0, sz);

        for (int j = 0; j < NUM_M; j++) {
            struct bench_dat_item *item = &bd.shapes[i].items[j];
            struct bench_nk *bnk = &bench.m[j].bnk[i];

            item->M = bench.m[j].M;
            item->cpu_init = get_bench_time(bnk->cpu_init, NUM_BENCH);
            item->cpu_comp = get_bench_time(bnk->cpu_comp, NUM_BENCH);
            item->gpu_init = get_bench_time(bnk->gpu_init, NUM_BENCH);
            item->gpu_comp = get_bench_time(bnk->gpu_comp, NUM_BENCH);
        }
    }

    // TODO: free bd and bd2.

    printf("\n====== print as csv: \n\n");

    write_benches_csv(&bd, stdout);

    printf("\n====== print as txt: \n\n");
    write_benches_txt(&bd, stdout);

    char bin_file[10];
    snprintf(bin_file, sizeof(bin_file), "%s.dat", model);

    printf("\n====== write to bin file: %s", bin_file);
    FILE *bin_fp = fopen(bin_file, "w");
    write_benches_dat(&bd, bin_fp);
    fclose(bin_fp);

    printf("\n====== read from bin file: %s", bin_file);
    bin_fp = fopen(bin_file, "r");
    struct bench_dat bd2;
    load_benches_dat(&bd2, bin_fp);

    printf("\n====== print bin dat as txt: \n\n");
    write_benches_txt(&bd2, stdout);
    fclose(bin_fp);

    // compare bd and bd2
    {
        BENCH_ASSERT(bd.n_shapes == bd2.n_shapes);

        for (int i = 0; i < bd.n_shapes; i++) {
            struct bench_dat_shape *s1 = &bd.shapes[i];
            struct bench_dat_shape *s2 = &bd2.shapes[i];
            BENCH_ASSERT(s1->N == s2->N);
            BENCH_ASSERT(s1->K == s2->K);

            BENCH_ASSERT(s1->n_items == s2->n_items);

            for (int j = 0; j < s1->n_items; j++) {
                struct bench_dat_item *item1 = &s1->items[j];
                struct bench_dat_item *item2 = &s2->items[j];

                BENCH_ASSERT(item1->M == item2->M);
                BENCH_ASSERT(item1->cpu_init == item2->cpu_init);
                BENCH_ASSERT(item1->cpu_comp == item2->cpu_comp);
                BENCH_ASSERT(item1->gpu_init == item2->gpu_init);
                BENCH_ASSERT(item1->gpu_comp == item2->gpu_comp);
            }
        }
    }

    return 0;
}

static void test_estimate_xpu_time(void) {
    struct bench_dat bd;

    const char *data_file = "7b.dat";

    FILE *fp = fopen(data_file, "r");
    BENCH_ASSERT(fp);

    load_benches_dat(&bd, fp);
    fclose(fp);

    printf("\n====== estimate CPU time\n");
    printf("M:  8, N, 4096, K: 4096, nth: 1, time: %lld\n",
           estimate_cpu_time(&bd, 8, 4096, 4096, 1));
    printf("M: 16, N, 4096, K: 4096, nth: 1: %lld\n",
           estimate_cpu_time(&bd, 16, 4096, 4096, 1));
    printf("M: 12, N, 4096, K: 4096, nth: 1: %lld\n",
           estimate_cpu_time(&bd, 12, 4096, 4096, 1));

    printf("\n====== estimate GPU time\n");

    printf("M:  8, N, 4096, K: 4096, nth: 1, time: %lld\n",
           estimate_gpu_time_blas(&bd, 8, 4096, 4096, 1));
    printf("M: 16, N, 4096, K: 4096, nth: 1: %lld\n",
           estimate_gpu_time_blas(&bd, 16, 4096, 4096, 1));
    printf("M: 12, N, 4096, K: 4096, nth: 1: %lld\n",
           estimate_gpu_time_blas(&bd, 12, 4096, 4096, 1));
}

// for given work load and number of threads, estimate cpu time.
static int64_t estimate_cpu_time(struct bench_dat *bd, int M, int N, int K,
                                 int nth) {
    struct bench_dat_shape *shape = NULL;
    for (int i = 0; i < bd->n_shapes; i++) {
        if (bd->shapes[i].N == N && bd->shapes[i].K == K) {
            shape = &bd->shapes[i];
            break;
        }
    }

    if (shape == NULL) {
        return -1;
    }

    for (int i = 0; i < shape->n_items; i++) {
        struct bench_dat_item *item = &shape->items[i];
        if (item->M == M) {
            return item->cpu_init + item->cpu_comp / nth;
        }
    }

    for (int i = 0; i < shape->n_items - 1; i++) {
        struct bench_dat_item *prev = &shape->items[i];
        struct bench_dat_item *next = &shape->items[i + 1];
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

    return -1;
}

// for given work load and number of threads, estimate gpu time.
static int64_t estimate_gpu_time_blas(struct bench_dat *bd, int M, int N,
                                          int K, int nth) {
    struct bench_dat_shape *shape = NULL;
    for (int i = 0; i < bd->n_shapes; i++) {
        if (bd->shapes[i].N == N && bd->shapes[i].K == K) {
            shape = &bd->shapes[i];
            break;
        }
    }

    if (shape == NULL) {
        // abort();
        return -1;
    }

    for (int i = 0; i < shape->n_items; i++) {
        struct bench_dat_item *item = &shape->items[i];
        if (item->M == M) {
            return item->gpu_init / nth + item->gpu_comp;
        }
    }

    for (int i = 0; i < shape->n_items - 1; i++) {
        struct bench_dat_item *prev = &shape->items[i];
        struct bench_dat_item *next = &shape->items[i + 1];

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

    return -1;
}

static int64_t time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}

static int64_t get_bench_time(int64_t *a, int len) {
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
static void write_benches_csv(struct bench_dat *bd, FILE *fp) {
    int n_items = bd->shapes[0].n_items;
    BENCH_ASSERT(n_items == NUM_M);

    fprintf(fp, "M");
    for (int i = 0; i < n_items; i++) {
        fprintf(fp, ";%3d", bd->shapes[0].items[i].M);
    }
    fprintf(fp, "\n");

    for (int i = 0; i < bd->n_shapes; i++) {
        struct bench_dat_shape *s = &bd->shapes[i];
        fprintf(fp, "NxK=%dx%d", s->N, s->K);
        for (int j = 0; j < n_items; j++) {
            fprintf(fp, ";%6.3f", s->items[j].gpu_comp / 1000.0);
        }
        fprintf(fp, "\n");
    }
}

// print benchmark as binary file: for load by llama.
static void write_benches_txt(struct bench_dat *bd, FILE *fp) {
    fprintf(fp, "%d\n", bd->n_shapes);

    for (int i = 0; i < bd->n_shapes; i++) {
        struct bench_dat_shape *s = &bd->shapes[i];

        fprintf(fp, "%5d %d\n", s->N, s->K);
        fprintf(fp, "%d\n", s->n_items);

        for (int j = 0; j < s->n_items; j++) {
            struct bench_dat_item *item = &s->items[j];
            fprintf(fp, "%3d %6lld %6lld %6lld %6lld\n", item->M,
                    item->cpu_init, item->cpu_comp, item->gpu_init,
                    item->gpu_comp);
        }
    }
}

static void write_benches_dat(struct bench_dat *bd, FILE *fp) {
    fwrite(&bd->n_shapes, sizeof(int32_t), 1, fp);

    for (int i = 0; i < bd->n_shapes; i++) {
        struct bench_dat_shape *s = &bd->shapes[i];

        fwrite(&s->N, sizeof(int32_t), 1, fp);
        fwrite(&s->K, sizeof(int32_t), 1, fp);

        fwrite(&s->n_items, sizeof(int32_t), 1, fp);

        for (int j = 0; j < s->n_items; j++) {
            struct bench_dat_item *item = &s->items[j];
            fwrite(&item->M, sizeof(int32_t), 1, fp);
            fwrite(&item->cpu_init, sizeof(int64_t), 1, fp);
            fwrite(&item->cpu_comp, sizeof(int64_t), 1, fp);
            fwrite(&item->gpu_init, sizeof(int64_t), 1, fp);
            fwrite(&item->gpu_comp, sizeof(int64_t), 1, fp);
        }
    }
}

static void load_benches_dat(struct bench_dat *bd, FILE *fp) {
    fread(&bd->n_shapes, sizeof(int32_t), 1, fp);
    bd->shapes = malloc(sizeof(struct bench_dat_shape) * bd->n_shapes);

    for (int i = 0; i < bd->n_shapes; i++) {
        struct bench_dat_shape *s = &bd->shapes[i];

        fread(&s->N, sizeof(int32_t), 1, fp);
        fread(&s->K, sizeof(int32_t), 1, fp);
        fread(&s->n_items, sizeof(int32_t), 1, fp);

        s->items = malloc(sizeof(struct bench_dat_item) * s->n_items);

        for (int j = 0; j < s->n_items; j++) {
            struct bench_dat_item *item = &s->items[j];
            fread(&item->M, sizeof(int32_t), 1, fp);
            fread(&item->cpu_init, sizeof(int64_t), 1, fp);
            fread(&item->cpu_comp, sizeof(int64_t), 1, fp);
            fread(&item->gpu_init, sizeof(int64_t), 1, fp);
            fread(&item->gpu_comp, sizeof(int64_t), 1, fp);
        }
    }
}
