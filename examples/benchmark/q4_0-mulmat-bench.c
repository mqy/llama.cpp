#include "ggml.h"

#define GGML_USE_ACCELERATE 1
//#define GGML_USE_OPENBLAS

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(GGML_USE_OPENBLAS)
#include <cblas.h>
#endif

int64_t time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}

int main(void) {
#if !(defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS))
  fprintf(stderr, "not defined: GGML_USE_ACCELERATE or GGML_USE_OPENBLAS\n");
  abort();
#endif

  const int max_threads = 4;
  const int max_M = 128;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int arr_len = 6;
  int arrN[arr_len] = { 5120, 13824, 5120, 4096, 4096, 11008};
  int arrK[arr_len] = { 5120, 5120, 13824, 4096, 11008, 4096};

  quantize_fns_t funcs = ggml_internal_get_quantize_fn(GGML_TYPE_Q4_0);
  dequantize_row_q_t dequantize_row_q = funcs.dequantize_row_q;
  vec_dot_q_t vec_dot_q = funcs.vec_dot_q;

  for (int i = 0; i < arr_len; i++) {
    int N = arrN[i];
    int K = arrK[i];

    void * q4_0_buf = malloc(N*K*sizeof(int64_t));
    void * wdata = malloc(K * N * sizeof(float));

    size_t ctx_size = K * N * ggml_type_sizef(GGML_TYPE_F32) +  K * sizeof(float) + 1024 * 1024 * 300;

    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = 0,
    };

    // for (int M = 1; M <= max_M; M++) {
    for (int M = 1; M <= max_M; M *= 2) {
      struct ggml_context *ctx = ggml_init(params);
      if (!ctx) {
        fprintf(stderr, "Error: ggml_init() returned empty ctx\n");
        return -1;
      }

      // src0: K x N
      struct ggml_tensor *src0_f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
      ggml_set_f32(src0_f32, 0.1f);

      struct ggml_tensor *src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
      ggml_quantize_q4_0((const float *)src0_f32->data, src0->data, N, K, (int64_t *)q4_0_buf);

      //src1: M x K
      struct ggml_tensor *src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
      ggml_set_f32(src1, 0.5f);

      int64_t t0 = time_us();

      // This is t_blas_init(N, K)
      // NOTE: first run takes about 4~6x of the avg time !!!!
      for (int n = 0; n < N; n++) {
        dequantize_row_q((float *)src0->data + n*K, (float *)wdata + n*K, K);
      }

      int64_t t1 = time_us();

      const int lda = K;
      const int ldb = K;
      const int ldc = N;

      const float * A = (float *)src1->data;
      const float * B = (float *)wdata;
      void * C = malloc(M * N * sizeof(float));

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha,
          A, lda, B, ldb, beta, (float *)C, ldc);

      int64_t t2 = time_us();

      // This is M * t_cpu_compute(N, K)
      for (int m = 0; m < M; m++) {
        float * src0_row = (float *)src0->data + m * N;
        for (int n = 0; n < N; n++) {
          vec_dot_q(K, (float *)C + m*N, src0_row, (float *)wdata + m * K + n);
        }
      }

      int64_t t3 = time_us();

      double blas_init_ms = (t1-t0) / 1000.0;
      double blas_compute_ms = (t2-t1) / 1000.0;
      double cpu_compute_ms = (t3-t2) / 1000.0;

      printf("M: %4d, N: %5d, K: %5d, blas_init: %7.3f ms, blas_compute: %7.3f ms, cpu_compute: %7.3f ms\n",
        M, N, K, blas_init_ms, blas_compute_ms, cpu_compute_ms);

      for (int n = 1; n <= max_threads; n++) {
        double cpu_compute_avg = cpu_compute_ms / n;
        double blas_init_avg = blas_init_ms / n;
        
        double cpu_gpu = cpu_compute_avg / (blas_init_avg + blas_compute_ms);
        if (blas_init_avg + blas_compute_ms > cpu_compute_avg) {
          printf("\t%d thread: cPU win! cPU/gPU: %4.2f\n", n, cpu_gpu);
        } else {
          printf("\t%d thread: gPU win! cPU/gPU: %4.2f\n", n, cpu_gpu);
        }
      }

      free(C);
      ggml_free(ctx);
    }

    free(wdata);
    free(q4_0_buf);
  }

  return 0;
}
