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

struct NK {
  int N;
  int K;
};

void bubble_sort(int64_t * a, int len) {
  for (int i = 0; i < len-1; i++) {
    for (int j = i+1; j < len; j++) {
      if (a[j] < a[i]) {
        int64_t temp = a[j];
        a[j] = a[i];
        a[i] = temp;
      }
    }
  }
}

double avg_ms(int64_t *a, int len) {
  if (len < 4) {
    abort();
  }
  
  bubble_sort(a, len);

  int64_t total = 0;
  for (int i = 2; i < len - 2; i++) { // kickoff first two and last two
    total += a[i];
  }
  return (total / (len - 4)) / 1000.0;
}

double min_ms(int64_t *a, int len) {
  int64_t min = INT64_MAX;
  for (int i = 0; i < len; i++) { // kickoff first two and last two
    if (a[i] < min) {
      min = a[i];
    }
  }
  return min / 1000.0;
}

int main(void) {
#if !(defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS))
  fprintf(stderr, "not defined: GGML_USE_ACCELERATE or GGML_USE_OPENBLAS\n");
  abort();
#endif

  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int n_pairs = 6;

  struct NK pairs[n_pairs]  = {
    /* 7B */
    {4096, 4096},
    {4096, 11008},
    {11008, 4096},
    /* 13B */
    {5120, 5120},
    {5120, 13824},
    {13824, 5120}
  };

  quantize_fns_t funcs = ggml_internal_get_quantize_fn(GGML_TYPE_Q4_0);
  dequantize_row_q_t dequantize_row_q = funcs.dequantize_row_q;
  quantize_row_q_t quantize_row_q = funcs.quantize_row_q;
  vec_dot_q_t vec_dot_q = funcs.vec_dot_q;

  size_t maxNxK = 13824 * 5120;
  void * q4_0_buf = malloc(maxNxK * sizeof(int64_t));
  size_t wdata_size = maxNxK * sizeof(float);
  void * wdata = malloc(wdata_size);

  const int maxM = 128;
  const int stepM = 8;
  int M;

  printf("M;");
  for (int i=0; i<n_pairs; i++) {
    if (i > 0) {
      printf(";");
    }
    printf("NxK=%dx%d", pairs[i].N, pairs[i].K);
  }
  printf("\n");
  
  for (int im = 0; im <= maxM/stepM; im++) {
    if (im > 0) {
      M = stepM * im;
    } else {
      M = 1;
    }

    double gpu_comp_time[n_pairs];
  
    for (int i = 0; i < n_pairs; i++) {
      int N = pairs[i].N;
      int K = pairs[i].K;

      const int lda = K;
      const int ldb = K;
      const int ldc = N;

      size_t ctx_size = K * N * ggml_type_sizef(GGML_TYPE_F32) +  K * sizeof(float) + 1024 * 1024 * 300;

      struct ggml_init_params params = {
          .mem_size = ctx_size,
          .mem_buffer = NULL,
          .no_alloc = 0,
      };

      //printf("M, nth,   cpu_i,   cpu_c,   gpu_i,   gpu_c,  cpu_all,  gpu_all,  cpu/gpu\n");

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

      // src1: M x K
      struct ggml_tensor *src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
      ggml_set_f32(src1, 0.5f);

      const int n_bench = 5; // must >= 5

      int64_t cpu_init_arr[n_bench], cpu_comp_arr[n_bench];
      int64_t gpu_init_arr[n_bench], gpu_comp_arr[n_bench];

      memset(wdata, 0, wdata_size);
      // cpu init (very fast: several us)
      for (int nb = 0; nb < n_bench; nb++) {
        int64_t t0 = time_us();
        for (int64_t m = 0; m < M; m++) {
            quantize_row_q((float *)((char *) src1->data + m*K), (char *)wdata + m * src1->nb[1], K);
        }
        cpu_init_arr[nb] = time_us() - t0;
      }

      size_t sizeC = maxM * N * sizeof(float);
      void * C = malloc(sizeC);
      memset(C, 0, sizeC);

      memset(wdata, 0, wdata_size);

      // cpu comp (support multi-threads)
      for (int nb = 0; nb < n_bench; nb++) {
        int64_t t0 = time_us();
        for (int m = 0; m < M; m++) {
          float * src0_row = (float *)src0->data + m * N;
          for (int n = 0; n < N; n++) {
            vec_dot_q(K, (float *)C + m*N, src0_row, (float *)wdata + m*K + n);
          }
        }
        cpu_comp_arr[nb] = time_us() - t0;
      }

      // gpu init (support multi-threads)
      for (int nb = 0; nb < n_bench; nb++) {
        int64_t t0 = time_us();
        for (int n = 0; n < N; n++) {
          dequantize_row_q((const float *)src0->data + n*K, (float *)wdata + n*K, K);
        }
        gpu_init_arr[nb] = time_us() - t0;
      }

      const float * A = (float *)src1->data;
      const float * B = (float *)wdata;

      // gpu comp (single thread).
      for (int nb = 0; nb < n_bench; nb++) {
        int64_t t0 = time_us();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha,
            A, lda, B, ldb, beta, (float *)C, ldc);
        gpu_comp_arr[nb] = time_us() - t0;
      }

#if defined(GGML_MUL_MAT_BENCH_AVG)
      // double cpu_init_ms = avg_ms(cpu_init_arr, n_bench);
      // double cpu_comp_ms = avg_ms(cpu_comp_arr, n_bench); 
      // double gpu_init_ms = avg_ms(gpu_init_arr, n_bench);
      // double gpu_comp_ms = avg_ms(gpu_comp_arr, n_bench);
#else
      double cpu_init_ms = min_ms(cpu_init_arr, n_bench);
      double cpu_comp_ms = min_ms(cpu_comp_arr, n_bench); 
      double gpu_init_ms = min_ms(gpu_init_arr, n_bench);
      double gpu_comp_ms = min_ms(gpu_comp_arr, n_bench);
#endif

      gpu_comp_time[i] = gpu_comp_ms;

      double cpu_init_row = cpu_init_ms / M;
      double cpu_comp_row = cpu_comp_ms / M;

      //for (int n = 0; n <= 4; n++) {
        //int n_threads = n == 0? 1 : n * 2;
        int n_threads = 1;
        double cpu_comp_ith = cpu_comp_ms / n_threads;
        double gpu_init_ith = gpu_init_ms / n_threads;
        double cpu_total = cpu_init_ms + cpu_comp_ith;
        double gpu_total = gpu_init_ith + gpu_comp_ms;
        double cpu_to_gpu = cpu_total / gpu_total;

      //   printf("%4d, %5d, %5d, %7.3f, %7.3f, %7.3f, %7.3f, %7.3f, %7.3f, %7.3f\n",
      //   M, N, K, n_threads, cpu_init_ms, cpu_comp_ith, gpu_init_ith, gpu_comp_ms, cpu_total, gpu_total, cpu_to_gpu);
      // //}
      // printf("\tcpu_init_row: %7.3f, cpu_comp_row: %7.3f\n", cpu_init_row, cpu_comp_row);
      // printf("\tgpu_init:     %7.3f, cpu_comp:     %7.3f\n", gpu_init_ms, gpu_comp_ms);

      ggml_free(ctx);
      free(C);
    }

    printf("%d;", M);
    for (int i = 0; i < n_pairs; i++) {
      if (i > 0) {
        printf(";");
      }
      printf("%.3f", gpu_comp_time[i]);
    }
    printf("\n");
  }

  free(wdata);
  free(q4_0_buf);
  
  return 0;
}

// // TODO: lookup table.
// static double cpu_init_nk(int N, int K) {
//   return 0;
// }

// // TODO: lookup table.
// // constant?
// static double cpu_comp_nk(int N, int K) {
//   return 0;
// }

// // TODO: lookup table.
// // constant?
// static double gpu_init_nk(int N, int K) {
//   return 0;
// }

// // TODO: lookup table.
// // y = ax + b
// static double gpu_comp_mnk(int M, int N, int K) {
//   return 0;
// }

// // cpu: Init_NK * M + Compute_NK * M / nth
// static double cpu_time(int M, int N, int K, int nth) {
//   double a = cpu_init_nk(N, K);
//   double b = cpu_comp_nk(N, K);
//   return a * M + b * M / nth;
// }

// // gpu: Init_NK * M / nth + Compute_MNK + WaitNotifyDelay
// static double gpu_time(int M, int N, int K, int nth) {
//   double a = gpu_init_nk(N, K);
//   double b = gpu_comp_mnk(M, N, K);
//   return a * M / nth + b;
// }
