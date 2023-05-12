// #define GGML_USE_ACCELERATE 1
#define GGML_USE_OPENBLAS

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(GGML_USE_OPENBLAS)
#include <cblas.h>
#endif

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int64_t time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}

int64_t time_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000 + (int64_t)ts.tv_nsec;
}


void print_array(const float *A, int m, int n) {
  for (int i = 0; i < m; i++) {
    if (i > 0) {
      printf("\n");
    }
    for (int j = 0; j < n; j++) {
      printf("%4.0f\t", *(A + n * i + j));
    }
  }
  printf("\n");
}

// B: no trans
void test_mul_mat_1() {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int M = 3;
  const int N = 2;
  const int K = 4;

  float A[M][K];
  float B[K][N];
  float C[M][N];

  // A: row major: K x M blocks
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i][j] = 1 + i * K + j;
    }
  }

  // B: row major: N x K blocks
  for (int i = 0; i < K; i++) {
    for (int j = 0; j < N; j++) {
      B[i][j] = 1 + i * N + j;
    }
  }

  // Expected C
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int k = 0; k < K; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }

  printf("A:\n");
  print_array((float *)A, M, K);
  printf("\n");

  printf("B:\n");
  print_array((float *)B, K, N);
  printf("\n");

  printf("Expected C:\n");
  print_array((float *)C, M, N);
  printf("\n");

  // Reset C
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i][j] = 0;
    }
  }

  // C ← αAB + βC
  {
    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
                (const float *)A, lda, (const float *)B, ldb, beta, (float *)C,
                ldc);

    printf("B no trans, C:\n");
    print_array((float *)C, M, N);
    printf("\n");
  }
}

// B: trans
void test_mul_mat_2() {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int M = 3;
  const int N = 2;
  const int K = 4;

  float A[M][K];
  float B[N][K]; // NOTE: N x K (not K x N). Thus cblas_sgemm need CblasTrans.
  float C[M][N];

  // A: row major: K x M blocks
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i][j] = 1 + i * K + j;
    }
  }

  // B: row major: K x N blocks
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      B[i][j] = 1 + i * K + j;
    }
  }

  // C = A x B(T)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int k = 0; k < K; k++) {
        sum += A[i][k] * B[j][k];
      }
      C[i][j] = sum;
    }
  }

  printf("A:\n");
  print_array((float *)A, M, K);
  printf("\n");

  printf("B:\n");
  print_array((float *)B, N, K);
  printf("\n");

  printf("Expected C:\n");
  print_array((float *)C, M, N);
  printf("\n");

  // Reset C
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i][j] = 0;
    }
  }

  // C ← αAB + βC
  {
    const int lda = K;
    const int ldb = K;
    const int ldc = N;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha,
                (const float *)A, lda, (const float *)B, ldb, beta, (float *)C,
                ldc);

    printf("B trans, C:\n");
    print_array((float *)C, M, N);
    printf("\n");
  }
}

// B: trans, A x B[i]
void test_mul_mat_3() {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int M = 3;
  const int N = 2;
  const int K = 4;

  float A[M][K];
  float B[N][K]; // NOTE: N x K (not K x N). Thus cblas_sgemm need CblasTrans.
  float C[M][N];

  // A: row major: K x M blocks
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i][j] = 1 + i * K + j;
    }
  }

  // B: row major: K x N blocks
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      B[i][j] = 1 + i * K + j;
    }
  }

  // C = A x B(T)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int k = 0; k < K; k++) {
        sum += A[i][k] * B[j][k];
      }
      C[i][j] = sum;
    }
  }

  printf("A:\n");
  print_array((float *)A, M, K);
  printf("\n");

  printf("B:\n");
  print_array((float *)B, N, K);
  printf("\n");

  printf("Expected C:\n");
  print_array((float *)C, M, N);
  printf("\n");

  // Reset C
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i][j] = 0;
    }
  }

  // C ← αAB + βC
  {
    const int lda = K;
    const int ldb = K;
    const int ldc = 1;

    float *pa = (float *)A;
    for (int i = 0; i < N; i++) {
      float *pb = (float *)B + i * K;
      float *pc = (float *)C + i * M;

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, 1, K, alpha, pa,
                  lda, pb, ldb, beta, pc, ldc);
    }

    printf("before transpose, memory layout of C:\n");
    print_array((float *)C, N, M);
    printf("\n");

    float *cp = (float *)C;
    float C_trans[M][N];
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < M; j++) {
        C_trans[j][i] = *(cp + i * M + j);
      }
    }

    printf("C transposed:\n");
    print_array((float *)C_trans, M, N);
    printf("\n");
  }
}

// B: trans, A[i] x B
void test_mul_mat_4() {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int M = 3;
  const int N = 2;
  const int K = 4;

  float A[M][K];
  float B[N][K]; // NOTE: N x K (not K x N). Thus cblas_sgemm need CblasTrans.
  float C[M][N];

  // A: row major: K x M blocks
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i][j] = 1 + i * K + j;
    }
  }

  // B: row major: K x N blocks
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      B[i][j] = 1 + i * K + j;
    }
  }

  // C = A x B(T)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int k = 0; k < K; k++) {
        sum += A[i][k] * B[j][k];
      }
      C[i][j] = sum;
    }
  }

  printf("A:\n");
  print_array((float *)A, M, K);
  printf("\n");

  printf("B:\n");
  print_array((float *)B, N, K);
  printf("\n");

  printf("Expected C:\n");
  print_array((float *)C, M, N);
  printf("\n");

  // Reset C
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i][j] = 0;
    }
  }

  // C ← αAB + βC
  {
    const int lda = K;
    const int ldb = K;
    const int ldc = N;

    float *pb = (float *)B;
    for (int i = 0; i < M; i++) {
      float *pa = (float *)A + i * K;
      float *pc = (float *)C + i * N;

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, N, K, alpha, pa,
                  lda, pb, ldb, beta, pc, ldc);
    }

    printf("calculated C:\n");
    print_array((float *)C, M, N);
    printf("\n");
  }
}

struct mul_mat_data_t {
  int M;
  int N;
  int K;
  float *A;
  float *B;
  float *C;

  int ith;
  int nth;
};

void *mul_mat_runner(void *arg) {
  struct mul_mat_data_t *v = (struct mul_mat_data_t *)arg;

  if (v->M % v->nth != 0) {
    printf("== bad nth: %d==", v->nth);
    abort();
  }

  const int lda = v->K;
  const int ldb = v->K;
  const int ldc = v->N;

  float *pb = v->B;

  int avg_rows = v->M / v->nth; // avg block size.

  int start = avg_rows * v->ith;
  int end = start + avg_rows - 1;
  if (v->ith == v->nth - 1 && end != v->M - 1) {
    end = v->M - 1;
  }

  int rows = end - start + 1;

  printf("M: %d, %d-th, avg_rows: %d, start: %d, end: %d, rows: %d\n", v->M,
         v->ith, avg_rows, start, end, rows);

  int previous_rows = v->ith * avg_rows;
  float *pa = v->A + previous_rows * v->K;
  float *pc = v->C + previous_rows * v->N;

  int64_t t0 = time_us();
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, v->N, v->K, 1.0f,
              pa, lda, pb, ldb, 0.0f, pc, ldc);
  int64_t t1 = time_us();
  printf("%d-th: dur: %lld us\n", v->ith, t1 - t0);

  return NULL;
}

// B: trans, A[i] x B
// verify parallel with multi-threads.
void test_mul_mat_5() {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int M = 3;
  const int N = 2;
  const int K = 4;

  float A[M][K];
  float B[N][K]; // NOTE: N x K (not K x N). Thus cblas_sgemm need CblasTrans.
  float C[M][N];

  // A: row major: K x M blocks
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      A[i][j] = 1 + i * K + j;
    }
  }

  // B: row major: K x N blocks
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      B[i][j] = 1 + i * K + j;
    }
  }

  // C = A x B(T)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int k = 0; k < K; k++) {
        sum += A[i][k] * B[j][k];
      }
      C[i][j] = sum;
    }
  }

  printf("A:\n");
  print_array((float *)A, M, K);
  printf("\n");

  printf("B:\n");
  print_array((float *)B, N, K);
  printf("\n");

  printf("Expected C:\n");
  print_array((float *)C, M, N);
  printf("\n");

  // Reset C
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i][j] = 0;
    }
  }

  // C ← αAB + βC
  {
    const int n_threads = M;
    pthread_t pids[n_threads];
    struct mul_mat_data_t args[n_threads];

    for (int i = 0; i < n_threads; i++) {
      args[i] = (struct mul_mat_data_t){
          .M = M,
          .N = N,
          .K = K,
          .A = (float *)A,
          .B = (float *)B,
          .C = (float *)C,
          .ith = i,
          .nth = n_threads,
      };
    }

    for (int i = 0; i < n_threads; i++) {
      pthread_create(&pids[i], NULL, mul_mat_runner, &args[i]);
    }

    for (int i = 0; i < n_threads; i++) {
      pthread_join(pids[i], NULL);
    }

    printf("calculated C with multi-threads:\n");
    print_array((float *)C, M, N);
    printf("\n");
  }
}

// B: trans, A[i] x B
// perf huge array.
void test_mul_mat_6() {
  int n_threads = 1;

  const int M = 4;
  const int N = 11008;
  const int K = 4096;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  int64_t sizeA = sizeof(float) * M * K;
  int64_t sizeB = sizeof(float) * N * K;
  int64_t sizeC = sizeof(float) * M * N;

  float *A = malloc(sizeA);
  float *B = malloc(sizeB);
  float *C = malloc(sizeC);

  memset(A, 0, sizeA);
  memset(B, 0, sizeB);
  memset(C, 0, sizeC);

  // A: row major: K x M blocks
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      *(A + i * K + j) = 1 + i * K + j;
    }
  }

  // B: row major: K x N blocks
  // NOTE: N x K (not K x N). Thus cblas_sgemm need CblasTrans.
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < K; j++) {
      *(B + i * K + j) = 1 + i * K + j;
    }
  }

  // C ← αAB + βC
  {
    pthread_t pids[n_threads];
    struct mul_mat_data_t args[n_threads];

    for (int i = 0; i < n_threads; i++) {
      args[i] = (struct mul_mat_data_t){
          .M = M,
          .N = N,
          .K = K,
          .A = (float *)A,
          .B = (float *)B,
          .C = (float *)C,
          .ith = i,
          .nth = n_threads,
      };
    }

    int64_t t0 = time_us();
    for (int i = 0; i < n_threads; i++) {
      pthread_create(&pids[i], NULL, mul_mat_runner, &args[i]);
    }

    for (int i = 0; i < n_threads; i++) {
      pthread_join(pids[i], NULL);
    }

    int64_t t1 = time_us();
    printf("total dur: %.3f ms\n", (float)(t1 - t0) / 1000);

    // NOTE:
    // 1. multi-threading: slower than single thread.
    // 2. compute time log to M.
    //
    // AMD Radeon Pro 560X Graphics Processor Polaris 21 Cores 1024 TMUs 64 ROPs
    // 16 Memory Size 4 GB Memory
    //
    // N = 11008; K = 4096
    // ==== Accelerate ====
    // M = 1
    // - 1 thread:  21.7 ms
    //
    // M = 4
    // - 1 thread:  13 ms
    // - 2 threads: 42 ms
    // - 4 threads: 28 ms
    //
    // M = 32
    // - 1 thread:  22 ms
    //
    // M = 40
    // - 1 thread:  24 ms
    //
    // M = 48
    // - 1 thread:  25 ms
    //
    // M = 64
    // - 1 thread:  28 ms
    //
    // M = 128
    // - 1 thread:  37 ms
    //
    // M = 256
    // - 1 thread:  73 ms
    // - 4 threads: 125 ms

    // ==== openblas (generally slower than accelerate, and the variance is
    // large) ==== M = 4
    // - 1 thread:  15 ms
    // - 2 threads: 32 ms
    // - 4 threads: 65 ms
    //
    // M = 16
    // - 1 thread:  22 ms
    //
    // M = 32
    // - 1 thread:  33 ms
    //
    // M = 32
    // - 1 thread:  42 ms

    if (false) {
      printf("calculated C with multi-threads:\n");
      print_array((float *)C, M, N);
      printf("\n");
    }
  }

  free(A);
  free(B);
  free(C);
}

void mnk(int M, int N, int K, float *A, float *B, float *C, const int bench_n,
         int warmup_n) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int lda = K;
  const int ldb = K;
  const int ldc = N;

  // warm up
  for (int i = 0; i < warmup_n; i++) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha,
                (const float *)A, lda, (const float *)B, ldb, beta, (float *)C,
                ldc);
  }

  int64_t dur[bench_n];

  for (int i = 0; i < bench_n; i++) {
    int64_t t0 = time_ns();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha,
                (const float *)A, lda, (const float *)B, ldb, beta, (float *)C,
                ldc);
    dur[i] = time_ns() - t0;
  }

  int64_t min = INT64_MAX;
  int64_t max = 0;
  int64_t total = 0;
  int64_t avg = 0;

  for (int i = 0; i < bench_n; i++) {
    total += dur[i];
    if (dur[i] < min) {
      min = dur[i];
    }

    if (dur[i] > max) {
      max = dur[i];
    }
  }

  avg = total / bench_n;

  printf("K:%5d, N:%5d, M:%5d, min:%9lld, max:%9lld, avg:%9lld\n", K, N, M,
         min, max, avg);

  // 0 ns when M*N*K < 1024.
  // N*K == 128 * 1024: run time always go down (> 10%, or even 30%)
  // N*K == 256 * 1024 slightly >  64 * 1024
}

// bench M, M, K
// B: trans
void bench_mnk() {
  const int warmup_n = 2;
  const int bench_n = 10;

  // Radeon Pro 560X 4 GB
  // 8 * 8 * 64 = 4GB, suddenly go slow when M > 64.

  const int M_max = 1024;
  const int K_max = 8*1024;
  const int N_max = 8*1024;

    // 3 memcpy(M*K + N*K + M*N), 1 transpose (K*N), M * N * (K mul + (K-1) sub)
  for (int K = 4096; K <= K_max; K *= 2) {
    for (int N = 4096; N <= N_max; N *= 2) {
      for (int M = 1; M <= M_max; M *= 2) {
        if (M == M_max) {
          fprintf(stderr, "=== K: %5d, N: %5d, M: %5d ===\n", K, N, M);
        }

        int64_t sizeA = sizeof(float) * M * K;
        int64_t sizeB = sizeof(float) * N * K;
        int64_t sizeC = sizeof(float) * M * N;

        float *A = malloc(sizeA);
        float *B = malloc(sizeB);
        float *C = malloc(sizeC);

        memset(A, 0, sizeA);
        memset(B, 0, sizeB);
        memset(C, 0, sizeC);

        // A: row major: K x M blocks
        for (int i = 0; i < M; i++) {
          for (int j = 0; j < K; j++) {
            *(A + i * K + j) = 1 + i * K + j;
          }
        }

        // B: row major: K x N blocks
        // NOTE: N x K (not K x N). Thus cblas_sgemm need CblasTrans.
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < K; j++) {
            *(B + i * K + j) = 1 + i * K + j;
          }
        }

        mnk(M, N, K, A, B, C, bench_n, warmup_n);

        if (M == M_max) {
          printf("\n");
        }

        free(A);
        free(B);
        free(C);
      }
    }
  }
}

// bench M, M, K
// B: trans
void bench_mnk_given_values() {
  const int warmup_n = 2;
  const int bench_n = 10;

  // Radeon Pro 560X 4 GB
  // K(8192) * N(8192) * M(64) = 4GB, suddenly go slow when M > 64.

  const int M_max = 256;
  const int kv[3] = {4096, 11080, 4096};
  const int nv[3] = {11080, 4096, 32000};

    // 3 memcpy(M*K + N*K + M*N), 1 transpose (K*N), M * N * (K mul + (K-1) sub)
  for (int i = 0; i < 3; i++) {
      int K = kv[i];{
      int N = nv[i];
      for (int M = 1; M <= M_max; M *= 2) {
        if (M == M_max) {
          fprintf(stderr, "=== K: %5d, N: %5d, M: %5d ===\n", K, N, M);
        }

        int64_t sizeA = sizeof(float) * M * K;
        int64_t sizeB = sizeof(float) * N * K;
        int64_t sizeC = sizeof(float) * M * N;

        float *A = malloc(sizeA);
        float *B = malloc(sizeB);
        float *C = malloc(sizeC);

        memset(A, 0, sizeA);
        memset(B, 0, sizeB);
        memset(C, 0, sizeC);

        // A: row major: K x M blocks
        for (int i = 0; i < M; i++) {
          for (int j = 0; j < K; j++) {
            *(A + i * K + j) = 1 + i * K + j;
          }
        }

        // B: row major: K x N blocks
        // NOTE: N x K (not K x N). Thus cblas_sgemm need CblasTrans.
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < K; j++) {
            *(B + i * K + j) = 1 + i * K + j;
          }
        }

        mnk(M, N, K, A, B, C, bench_n, warmup_n);

        if (M == M_max) {
          printf("\n");
        }

        free(A);
        free(B);
        free(C);
      }
    }
  }
}

// Build and run with Accelerate:
// gcc -O3 -std=c11 -framework Accelerate test_blas.c -o test_blas && ./test_blas

// Build and run with OpenBlas:
// gcc -O3 -std=c11 -lopenblas -L/usr/local/opt/openblas/lib -I/usr/local/opt/openblas/include -o test_blas test_blas.c &&  ./test_blas
int main() {
  // printf("\n=== test_mul_mat_1 === \n");
  // test_mul_mat_1();

  // printf("\n=== test_mul_mat_2 === \n");
  // test_mul_mat_2();

  // printf("\n=== test_mul_mat_3 === \n");
  // test_mul_mat_3();

  // printf("\n=== test_mul_mat_4 === \n");
  // test_mul_mat_4();

  // printf("\n=== test_mul_mat_5 === \n");
  // test_mul_mat_5();

  // printf("\n=== test_mul_mat_6 === \n");
  // test_mul_mat_6();

  // bench_mnk();
  bench_mnk_given_values();

  return 0;
}