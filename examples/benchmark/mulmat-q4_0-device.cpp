/*
License: MIT License.
Copied and modified from benchmark-q4_0-matmult.c.
*/
#include <assert.h>
#include <locale.h>
#include <math.h>
#include <string.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <queue>
#include <string>
#include <unordered_map>

#include "ggml.h"

void print_usage(char **argv) {
  fprintf(stderr, "usage: %s [options]\n", argv[0]);
  fprintf(stderr, "\n");
  fprintf(stderr, "options:\n");
  fprintf(stderr, "  -h, --help            show this help message and exit\n");
  fprintf(stderr,
          "  -t N, --threads N     number of threads to use during computation "
          "(default: 4)\n");
  fprintf(stderr, "\n");
}

int main(int argc, char **argv) {
  int n_threads = 4;
  const int n_benches = 20;

  // parse args.
  if (argc >= 2) {
    std::string arg = argv[1];

    if (arg == "-t" || arg == "--threads") {
      if (argc != 3) {
        fprintf(stderr, "%s: expect value. argc: %d\n", arg.c_str(), argc);
        print_usage(argv);
        exit(1);
      }

      n_threads = std::stoi(argv[2]);
      if (n_threads <= 0 || n_threads > 8) {
        fprintf(stderr,
                "n_threads: bad value %d. Expected value in range [1, 8]\n",
                n_threads);
        exit(1);
      }
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv);
      exit(0);
    } else {
      fprintf(stderr, "Invalid option: %s\n", arg.c_str());
      print_usage(argv);
      exit(1);
    }
  }

  const int n_device_types = 3;
  const char *device_names[n_device_types] = {"CPU ", "GPU ", "AUTO"};
  enum ggml_device_type devices[n_device_types] = {GGML_DEVICE_CPU, GGML_DEVICE_GPU, GGML_DEVICE_AUTO};

  for (int K = 32; K <= 8 * 1024; K *= 2) {
    for (int N = 32; N <= 8 * 1024; N *= 2) {
      size_t ctx_size = 0;
      ctx_size += K * N * ggml_type_sizef(GGML_TYPE_F32);
      ctx_size += K * N * ggml_type_sizef(GGML_TYPE_F32);
      ctx_size += K * N * ggml_type_sizef(GGML_TYPE_F32);
      ctx_size += K * sizeof(float);
      ctx_size += 1024 * 1024 * 300;

      printf("\nDevice    K      N      M       Elapsed(min)   Elapsed(max)   "
             "Elapsed(avg)\n");

      for (int M = 1; M <= 128; M *= 2) {
        if (M > 1) {
          printf("\n");
        }

        for (int j = 0; j < n_device_types; j++) {
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

          std::vector<int64_t> q4_0_buf(1 << 4, 0);

          struct ggml_tensor *m11 =
              ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
          ggml_set_f32(m11, 0.1f);

          struct ggml_tensor *q11 =
              ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);

          ggml_quantize_q4_0((const float *)m11->data, q11->data, K * N, K,
                             q4_0_buf.data());

          struct ggml_tensor *m2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
          ggml_set_f32(m2, 0.5f);

          struct ggml_tensor *node = ggml_mul_mat(ctx, q11, m2);

          struct ggml_cgraph gf = ggml_build_forward(node);
          gf.n_threads = n_threads;
          gf.device = devices[j]; // set device.

          int64_t min = INT64_MAX;
          int64_t max = 0;
          int64_t total = 0;
          int64_t avg = 0;

          int64_t dur[n_benches];

          for (int i = 0; i < n_benches; i++) {
            int64_t start = ggml_time_us();
            ggml_graph_compute(ctx, &gf);
            dur[i] = ggml_time_us() - start;

            total += dur[i];
            if (dur[i] < min) {
              min = dur[i];
            }

            if (dur[i] > max) {
              max = dur[i];
            }
            // TODO: refresh cache lines.
          }

          avg = total / n_benches;

          printf("%s %6d %6d %6d %10lli(us) %10lld(us) %10lld(us)\n",
                 device_names[j], K, N, M, min, max, avg);

          ggml_free(ctx);
        }
      }
    }
  }
  return 0;
}
