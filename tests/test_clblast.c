#define CL_TARGET_OPENCL_VERSION 110

// #define CL_HPP_TARGET_OPENCL_VERSION 210
// #define CL_TARGET_OPENCL_VERSION 210

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// #include "ggml-opencl.h"
#include "ggml.h"
#include <clblast_c.h>

//#define USE_SGEMM_WRAPPER 1

// kernel implementations are copied frm ggml-opencl.c.

#define CL_CHECK(err, ...)                                                     \
    do {                                                                       \
        if ((err) != CL_SUCCESS) {                                             \
            fprintf(stderr, "OpenCL error %d at line %d. ", err, __LINE__);    \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, "\n");                                             \
            abort();                                                           \
        }                                                                      \
    } while (0)

#define ASSERT(x)                                                              \
    do {                                                                       \
        if (!(x)) {                                                            \
            fprintf(stderr, "assert failed. line: %d.", __LINE__);             \
            abort();                                                           \
        }                                                                      \
    } while (0)

static inline int64_t time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}

static cl_program program;
static cl_kernel kernel_q4_0;

static cl_mem cl_buffer_a, cl_buffer_qb, cl_buffer_b, cl_buffer_c;
static size_t cl_size_a = 0, cl_size_qb = 0, cl_size_b = 0, cl_size_c = 0;

static void ggml_cl_malloc(cl_context context, size_t req_size,
                           size_t *cur_size, cl_mem_flags flags, cl_mem *buf) {
    if (req_size <= *cur_size) {
        return;
    }

    // Reallocate buffer with enough space
    if (*cur_size > 0) {
        cl_int err = clReleaseMemObject(*buf);
        CL_CHECK(err, "clReleaseMemObject");
    }
    cl_int err;
    *buf = clCreateBuffer(context, flags, req_size, NULL, &err);
    *cur_size = req_size;
    CL_CHECK(err, "clCreateBuffer");
}

#define MULTILINE_QUOTE(...) #__VA_ARGS__
const char *clblast_dequant = MULTILINE_QUOTE(
    struct block_q4_0 {
        float d;
        uchar qs[16];
    };

    __kernel void dequantize_row_q4_0(__global struct block_q4_0 *blocks,
                                      __global float *result) {
        const uint i = get_global_id(0) / 32;
        const uint l = get_local_id(0);

        const float d = blocks[i].d;

        const uchar vi = blocks[i].qs[l];

        const uint index = i * 32 + l * 2;
        result[index + 0] = ((vi & 0xf) - 8) * d;
        result[index + 1] = ((vi >> 4) - 8) * d;
    });

static cl_program build_program_from_source(cl_context ctx, cl_device_id dev,
                                            const char *program_buffer) {
    cl_program p;
    char *program_log;
    size_t program_size, log_size;
    int err;

    program_size = strlen(program_buffer);

    p = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer,
                                  &program_size, &err);
    if (err < 0) {
        fprintf(stderr, "OpenCL error creating program");
        exit(1);
    }

    err = clBuildProgram(p, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, log_size + 1,
                              program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return p;
}

#if !defined(USE_SGEMM_WRAPPER)

struct test_ctx {
    int platform_id;
    int device_id;

    cl_platform_id platform;
    cl_device_id device;

    cl_context context;
    cl_command_queue queue;
};

// context, device, a blocking queue, q4_0 kernel.
void init_test_ctx(struct test_ctx *tc) {
    cl_int err;

    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    CL_CHECK(err, "clGetPlatformIDs for num_platforms");
    ASSERT(num_platforms > 0);

    cl_platform_id *platforms =
        (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    CL_CHECK(err, "clGetPlatformIDs for platforms");
    tc->platform = platforms[tc->platform_id];

    free(platforms);

    cl_uint num_devices;
    err =
        clGetDeviceIDs(tc->platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    CL_CHECK(err, "clGetDeviceIDs for num_devices");
    ASSERT(num_devices > 0);

    cl_device_id *devices =
        (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
    err = clGetDeviceIDs(tc->platform, CL_DEVICE_TYPE_ALL, num_devices, devices,
                         NULL);
    CL_CHECK(err, "clGetDeviceIDs for devices");
    tc->device = devices[tc->device_id];

    free(devices);

    char platform_buffer[1024];
    char device_buffer[1024];
    clGetPlatformInfo(tc->platform, CL_PLATFORM_NAME, sizeof(platform_buffer),
                      &platform_buffer, NULL);
    clGetDeviceInfo(tc->device, CL_DEVICE_NAME, sizeof(device_buffer),
                    &device_buffer, NULL);
    printf("Using Platform: %s, Device: %s\n", platform_buffer, device_buffer);

    tc->context = clCreateContext(NULL, 1, &tc->device, NULL, NULL, &err);
    CL_CHECK(err, "clCreateContext");

    tc->queue = clCreateCommandQueue(tc->context, tc->device, 0, &err);
    CL_CHECK(err, "clCreateCommandQueue");

    // kernels
    program =
        build_program_from_source(tc->context, tc->device, clblast_dequant);
    const char *kernel_name = "dequantize_row_q4_0";
    kernel_q4_0 = clCreateKernel(program, kernel_name, &err);
    CL_CHECK(err, "clCreateKernel %s", kernel_name);
}

const bool global_buffers = true;

// c = a x bT
// `size_qb` is char size of quantitized `b`, valid only if `dequant` is true.
void sgemm_mnk(struct test_ctx *tc, const float *a, const void *b, float *c,
               size_t m, size_t n, size_t k) {
    cl_context context = tc->context;
    cl_device_id device = tc->device;
    cl_command_queue queue = tc->queue;

    const bool dequant = true;
    size_t global = n * k;
    size_t local = 16;
    size_t size_qb = n*k * (sizeof(float) + 16) / 32;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // k, n, n for b not transpose
    // k, k, n for b transpose
    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldc = n;

    size_t size_a = sizeof(float) * m * k;
    size_t size_b = sizeof(float) * n * k;
    size_t size_c = sizeof(float) * m * n;

    cl_int err;

    // Create buffers.
    {
        int64_t t0 = ggml_time_us();

        if (global_buffers) {
            ggml_cl_malloc(tc->context, size_a, &cl_size_a, CL_MEM_READ_ONLY,
                           &cl_buffer_a);
            if (dequant) {
                ggml_cl_malloc(tc->context, size_qb, &cl_size_qb,
                               CL_MEM_READ_ONLY, &cl_buffer_qb);
            }
            ggml_cl_malloc(tc->context, size_b, &cl_size_b, CL_MEM_READ_WRITE,
                           &cl_buffer_b);
            ggml_cl_malloc(tc->context, size_c, &cl_size_c, CL_MEM_WRITE_ONLY,
                           &cl_buffer_c);
        } else {
            cl_int err = 0;
            cl_buffer_a =
                clCreateBuffer(context, CL_MEM_READ_ONLY, size_a, NULL, &err);
            CL_CHECK(err, "clCreateBuffer for a");
            cl_buffer_b =
                clCreateBuffer(context, CL_MEM_READ_WRITE, size_b, NULL, &err);
            CL_CHECK(err, "clCreateBuffer for b");
            cl_buffer_c =
                clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_c, NULL, &err);
            CL_CHECK(err, "clCreateBuffer for c");
            if (dequant) {
                cl_buffer_qb = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                              size_qb, NULL, &err);
                CL_CHECK(err, "clCreateBuffer for qb");
            }
        }
        printf("m: %3zu, create buffers: %5.3f ms", m,
               (ggml_time_us() - t0) / 1000.0);
    }

    if (dequant) {
        int64_t t0 = ggml_time_us();

        err = clEnqueueWriteBuffer(queue, cl_buffer_qb, CL_TRUE, 0, size_qb, b,
                                   0, NULL, NULL);
        CL_CHECK(err, "clEnqueueWriteBuffer for qb");

        cl_kernel kernel = kernel_q4_0;

        cl_mem *args[] = {&cl_buffer_qb, &cl_buffer_b};
        for (int i = 0; i < 2; i++) {
            err = clSetKernelArg(kernel, i, sizeof(cl_mem), args[i]);
            CL_CHECK(err, "clSetKernelArg #%d", i);
        }

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0,
                                     NULL, NULL);
        CL_CHECK(err, "clEnqueueNDRangeKernel");

        printf(", dequant: %7.3f ms", (ggml_time_us() - t0) / 1000.0);
    }

    {
        int64_t t0 = ggml_time_us();
        if (!dequant) {
            err = clEnqueueWriteBuffer(queue, cl_buffer_b, CL_TRUE, 0, size_b,
                                       b, 0, NULL, NULL);
            CL_CHECK(err, "clEnqueueWriteBuffer for b");
        }

        err = clEnqueueWriteBuffer(queue, cl_buffer_a, CL_TRUE, 0, size_a, a, 0,
                                   NULL, NULL);
        CL_CHECK(err, "clEnqueueWriteBuffer for a");

        printf(", write buffers: %5.3f ms", (ggml_time_us() - t0) / 1000.0);
    }

    // sgemm.
    {
        int64_t t0 = ggml_time_us();
        CLBlastStatusCode status = CLBlastSgemm(
            CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeYes, m,
            n, k, alpha, cl_buffer_a, 0, lda, cl_buffer_b, 0, ldb, beta,
            cl_buffer_c, 0, ldc, &queue, NULL);
        CL_CHECK(status, "CLBlastSgemm");
        printf(", segemm: %7.3f ms", (ggml_time_us() - t0) / 1000.0);
    }

    // copy data from buffer back to c.
    {
        int64_t t0 = ggml_time_us();
        err = clEnqueueReadBuffer(queue, cl_buffer_c, CL_TRUE, 0, size_c, c, 0,
                                  NULL, NULL);

        CL_CHECK(err, "clEnqueueReadBuffer for c");
        printf(", read buffer: %7.3f ms", (ggml_time_us() - t0) / 1000.0);
    }

    if (!global_buffers) {
        clReleaseMemObject(cl_buffer_a);
        clReleaseMemObject(cl_buffer_b);
        clReleaseMemObject(cl_buffer_c);
        if (cl_buffer_qb) {
            clReleaseMemObject(cl_buffer_qb);
        }
    }
}
#else

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;

static bool ggml_cl_inited = false;
static bool async_queue = true;
static bool enable_info = false;

void ggml_cl_init(bool show_info) {
    if (ggml_cl_inited) {
        return;
    }
    ggml_cl_inited = true;
    enable_info = show_info;

    cl_int err = 0;
    char *GGML_CLBLAST_PLATFORM = getenv("GGML_CLBLAST_PLATFORM");
    char *GGML_CLBLAST_DEVICE = getenv("GGML_CLBLAST_DEVICE");
    int plat_num =
        (GGML_CLBLAST_PLATFORM == NULL ? 0 : atoi(GGML_CLBLAST_PLATFORM));
    int dev_num = (GGML_CLBLAST_DEVICE == NULL ? 0 : atoi(GGML_CLBLAST_DEVICE));
    if (enable_info) {
        printf("\nInitializing CLBlast (First Run)...\n");
        printf("\nGGML_CLBLAST_PLATFORM=%s, GGML_CLBLAST_DEVICE=%s",
               GGML_CLBLAST_PLATFORM ? GGML_CLBLAST_PLATFORM : "",
               GGML_CLBLAST_DEVICE ? GGML_CLBLAST_DEVICE : "");
        printf("\nAttempting to use: Platform=%d, Device=%d (If invalid, "
               "program will crash)\n",
               plat_num, dev_num);
    }
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id *platforms =
        (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));

    clGetPlatformIDs(num_platforms, platforms, NULL);
    platform = platforms[plat_num];

    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);

    cl_device_id *devices =
        (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    device = devices[dev_num];

    if (enable_info) {
        char platform_buffer[1024];
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_buffer),
                        &platform_buffer, NULL);

        char device_buffer[1024];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_buffer),
                        &device_buffer, NULL);

        printf("Using Platform: %s, Device: %s\n", platform_buffer,
               device_buffer);
    }
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CL_CHECK(err, "clCreateContext");

    queue = clCreateCommandQueue(context, device,
                                 CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    if (err != CL_SUCCESS) {
        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err == CL_SUCCESS) {
            async_queue = false;
            if (enable_info) {
                printf("The device does not support async queue, fallback to "
                       "blocking queue.\n");
            }
        }
    }
    CL_CHECK(err, "clCreateCommandQueue");

    free(platforms);
    free(devices);

    program = build_program_from_source(context, device, clblast_dequant);

    // Prepare dequantize kernels
    kernel_q4_0 = clCreateKernel(program, "dequantize_row_q4_0", &err);
    CL_CHECK(err, "clCreateKernel");
}

void ggml_cl_sgemm_wrapper(const int m, const int n, const int k,
                           const float *host_a, const void *host_b,
                           float *host_c) {
    cl_int err = 0;

    bool dequant = true;

    cl_kernel kernel = kernel_q4_0;

    size_t global = n * k;
    size_t local = 16;
    size_t size_qb = n*k * (sizeof(float) + 16) / 32;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    const size_t lda = k;
    const size_t ldb = k;
    const size_t ldc = n;

    const size_t size_a = m * k * sizeof(float);
    const size_t size_b = n * k * sizeof(float);
    const size_t size_c = m * n * sizeof(float);

    // Prepare buffers
    ggml_cl_malloc(context, size_a, &cl_size_a, CL_MEM_READ_ONLY,
                    &cl_buffer_a);
    if (dequant) {
        ggml_cl_malloc(context, size_qb, &cl_size_qb,
                        CL_MEM_READ_ONLY, &cl_buffer_qb);
    }
    ggml_cl_malloc(context, size_b, &cl_size_b, CL_MEM_READ_WRITE,
                    &cl_buffer_b);
    ggml_cl_malloc(context, size_c, &cl_size_c, CL_MEM_WRITE_ONLY,
                    &cl_buffer_c);

    int64_t t0 = ggml_time_us();

    // if (dequant) {
    //     err = clEnqueueWriteBuffer(queue, cl_buffer_qb, CL_TRUE, 0, size_qb,
    //                                host_b, 0, NULL, NULL);
    //     CL_CHECK(err, "clEnqueueWriteBuffer qb");

    //     err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buffer_qb);
    //     CL_CHECK(err, "clSetKernelArg");

    //     err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buffer_b);
    //     CL_CHECK(err, "clSetKernelArg");

    //     err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0,
    //                                  NULL, NULL);
    //     CL_CHECK(err, "clEnqueueNDRangeKernel");
    // } else {
    //     err = clEnqueueWriteBuffer(queue, cl_buffer_b, CL_TRUE, 0, size_b,
    //                                host_b, 0, NULL, NULL);
    //     CL_CHECK(err, "clEnqueueWriteBuffer b");
    // }

    // err = clEnqueueWriteBuffer(queue, cl_buffer_a, CL_TRUE, 0, size_a, host_a,
    //                            0, NULL, NULL);
    // CL_CHECK(err, "clEnqueueWriteBuffer a");


    if (dequant) {
        int64_t t0 = ggml_time_us();

        err = clEnqueueWriteBuffer(queue, cl_buffer_qb, CL_TRUE, 0, size_qb, host_b,
                                   0, NULL, NULL);
        CL_CHECK(err, "clEnqueueWriteBuffer for qb");

        cl_kernel kernel = kernel_q4_0;

        cl_mem *args[] = {&cl_buffer_qb, &cl_buffer_b};
        for (int i = 0; i < 2; i++) {
            err = clSetKernelArg(kernel, i, sizeof(cl_mem), args[i]);
            CL_CHECK(err, "clSetKernelArg #%d", i);
        }

        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0,
                                     NULL, NULL);
        CL_CHECK(err, "clEnqueueNDRangeKernel");

        printf(", dequant: %7.3f ms", (ggml_time_us() - t0) / 1000.0);
    }

    {
        int64_t t0 = ggml_time_us();
        if (!dequant) {
            err = clEnqueueWriteBuffer(queue, cl_buffer_b, CL_TRUE, 0, size_b,
                                       host_b, 0, NULL, NULL);
            CL_CHECK(err, "clEnqueueWriteBuffer for b");
        }

        err = clEnqueueWriteBuffer(queue, cl_buffer_a, CL_TRUE, 0, size_a, host_a, 0,
                                   NULL, NULL);
        CL_CHECK(err, "clEnqueueWriteBuffer for a");

        printf(", write buffers: %5.3f ms", (ggml_time_us() - t0) / 1000.0);
    }


    int64_t t1 = ggml_time_us();

    // CLBlastStatusCode status = CLBlastSgemm(
    //     CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeYes, m, n, k,
    //     alpha, cl_buffer_a, 0, lda, cl_buffer_b, 0, ldb, beta, cl_buffer_c, 0,
    //     ldc, &queue, NULL);

    // sgemm.
    {
        int64_t t0 = ggml_time_us();
        CLBlastStatusCode status = CLBlastSgemm(
            CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeYes, m,
            n, k, alpha, cl_buffer_a, 0, lda, cl_buffer_b, 0, ldb, beta,
            cl_buffer_c, 0, ldc, &queue, NULL);
        CL_CHECK(status, "CLBlastSgemm");
        printf(", segemm: %7.3f ms", (ggml_time_us() - t0) / 1000.0);
    }

    int64_t t2 = ggml_time_us();

    // int rc = clEnqueueReadBuffer(queue, cl_buffer_c, CL_TRUE, 0, size_c, host_c,
    //                              0, NULL, NULL);
    // if (rc != CL_SUCCESS) {
    //     fprintf(stderr, "Error: clEnqueueReadBuffer returns %d\n", rc);
    //     abort();
    // }


    // copy data from buffer back to c.
    {
        int64_t t0 = ggml_time_us();
        err = clEnqueueReadBuffer(queue, cl_buffer_c, CL_TRUE, 0, size_c, host_c, 0,
                                  NULL, NULL);

        CL_CHECK(err, "clEnqueueReadBuffer for c");
        printf(", read buffer: %7.3f ms", (ggml_time_us() - t0) / 1000.0);
    }


    int64_t t3 = ggml_time_us();

    printf("\nm: %2d, n: %5d, k: %5d, prepare: %6.3f ms, sgemm: %6.3f ms, "
           "copy: %9.3f ms, total: %9.3f ms\n",
           m, n, k, 1.0 * (t1 - t0) / 1000.0, 1.0 * (t2 - t1) / 1000.0,
           1.0 * (t3 - t2) / 1000.0, 1.0 * (t3 - t0) / 1000.0);
}

#endif

// LDFLAGS for Darwin: -lclblast -framework OpenCL
// LDFLAGS for Linux: -lclblast -lOpenCL

// USE_SGEMM_WRAPPER=  gcc -O3 -std=c11 -lclblast -framework OpenCL -I..
// test_clblast.c ../ggml.c -o test_clblast
// && time ./test_clblast

// USE_SGEMM_WRAPPER=1 gcc -O3 -std=c11 -lclblast -framework OpenCL -I..
// test_clblast.c ../ggml.c ../ggml-opencl.c -o test_clblast
// && time ./test_clblast
int main(void) {
#if defined(USE_SGEMM_WRAPPER)
    ggml_cl_init(true);
#else
    struct test_ctx ctx = {.platform = 0, .device_id = 1};
    init_test_ctx(&ctx);
#endif

    const int m_step = 16;
    const int num_m = 16;
    const int max_m = m_step * num_m;

    const size_t n_list[] = {4096, 4096, 11008, 5120, 5120, 13824};
    const size_t k_list[] = {4096, 11008, 4096, 5120, 13824, 5120};
    const int n_groups = sizeof(n_list) / sizeof(size_t);

    int max_n = 0;
    int max_k = 0;
    int64_t max_nk = 0;

    for (int i = 0; i < n_groups; i++) {
        int64_t v = n_list[i] * k_list[i];
        if (v > max_nk) {
            max_nk = v;
        }
        if (n_list[i] > max_n) {
            max_n = n_list[i];
        }
        if (k_list[i] > max_k) {
            max_k = k_list[i];
        }
    }

    // M x K
    float *a = (float *)malloc(sizeof(float) * max_m * max_k);
    ASSERT(a);
    // N x K
    float *b = (float *)malloc(sizeof(float) * max_nk);
    ASSERT(b);
    // M x N
    float *c = (float *)malloc(sizeof(float) * max_m * max_n);
    ASSERT(c);

    for (int i = 0; i < n_groups; i++) {
        size_t N = n_list[i];
        size_t K = k_list[i];

        int64_t q40_block_size = (sizeof(float) + 16);
        char *qb = malloc(N * K * q40_block_size / 32);
        ASSERT(qb);

        int64_t *hist = malloc(2 * N * K * sizeof(int64_t));
        ASSERT(hist);

        printf("\n=== n: %5zu, k: %5zu ===\n", N, K);

        for (int i = 0; i < num_m; i++) {
            size_t M = (i + 1) * m_step;

            const int n_pass = 5;
            for (int pass = 0; pass < n_pass; pass++) {
                for (size_t i = 0; i < M * K; ++i) {
                    a[i] = 0.1f;
                }
                for (size_t i = 0; i < N * K; ++i) {
                    b[i] = 0.2f;
                }
                for (size_t i = 0; i < M * N; ++i) {
                    c[i] = 0.0f;
                }

                // 32 floats => a Q4_0 block
                // b is a N x K matrix.
                size_t size_qb = ggml_quantize_q4_0((const float *)b,
                                                    (void *)qb, N * K, K, hist);
                ASSERT(size_qb == N * K / 32 * (sizeof(float) + 16));

                int64_t sgemm_us;

#if defined(USE_SGEMM_WRAPPER)
                int64_t t0 = time_us();

                ggml_cl_sgemm_wrapper(M, N, K, a, qb, c);

                sgemm_us = time_us() - t0;
#else

                int64_t t0 = time_us();

                sgemm_mnk(&ctx, a, qb, c, M, N, K);

                sgemm_us = time_us() - t0;
#endif

                ASSERT(c[0] > 0);

                printf(", total: %7.3f ms\n", 1.0 * sgemm_us / 1000.0f);
            }

            printf("\n");
        }

        if (qb) {
            free(qb);
        }
        if (hist) {
            free(hist);
        }
    }

    free(a);
    free(b);
    free(c);

#if !defined(USE_SGEMM_WRAPPER)
    clReleaseCommandQueue(ctx.queue);
    clReleaseContext(ctx.context);
#endif

    return 0;
}
