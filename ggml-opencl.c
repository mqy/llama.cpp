#include "ggml-opencl.h"

#define CL_TARGET_OPENCL_VERSION 110
#include <clblast_c.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "ggml.h"

#define MULTILINE_QUOTE(...) #__VA_ARGS__
const char * clblast_dequant = MULTILINE_QUOTE(

typedef uchar uint8_t;
typedef int int32_t;
typedef uint uint32_t;

constant uint QK4_0 = 32;
struct block_q4_0
{
    float d;
    uint8_t qs[32 / 2]; // 32: QK4_0
};

constant uint QK4_1 = 32;
struct block_q4_1
{
    float d;
    float m;
    uint8_t qs[32 / 2]; // 32: QK4_1
};

constant uint QK5_0 = 32;
struct __attribute__((packed)) block_q5_0
{
    half d;
    uint32_t qh;
    uint8_t qs[32 / 2]; // 32: QK5_0
};

constant uint QK5_1 = 32;
struct block_q5_1
{
    half d;
    half m;
    uint32_t qh;
    uint8_t qs[32 / 2]; // 32: QK5_1
};

constant uint QK8_0 = 32;
struct block_q8_0
{
    float d;
    uint8_t qs[32]; // 32: QK8_0
};


__kernel void dequantize_row_q4_0(__global struct block_q4_0* x, __global float* y) {
    uint qk = QK4_0;

    const uint i = get_global_id(0) / qk;
    const uint j = get_local_id(0);

    const float d = x[i].d;

    const int x0 = (x[i].qs[j] & 0xf) - 8;
    const int x1 = (x[i].qs[j] >>  4) - 8;

    y[i*qk + j + 0   ] = x0*d;
    y[i*qk + j + qk/2] = x1*d;
}

__kernel void dequantize_row_q4_1(__global struct block_q4_1* x, __global float* y) {
    uint qk = QK4_1;

    const uint i = get_global_id(0) / qk;
    const uint j = get_local_id(0);

    const float d = x[i].d;
    const float m = x[i].m;

    const int x0 = (x[i].qs[j] & 0xf);
    const int x1 = (x[i].qs[j] >>  4);

    y[i*qk + j + 0   ] = x0*d + m;
    y[i*qk + j + qk/2] = x1*d + m;
}

__kernel void dequantize_row_q5_0(__global struct block_q5_0* x, __global float* y) {
    uint qk = QK5_0;

    const uint i = get_global_id(0) / qk;
    const uint j = get_local_id(0);

    const float d = vload_half(0, (__global half*) &x[i].d);

    uint32_t qh = x[i].qh;

    const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
    const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

    const int32_t x0 = ((x[i].qs[j] & 0xf) | xh_0) - 16;
    const int32_t x1 = ((x[i].qs[j] >>  4) | xh_1) - 16;

    y[i*qk + j + 0   ] = x0*d;
    y[i*qk + j + qk/2] = x1*d;
}

 __kernel void dequantize_row_q5_1(__global struct block_q5_1* x, __global float* y) {
    uint qk = QK5_1;

    const uint i = get_global_id(0) / qk;
    const uint j = get_local_id(0);

    const float d = vload_half(0, (__global half*) &x[i].d);
    const float m = vload_half(0, (__global half*) &x[i].m);

    uint32_t qh = x[i].qh;

    const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
    const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

    const int x0 = (x[i].qs[j] & 0xf) | xh_0;
    const int x1 = (x[i].qs[j] >>  4) | xh_1;

    y[i*qk + j + 0   ] = x0*d + m;
    y[i*qk + j + qk/2] = x1*d + m;
}

__kernel void dequantize_row_q8_0(__global struct block_q8_0* x, __global float* y) {
    uint qk = QK8_0;
    const uint i = get_global_id(0) / qk;
    const uint j = get_local_id(0);

    const float d = x[i].d;
    y[i*qk + j] = x[i].qs[j]*d;
}

);

#define CL_CHECK(err, name)                                                                     \
    do {                                                                                        \
        cl_int err_ = (err);                                                                    \
        if (err_ != CL_SUCCESS) {                                                               \
            fprintf(stderr, "OpenCL %s error %d at %s:%d\n", name, err_, __FILE__, __LINE__);   \
            exit(1);                                                                            \
        }                                                                                       \
    } while (0)

#define ASSERT(cond)                                                           \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "Assert failed at %s:%d\n", __FILE__, __LINE__);   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define MIN(x, y) (x) < (y) ? (x) : (y)
#define MAX(x, y) (x) > (y) ? (x) : (y)

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel_q4_0, kernel_q4_1, kernel_q5_0, kernel_q5_1, kernel_q8_0;
static cl_mem cl_buffer_a, cl_buffer_qb, cl_buffer_b, cl_buffer_c;
static size_t cl_size_a = 0, cl_size_qb = 0, cl_size_b = 0, cl_size_c = 0;

static size_t device_global_mem_size = 0;
static size_t device_max_mem_alloc_size = 0;

static bool is_blocking_queue = false;
static bool ggml_cl_inited = false;

static cl_program build_program_from_source(cl_context ctx, cl_device_id dev, const char* program_buffer) {
    cl_program p;
    char *program_log;
    size_t program_size, log_size;
    int err;

    program_size = strlen(program_buffer);

    p = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        fprintf(stderr, "OpenCL error creating program");
        exit(1);
    }

    err = clBuildProgram(p, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {

        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return p;
}

void ggml_cl_init(void) {
    if (ggml_cl_inited) {
        return;
    }
    ggml_cl_inited = true;

    cl_int err = 0;
    char * GGML_CLBLAST_PLATFORM = getenv("GGML_CLBLAST_PLATFORM");
    char * GGML_CLBLAST_DEVICE = getenv("GGML_CLBLAST_DEVICE");
    int plat_num = (GGML_CLBLAST_PLATFORM == NULL ? 0 : atoi(GGML_CLBLAST_PLATFORM));
    int dev_num = (GGML_CLBLAST_DEVICE == NULL ? 0 : atoi(GGML_CLBLAST_DEVICE));

    char *GGML_CLBLAST_DEVICE_GLOBAL_MEM_SIZE =
        getenv("GGML_CLBLAST_DEVICE_GLOBAL_MEM_SIZE_MiB");
    if (GGML_CLBLAST_DEVICE_GLOBAL_MEM_SIZE != NULL) {
        device_global_mem_size =
            (size_t)atoi(GGML_CLBLAST_DEVICE_GLOBAL_MEM_SIZE);
        device_global_mem_size *= 1024 * 1024;
    }

    char *GGML_CLBLAST_DEVICE_MAX_MEM_ALLOC_SIZE =
        getenv("GGML_CLBLAST_DEVICE_MAX_MEM_ALLOC_SIZE_MiB");
    if (GGML_CLBLAST_DEVICE_MAX_MEM_ALLOC_SIZE != NULL) {
        device_max_mem_alloc_size =
            (size_t)atoi(GGML_CLBLAST_DEVICE_MAX_MEM_ALLOC_SIZE);
        device_max_mem_alloc_size *= 1024 * 1024;
    }

    printf("\nInitializing CLBlast (First Run)...");
    printf("\nAttempting to use: Platform=%d, Device=%d (If invalid, program will crash)\n",plat_num,dev_num);
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms*sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);
    platform = platforms[plat_num];
    char platform_buffer[1024];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_buffer), &platform_buffer, NULL);
    cl_uint num_devices;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    cl_device_id* devices = (cl_device_id*)malloc(num_devices*sizeof(cl_device_id));
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    device = devices[dev_num];
    char device_buffer[1024];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_buffer), &device_buffer, NULL);
    printf("Using Platform: %s Device: %s\n", platform_buffer, device_buffer);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CL_CHECK(err, "clCreateContext");
    queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    if (err != CL_SUCCESS) {
        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err == CL_SUCCESS) {
            is_blocking_queue = true;
            printf("CLBlast: the device does not support out of order queue, "
                   "fallback.\n\n");
        }
    }
    CL_CHECK(err, "clCreateCommandQueue");

    free(platforms);
    free(devices);

    program = build_program_from_source(context, device, clblast_dequant);

    // Prepare dequantize kernels
    kernel_q4_0 = clCreateKernel(program, "dequantize_row_q4_0", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q4_1 = clCreateKernel(program, "dequantize_row_q4_1", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q5_0 = clCreateKernel(program, "dequantize_row_q5_0", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q5_1 = clCreateKernel(program, "dequantize_row_q5_1", &err);
    CL_CHECK(err, "clCreateKernel");
    kernel_q8_0 = clCreateKernel(program, "dequantize_row_q8_0", &err);
    CL_CHECK(err, "clCreateKernel");
}

static int ggml_cl_calculate_pass(size_t size_a, size_t size_b, size_t size_c,
                            size_t size_qb) {
    // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetDeviceInfo.html
    // CL_DEVICE_MAX_MEM_ALLOC_SIZE:
    // Max size of memory object allocation in bytes. The minimum value is
    // max(min(1024 × 1024 × 1024, 1/4th of CL_DEVICE_GLOBAL_MEM_SIZE), 32 ×
    // 1024 × 1024) for devices that are not of type CL_DEVICE_TYPE_CUSTOM.

    size_t sz[] = {size_a, size_b, size_c, size_qb};

    size_t size_total = 0;
    size_t size_max = 0;

    for (int i = 0; i < 4; i++) {
        size_total += sz[i];
        if (sz[i] > size_max) {
            size_max = sz[i];
        }
    }

    int n_pass_0 = 0;
    if (device_global_mem_size > 0 && size_total > device_global_mem_size) {
        n_pass_0 = size_total / device_global_mem_size;
        if (size_total % device_global_mem_size != 0) {
            ++n_pass_0;
        }
    }

    size_t alloc_limit = device_max_mem_alloc_size;
    if (alloc_limit == 0 && device_global_mem_size > 0) {
        alloc_limit = MAX(MIN(1024 * 1024 * 1024, device_global_mem_size / 4),
                          32 * 1024 * 1024);
    }

    int n_pass_1 = 0;
    if (alloc_limit > 0 && size_max > alloc_limit) {
        n_pass_1 = size_max / alloc_limit;
        if (size_max % alloc_limit != 0) {
            ++n_pass_1;
        }
    }

    int n_pass = 1;
    if (n_pass_0 > 0 || n_pass_1 > 0) {
        n_pass = MAX(n_pass_0, n_pass_1);
    }

    if (n_pass > 1 && n_pass % 2 != 0) {
        ++n_pass;
    }

    // printf("==== device_global_mem_size: %zu MiB, device_max_mem_alloc_size: "
    //        "%zu MiB, "
    //        "total size: %zu MiB, max size: %zu MiB, "
    //        "alloc_limit: %zu MiB, n_pass_0: %d, n_pass_1: %d, n_pass: %d\n",
    //        device_global_mem_size / 1024 / 1024,
    //        device_max_mem_alloc_size / 1024 / 1024, size_total / 1024 / 1024,
    //        size_max / 1024 / 1024, alloc_limit / 1024 / 1024, n_pass_0,
    //        n_pass_1, n_pass);

    return n_pass;
}

static void ggml_cl_malloc(size_t req_size, size_t* cur_size, cl_mem_flags flags, cl_mem* buf) {
    if (req_size <= *cur_size) {
        return;
    }

    // Reallocate buffer with enough space
    if (*cur_size > 0) {
        clReleaseMemObject(*buf);
    }
    cl_int err;
    *buf = clCreateBuffer(context, flags, req_size, NULL, &err);
    *cur_size = req_size;
    CL_CHECK(err, "clCreateBuffer");
}

void ggml_cl_sgemm_wrapper(
        const enum ggml_blas_order order, const enum ggml_blas_op trans_a, const enum ggml_blas_op trans_b,
        const int m, const int n, const int k,
        const float alpha, const void *host_a, const int lda,
        const float *host_b, const int ldb, const float beta,
        float *host_c, const int ldc, const int btype) {
    cl_int err = 0;

    cl_kernel kernel;
    size_t global = n * k, local, size_qb = 0;
    bool dequant;

    switch (btype) {
    case GGML_TYPE_F32:
        dequant = false;
        break;
    case GGML_TYPE_Q4_0:
        dequant = true;
        kernel = kernel_q4_0;
        local = 16;
        size_qb = global * (sizeof(float) + local) / 32;
        break;
    case GGML_TYPE_Q4_1:
        dequant = true;
        kernel = kernel_q4_1;
        local = 16;
        size_qb = global * (sizeof(float) * 2 + local) / 32;
        break;
    case GGML_TYPE_Q5_0:
        dequant = true;
        kernel = kernel_q5_0;
        local = 16;
        size_qb = global * (sizeof(ggml_fp16_t) + sizeof(uint32_t) + local) / 32;
        break;
    case GGML_TYPE_Q5_1:
        dequant = true;
        kernel = kernel_q5_1;
        local = 16;
        size_qb = global * (sizeof(ggml_fp16_t) * 2 + sizeof(uint32_t) + local) / 32;
        break;
    case GGML_TYPE_Q8_0:
        dequant = true;
        kernel = kernel_q8_0;
        local = 32;
        size_qb = global * (sizeof(float) + local) / 32;
        break;
    default:
        fprintf(stderr, "Error: Unsupported OpenCL btype %d\n", btype);
        abort();
    }

    const size_t size_a = m * k * sizeof(float);
    size_t size_b = n * k * sizeof(float);
    size_t size_c = m * n * sizeof(float);

    // Old devices tend unable to run out-of-order queue, and suffer from low
    // memory. So let's try n pass.

    int num_pass = 1;
    if (device_global_mem_size > 0 || device_max_mem_alloc_size > 0) {
        num_pass = ggml_cl_calculate_pass(size_a, size_b, size_c, dequant ? size_qb : 0);
    }

    int N = n;
    int LDC = ldc;

    if (num_pass > 1) {
        ASSERT(size_b % num_pass == 0);
        ASSERT(size_c % num_pass == 0);
        ASSERT(size_qb % num_pass == 0);
        ASSERT(n % num_pass == 0);
        ASSERT(ldc % num_pass == 0);

        size_b /= num_pass;
        size_c /= num_pass;
        size_qb /= num_pass;
        N = n / num_pass;
        LDC = ldc / num_pass;
    }

    // Prepare buffers
    ggml_cl_malloc(size_a, &cl_size_a, CL_MEM_READ_ONLY, &cl_buffer_a);
    if (dequant) {
        ggml_cl_malloc(size_qb, &cl_size_qb, CL_MEM_READ_ONLY, &cl_buffer_qb);
    }
    ggml_cl_malloc(size_b, &cl_size_b, CL_MEM_READ_WRITE, &cl_buffer_b);
    ggml_cl_malloc(size_c, &cl_size_c, CL_MEM_WRITE_ONLY, &cl_buffer_c);

    float *temp_host_c = host_c;
    if (num_pass > 1) {
        temp_host_c = malloc(sizeof(float) * size_c / num_pass);
    }

    cl_event ev_a, ev_qb, ev_b;

    cl_int is_async = is_blocking_queue ? CL_TRUE : CL_FALSE;
    for (int pass = 0; pass < num_pass; pass++) {
        if (dequant) {
            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_buffer_qb);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_buffer_b);
            CL_CHECK(err, "clSetKernelArg");
            err = clEnqueueWriteBuffer(queue, cl_buffer_qb, is_async, 0, size_qb, host_b + (pass * size_b), 0, NULL, &ev_qb);
            CL_CHECK(err, "clEnqueueWriteBuffer qb");
        } else {
            err = clEnqueueWriteBuffer(queue, cl_buffer_b, is_async, 0, size_b, host_b + (pass * size_b), 0, NULL, &ev_b);
            CL_CHECK(err, "clEnqueueWriteBuffer b");
        }

        err = clEnqueueWriteBuffer(queue, cl_buffer_a, is_async, 0, size_a, host_a, 0, NULL, &ev_a);
        CL_CHECK(err, "clEnqueueWriteBuffer a");
        if (dequant) {
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 1, &ev_qb, &ev_b);
            CL_CHECK(err, "clEnqueueNDRangeKernel");
            clReleaseEvent(ev_qb);
        }
        clWaitForEvents(1, &ev_a);
        clWaitForEvents(1, &ev_b);
        clReleaseEvent(ev_a);
        clReleaseEvent(ev_b);

        cl_event ev_sgemm;
        CLBlastStatusCode status = CLBlastSgemm((CLBlastLayout)order,
                                                (CLBlastTranspose)trans_a, (CLBlastTranspose)trans_b,
                                                m, N, k,
                                                alpha,
                                                cl_buffer_a, 0, lda,
                                                cl_buffer_b, 0, ldb,
                                                beta,
                                                cl_buffer_c, 0, LDC,
                                                &queue, &ev_sgemm);

        if (status != CLBlastSuccess) {
            fprintf(stderr, "Error: CLBlast SGEMM %d\n", status);
            abort();
        }

        cl_event ev_c;
        clEnqueueReadBuffer(queue, cl_buffer_c, CL_TRUE, 0, size_c, temp_host_c, 1, &ev_sgemm, &ev_c);

        // Wait for completion
        clWaitForEvents(1, &ev_c);
        clReleaseEvent(ev_sgemm);
        clReleaseEvent(ev_c);

        if (num_pass > 1) {
            // copy c to host_c row by row.
            // TODO: optimize performance.
            size_t block_offset = pass * N;
            for (int im = 0; im < m; im++) {
                memcpy((void *)(host_c + im * n + block_offset),
                        (const void *)(temp_host_c + im * N), N * sizeof(float));
            }
        }
    }

    if (num_pass > 1) {
        free(temp_host_c);
    }
}
