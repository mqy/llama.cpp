# Fine Tune MUL_MAT with Bench

## Introduction

GGML defines three task types(stages): INIT, COMPUTE, FINALIZE. All nodes has
COMPUTE stage, some has INIT stage, the FINALIZE is never used.

General speaking, codes run in GPU(BLAS) are not suitable to run with multi OS
threads -- sometimes very slow, but CPU could and scales well. So to speedup
large prompt and avoid spinning, the master code force 1-thread when (M >=32).

In `master` branch, the `mul_mat` codes run in several implicit profiles.

- pure cpu: INIT: very fast, COMPUTE: the computation time is proportional to M.
- CUDA(cuBLAS)/ClBlast: COMPUTE: de-quantization and mul_mat in GPU.
- Accelerate/OpenBLAS: COMPUTE: de-quantization in CPU, mul_mat in GPU.

I observed the following "facts" for Accelerate/OpenBLAS:

- Whatever the M is, given N and K, the de-quantization time is constant in theory.
- The mul_mat time in GPU is heavy (tens to hundreds ms), goes up very slow when
  M doubles.
- In the large M range, the de-quantization time accounts for a large proportion
  of the total calculation time. For example, for 7B, Q4_0, NxK=4096x4096, when
  about M < 100, the proportion of de-quantization time is far more than `50%`.
  Other NxK combinations have a similar situation, except that the range of M is
  not as large as NxK=4096x4096.

In theory, if we split COMPUTE stage as INIT + COMPUTE, we MAY speedup prompt
eval time a lot: up to 50% for large M range (e.g. 32 - 128). The following
diagram shows the times (INIT in CPU, COMPUTE in GPU) for 7B/Q4_0/Accelerate.

<image src="images/q40-1-thread.png" with="600"></image>

I manged to improve the `threading` problem(s) since Apr this year. I dropped two
pull requests because they did not solve the key problem but introduced noisy.
In the second pull request, @gerganov hinted me the `1-thread blas` problem, so
I followed this direction since then.

At first I implemented the new threading framework that supports `wait/notify`,
subtle and fragile to dead lock. I'm happy that it works. I had tried bench online
by comparing CPU/GPU time, finally I replace that with offline bench. To explicitly
control details (how to parallel, when to wait, how to select the best executing plan),
I had to define task config, task profiles. Finally I got the demo solution as follows.

The eval time of long prompt decreases a lot. For example, `examples/chat.sh` with
4 threads, the prompt eval time of 99 tokens decreases up to **-40%** in my device.
Tests for broad prompt size show speed up of 10% - 40%.

The key factor for speeding up is parallelism, followed by more accurate profile
selection. The latter, although secondary, is necessary in the case of multithreading.

Just like Accelerate/OpenBLAS, the de-quantization time in CUDA/ClBlast MAY NOT
compete multiple CPU threads on some devices. In case of this, we can add profiles
for them to run de-quantization in CPU and run mul_mat in GPU.

With explicit task config profiles and bench data, I'm expecting that we are able
to run any task stage in any backend. For example: for q4_0, we could run INIT in
CUDA and COMPUTE in Accelerate -- if the overall speed competes other profiles.

Anyway, current solution is in demo stage and is incomplete due to various reasons,
you will read them in the following sections.

I'm new to machine learning this year and have almost no knowledge of AI models
and algorithms. There must be a lot of problems with this CL, please do not
hesitate to advise. Thanks!

## Solutions

- Update mul_mat BLAS codes: allow de-quantizing in CPU or GPU (if possible).
- Explicitly task config and profiles:
  * define conf profiles (for example, init in CPU, compute in GPU);
  * define for any stage: compute in parallel or not, idle wait or not.
  Therefor the task stages with GPU backend are not limited to 1 thread.
- New threading framework: combine `spin` + `cond-wait/broadcast`. Without wait,
  workers busy spinning may causes overheat and slow down the overall speed.
  The mul_mat compute time is long enough (often tens of ms), so the wait/notify
  overhead (at most tens of us) is OK.
- A tune tool for benching. With bench data, given N/K and n_threads, we could
  estimate total computing time for any M (if within range), thus could select
  the fastest profile.

We do bench and save bench results. Every llama model defines limited (N, K) pairs. 
Given M/N/K, we can run every compute stage (if exists) for some times, choose
the minimal overall time as reference value.

On llama start, load bench data. Before compute node, we select the fastest profile.
When compute, the `dst->task_conf` along with `params` controls which part of the
codes to run.

See section "**How To Estimate Execution Time**" for the reference algorithm.

**Important data structures**

```c
// update
enum ggml_backend {
    GGML_BACKEND_UNKNOWN = 0, // new
    GGML_BACKEND_CPU = 1, // original value 0
    GGML_BACKEND_CUDA = 2, // new
    GGML_BACKEND_ACCELERATE = 3, // new
    GGML_BACKEND_OPENBLAS = 4, // new
    GGML_BACKEND_CLBLAST = 5, // new
};

// new
struct ggml_task_conf {
    int backend; // enum ggml_backend
    bool parallel;
    bool wait;
};

struct ggml_tensor {
    //...
    struct ggml_task_conf task_conf[3]; // new
    //...
}
```

**Explicitly task config and profiles**

```c
void ggml_mulmat_tune_setup_task_conf(struct ggml_mulmat_tune *tune) {
    tune->n_profiles = 1;
    // cpu init + cpu compute
    tune->conf[0][0] = (struct ggml_task_conf){
        .backend = GGML_BACKEND_CPU,
    };
    tune->conf[0][1] = (struct ggml_task_conf){
        .backend = GGML_BACKEND_CPU,
        .parallel = true,
    };
#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)
    tune->n_profiles++;
    // cpu init + gpu compute
    tune->conf[1][0] = (struct ggml_task_conf){
        .backend = GGML_BACKEND_CPU,
        .parallel = true,
    };
    tune->conf[1][1] = (struct ggml_task_conf){
        .backend = tune->gpu_backend,
        .wait = true,
    };
#elif defined(GGML_USE_CUBLAS) || defined(GGML_USE_CLBLAST)
    tune->n_profiles++;
    // gpu only: compute
    tune->conf[1][1] = (struct ggml_task_conf){
        .backend = tune->gpu_backend,
        .wait = true,
    };
#else
    abort();
#endif
}
```

## Limitations and TODOs

- Only tested models 7B and 13B with type Q4_0.
- TODO: support Q5_0, Q5_1, Q8_0, ...
- My OS/device can not use cuBLAS, so can not generate example bench result.
  TODO: evaluate performance and energy for cuBlas and ClBlast.
- Anti-inconsistency is not implemented.
  TODO: validate model, type, stage settings.
- The codes are in draft state, with many shortcomings and TODOs.
- Big change, hard to review, evaluate and merge.

## How to Evaluate

**Build**

- Accelerate:
  ```
  make clean; LLAMA_NO_ACCELERATE= make
  ```
- OpenBLAS
  ```
  make clean; LLAMA_NO_ACCELERATE=1 LLAMA_OPENBLAS=1 make
  ```

**Bench**

```
./mulmat-tune help
usage: ./mulmat-tune [bench ...] | [analyze FILE] | test | help

bench [-m MODEL] [-t TYPE] [-f FILE] [-y]
-model  MODEL  7B | 13B | 30B | 64B
               default 7B
-type   TYPE   Q4_0 |  Q4_1 | Q5_0 | Q5_1 | Q8_0 | ...
               default Q4_0
-m_step M_STEP the step of M, also as start value
               suggest M_STEP %% 8 == 0
               default 8
-m_num  M_NUM  number of M, total M = M_STEP * M_NUM
               default 16
-file   FILE   data file to write
               default stdout
-y             always answer "yes" to all prompts
```

NOTE: at present, only supports model 7B/13B, type Q4_0/Q4_1.

Examples:

```
# run with default params (7B, Q4_0, ...)
./mulmat-tune

# to run 13B and Q4_1 with alway-yes
./mulmat-tune bench -model 13B -type Q4_1 -y

# customized m_step (16 * 8)
./mulmat-tune bench -model 7B -m_step 16 -num_m 8

# save to file
./mulmat-tune bench -model 7B -file mulmat-tune.txt

# analyze
./mulmat-tune analyze mulmat-tune.txt
```

Apart from the default `bench` sub-command, you MAY want to run `analyze` to see
the comprehensive analysis, such as affect of `n_threads`.

## Evaluate in main

Firstly, prepare the bench file for your model. SHOULD run in idle and cool status.
Place `mulmat-tune.txt` into the llama.cpp dir.

Then run your favorite program: main, perplexity, etc.
The program will print debug log when or when not found the file.

I suggest do not use too many threads, 4 threads almost works best on my 5-year
old device with total 12 cores (6 physical cores).

## Bench Data Format

**Example**

```
1 7B Q4_0 3 ACCELERATE 4 8 3 2
1 0 0 1 1 0 0 0 0
1 1 0 3 0 1 0 0 0
4096 4096
  8        8     7477 0    12438     6448 0
 16       22    14387 0    12468     7173 0
 24       31    18584 0    10948     6641 0
4094 11008
  8       24    16279 0    29243     8578 0
 16       48    32857 0    29370     9313 0
 24       73    51046 0    29593     9684 0
11008 4096
  8        9    16783 0    28571    20398 0
 16       17    32989 0    28463    21112 0
 24       25    48836 0    28881    21928 0
32000 4096
  8        8    49100 0    83752    63537 0
 16       18    97093 0    83577    66874 0
 24       24   141918 0    83565    71988 0
 ```

See files in dir [bench-out](bench-out) for details.

These files are generated on Macbook Pro 2018:

- OS: macOS Version 13.3.1 (a)
- Memory: 32 GB 2400 MHz DDR4
- CPU: 2.6 GHz 6-Core Intel Core i7-8850H @2.60GHz, with integrated Intel UHD Graphics 630 1536 MB

**Informal Explanation**

```
head
groups+

head := version model type gpu_backend gpu_backend_name n_shapes m_step num_m n_profiles
task_conf_profile+
shape
bench_item+

# head
version: 1
model: "7B" | "13B" | "30B" | "65B"
type: "Q4_0" | "Q4_1" | "Q5_0" | "Q5_1" | "Q8_0" | ...
gpu_backend: 1 | 2 | 3 | 4
gpu_backend_name: "CUDA" | "ACCELERATE" | "OPENBLAS" | "CLBLAST"
n_shapes: 4
m_step: 8
m_num: 16

task_conf_profile: stage_conf(init) stage_conf(compute) stage_conf(finalize)
stage_conf: backend parallel wait
backend: 0 | 1 | 2 | 3 | 4
parallel: 0 | 1
wait: 0 | 1

shape := N K

bench_item: M profile_time+
profile_time := stage_time[3]
stage_time[3]: init_time, compute_time, finalize_time
```

Time unit is `us`. A column is all zeros when that stage does not exist.

## How To Estimate Execution Time

For Accelerate/OpenBLAS, mul_mat_q_f32, the init stage is run on CPU, and is
paralleled in this CL. The `cpu-only` approach runs init stage with 1 threads and
compute stage in N threads. The `with_gpu` approach runs init stage in CPU with
N threads, and run compute stage in GPU with 1 thread.

For any given M/N/K/n_threads, we interpolate time for M between two `M`s.
When no bench to load or M out of range, fall back to original algorithm:
when M >= 32, change n_threads to 1 and use blas.

For example, given cpu_compute_m_8 and cpu_compute_m_16, for single thread:

```
cpu_compute_m_12 = (cpu_compute_m_8 + cpu_compute_m_16) / 2
cpu_compute_m_10 = cpu_compute_m_8 + (10 - 8) / ( 16 - 8) * (cpu_compute_m_16 - cpu_compute_m_8)
                 = cpu_compute_m_8 + (cpu_compute_m_16 - cpu_compute_m_8) / 4
```

For any thread number `nth`:

```
cpu_only_time = cpu_init_time + cpu_compute_time / nth
with_gpu_time = cpu_init_time / nth + gpu_compute_time
```

Backend-planning is executed at the very beginning of `ggml_graph_compute()`.
Generally speaking, at most 16 data records is enough with max M = 8 * 16 = 128.
So the total plan time is very small: tens of us. When use a simple cache, the
time goes down to about 10 us.

## Wait-Notify Overhead

Each call is about 10 us, may vary 5x. Since every mul_mat that run with-gpu
takes several ms to hundreds of ms, and the average boost is large, so the
wait-notify overhead is acceptable. 

## High Level Guide to Code Review

**Major Changes**

- examples/mulmat-tune provides the tool, data file format and data
  structure/APIs for graph compute. Some of them are expected be integrated
  into ggml.c/ggml.h.
- ggml.h: exposes a test function for mulmat-tune-bench.c; new fields and structs.
- ggml.c: new threading framework, update to `ggml_compute_forward_mul_mat()`.
  Defined several macros for debugging.
- llama.cpp: temp work-round to load mulmat tune file named `mulmat-tune.txt`.
- ggml-opencl.c fixed OOM error -- for local testing only. Changes are not quite
  relevant to this pull request. The OpenCL + ClBlast (from macOS and Homebew)
  performs 5-10x slower than Accelerate on my device. Not familiar with how to
  fine tune ClBlast.

**Assumed Merge RoadMap**

I assume we agree that:

1. discuss, improve, make decisions.
2. If this CL makes sense, then split CLs and merge step by step, use feature flag.

Here is the assumed merge steps:

1. Add experimental mulmat tune tool.
2. define feature flag `GGML_GRAPH_COMPUTE_V2`, apply the task config framework,
   keep `tensor.n_tasks` for v1.
3. Adjust `ggml_compute_forward_mul_mat` for Accelerate/OpenBLAS: add new
   init stage that can parallel. Make existing compute functions compatible to
   incoming `ggml_graph_compute v2`.
4. Keep current implementation of `ggml_graph_compute`, add the experimental `v2`.
5. Add command line args for v2: --mulmat-tune, load offline bench data from file.
6. Optional: support bench on startup -- this can solve various problems caused
   by mismatching between llama/mulmat-tune, llama/data.

