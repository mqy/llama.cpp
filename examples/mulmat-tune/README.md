# Fine tune MUL_MAT Qx_x with Bench

## Introduction

When either Accelerate or OpenBLAS was enabled, for long prompt tokens (M >=32),
graph compute is run with 1 thread.

Things to improve:

1. Run graph compute for Accelerate or OpenBLAS with N threads.
2. Parallel task initialization stage for Accelerate/OpenBLAS.

Observations on my device.

1. De-quantization in CPU takes fixed time for given N x K combination.
2. GPU out-performs CPU for big workload (for example M = 32).
   When workload doubles, the GPU time almost no change or steadily grows.
   With given N/K and 1-thread, the line curves of GPU time may go down at some
   place. In additional, the use-blas time may out-performs cpu-only at M-16.
3. De-quantization time takes significant time: >= 1/2 for M <= 128, about 1/10
   for M == 512.
4. Accelerate performs good than OpenBLAS (about -10%).
   To test ClBLAST, I fixed ggml-opencl.c.

Solutions:

1. Improve threading infrastructure: combine `spin` + `pthread cond-wait/broadcast`.
2. Extend task stage config: allow parallel any stage, and allow configure wait.
3. A tool for collecting execution times: model, Qxx, M/N/K, cpu-only/use-blas.
4. The bench result can be loaded into llama for comparing run time of `cpu-only`
   and `use-blas` with given M/N/K/n_threads, then choose the best execution plan.

## Benefits

1. A experimental show case of fine tune.
2. Prompt eval time may decrease up to -40%.
3. Try balance between best performance and energy at given number of thread.
4. Complicated but may be better threading infrastructure for flexibility, COULD
   be extended to run de-quantization in CPU for cuBLAS or CLBlast.

## Limitations

- Only tested 7B and 13B.
- Only tested Q4_0.
- OS can not use cuBLAS.
- The codes are in draft state, with many TODOs.
- Big change, hard to review, evaluate and merge.

## Examples

The token eval time almost same. But prompt eval time may decrease pretty much.

**chat-with-bob 7B Q4_0**

```
./main -m ./models/7B/ggml-model-q4_0.bin -c 512 -b 1024 -n 256 --keep 48 -t 4 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt
```

Prompt eval time:

* -20% (2 threads)
* -40% (4 threads) 

**7B wiki.valid.raw**

```
./perplexity -m models/7B/ggml-model-q4_0.bin -f ./models/wikitext-2-raw/wiki.valid.raw --mlock
```

Prompt eval time:

* -9% (2 threads)
* -13% (4 threads)

**prompt.sh**

```
./examples/mulmat-tune/prompt.sh -b -f
```

10% ~ 18% decrease with 2 or 4 threads. 

Run ```./examples/mulmat-tune/prompt.sh -h``` for help.

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
-model  MODEL   7B | 13B | 30B | 64B
                default 7B
-q_type TYPE    Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1
                default Q4_0
-step_m STEP_M  the step of M, also as start value
                suggest STEP_M %% 8 == 0
                default 8
-num_m  NUM_M   number of M, total M = STEP_M * NUM_M
                default 16
-file   FILE    data file to write
                default stdout
-y              always answer "yes" to all prompts
```

NOTE: at present, only supports model 7B/13B, q_type Q4_0/Q4_1.

Examples:

```
# run with default params (7B, Q4_0, ...)
./mulmat-tune

# to run 13B and Q4_1 with alway-yes
./mulmat-tune bench -model 13B -q_type Q4_1 -y

# customized step_m (16 * 8)
./mulmat-tune bench -model 13B -step_m 16 -num_m 8

# save to file
./mulmat-tune bench -model 13B -file mulmat-tune.13b.txt
```

Apart from the default "bench" sub-command, you MAY want to run "analyze" to see
the estimated `cpu-only` and `use-blas` time per M/N/K/n_threads. It's good to draw
them, interesting!

## How to Evaluate

1. Prepare the bench file for your model. SHOULD run in idle and cool status.
2. Put the file into the llama.cpp dir.
   File name pattern: ```mulmat-tune.<MODEL>.txt```, where ```<MODEL>``` is "7b"
   or "13b" -- NOTE: lower case.
3. Run your favorite program: main, perplexity, etc. The program will print debug
   log when or when not found the file.
   I suggest do not use too many threads, 4 threads almost works best on my
   5-year old device with total 12 cores (6 physical cores).

## Bench Data Format

**Example**

```
1 7B Q4_0 Accelerate 4 8 2 1 3 0 3 2 0
4096 4096
  8      46    9329 0   6401   6284 0
 16      93   19410 0   6056   6433 0
4096 11008
  8     111   20209 0  14990   9901 0
 16     265   39140 0  15022  10543 0
11008 4096
  8      44   22161 0  15264  22031 0
 16      90   38207 0  15059  22912 0
32000 4096
  8        8  48768 0  83870  60724 0
 16       16  97744 0  83893  63991 0
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

head := version model q_type blas_name num_groups m_step num_m stage_flags(first 3 for cpu-only, rest 3 for use-blas)

model: "7B" | "13B" | "30B" | "65B"

q_type: "Q4_0" | "Q4_1" | "Q5_0" | "Q5_1" | "Q8_0" | "Q8_1"

blas_name: "Accelerate" | "OpenBLAS" | "cuBLAS" | "clBlast"

stage_flags := init_flag compute_flag finalize_flag

groups := group+

group := N K M_record+

M_record := M stage_times(first 3 for cpu-only, rest 3 for use-blas)

stage_times := init_time compute_time finalize_time
```

Enum of stage flag value:
- 0: stage not exists
- 1: single thread
- 2: single thread + worker idle wait
- 3: multi threads

Time unit is `us`. A column is all zeros when that stage does not exist.

## How To Estimate Execution Time

For Accelerate/OpenBLAS, mul_mat_q_f32, the init stage is run on CPU, and is
paralleled in this CL. The `cpu-only` approach runs init stage with 1 threads and
compute stage in N threads. The `use-blas` approach runs init stage in CPU with
N threads, and run compute stage in GPU with 1 thread.

For any given M/N/K/n_threads, we interpolate time for M between two `M`s.
When M out of range, fall back to original algorithm: use blas when M >= 32.

For example, given cpu_compute_m_8 and cpu_compute_m_16, for single thread:

```
cpu_compute_m_12 = (cpu_compute_m_8 + cpu_compute_m_16) / 2
cpu_compute_m_10 = cpu_compute_m_8 + (10 - 8) / ( 16 - 8) * (cpu_compute_m_16 - cpu_compute_m_8)
                 = cpu_compute_m_8 + (cpu_compute_m_16 - cpu_compute_m_8) / 4
```

For any thread number `nth`:

```
cpu_only_time = cpu_init_time + cpu_compute_time / nth
use_blas_time = cpu_init_time / nth + blas_compute_time
```

Plan blas is executed at the very beginning of ```ggml_compute_graph```.
Generally speaking, at most 16 data records is enough with max M = 8 * 16 = 128.
So the total plan time is very small: tens of us. When use a simple cache, the
time goes down to about 10 us.

## Wait-Notify Overhead

Each call is about 10 us, may vary 5x. Since every mul_mat that run with blas
takes several ms to hundreds of ms, and the average boost is large, so the
wait-notify overhead is acceptable. 

## High Level Guide to Code Review

Major changes:

1. examples/mulmat-tune provides the tool, data file format and data
   structure/APIs for graph compute. Some of them are expected be integrated
   into ggml.c/ggml.h.
2. ggml.h: exposes a test function for mulmat-tune-bench.c; new fields and structs.
3. ggml.c: threading, update to ggml_compute_forward_q_xxx.
4. Makefile: minor changes.
5. ggml-opencl.c for tests only, performs bad on my device. Changes are not quite
   relevant to this pull request.

Also defined several macros for debugging.

I'm new to machine learning this year and have almost no knowledge of AI models
and algorithms. There must be a lot of problems with this CL, please do not
hesitate to advise. Thanks!
