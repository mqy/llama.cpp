## Bench Data Format

### Example data

```
1 7B Accelerate 3 8 2 1 3 0 3 1 0
4096 4096
  8      46    9329 0   6401   6284 0
 16      93   19410 0   6056   6433 0
4096 11008
  8     111   20209 0  14990   9901 0
 16     265   39140 0  15022  10543 0
11008 4096
  8      44   22161 0  15264  22031 0
 16      90   38207 0  15059  22912 0
 ```

See files in [example-bench-results](example-bench-results) for details.

These files are generated on Macbook Pro 2018:

- OS: macOS Version 13.3.1 (a)
- Memory: 32 GB 2400 MHz DDR4
- CPU: 2.6 GHz 6-Core Intel Core i7-8850H @2.60GHz, with integrated Intel UHD Graphics 630 1536 MB

### Informal explain of data format

```
head
groups+

head := version model blas_name num_groups m_step num_m stage_flags(first 3 for cpu, rest 3 for gpu)

model: "7B" | "13B" | "30B" | "65B"

blas_name: "Accelerate" | "OpenBLAS" | "cuBLAS" | "clBlast"

stage_flags := init_flag compute_flag finalize_flag

groups := group+

group := N K M_record+

M_record := M stage_times(first 3 for cpu, rest 3 for gpu)

stage_times := init_time compute_time finalize_time
```

Enum of stage flag value:
- 0: stage not exists
- 1: stage exists
- 3: stage exists and can parallel

Time unit is us. A column is all zeros when that stage does not exist.

### Limitations

Due to device capability:

- Only tested "7B" and "13B".
- I can not test cuBLAS.

### TODOs

- get NKs for 30B and 65B, support both models. 
- support CLBLAS
- ...
