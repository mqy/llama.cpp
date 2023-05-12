```
./clblast_tuner_xgemm -precision 32 -device 1 -n 4096 -k 4096 -alpha 1.0 -beta 0.0
* (1/4) Tuning main GEMM kernel (GEMMK == 0) for fixed set of parameters

* Options given/available:
    -platform 0 [=default]
    -device 1 
    -precision 32 (single) [=default]
    -m 1024 [=default]
    -n 4096 
    -k 4096 
    -alpha 1.00 
    -beta 0.00 
    -fraction 1.00 [=default]
    -runs 2 [=default]
    -max_l2_norm 0.00 [=default]
```