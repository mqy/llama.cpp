#include <stdatomic.h>
#include <emmintrin.h>

static inline void spin_hint(void) {
#if defined(__SSE2__)
    _mm_pause();
#elif defined(__i386__) || defined(__x86_64__)
    __asm__ __volatile__ ("pause");
#elif defined(__aarch64__)
    __asm__ __volatile__ ("wfe");
#endif
}

static atomic_int flag = 0;

// gcc -O3 -std=c11 -o spin_hint tests/spin_hint.c -DSPIN_HINT
// time ./spin_hint # 37s

// gcc -O3 -std=c11 -o spin_hint tests/spin_hint.c
// time ./spin_hint # 0.06s
int main() {
    const int n = 1000*1000*1000;
    for (int i=0; i<n; i++) {
        int v = atomic_load(&flag);
        if (v > 0) {} else if (v < 0) {}
 
#ifdef SPIN_HINT
        pin_hint();
#endif
    }
}