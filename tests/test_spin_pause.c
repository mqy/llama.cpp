#include <stdatomic.h>
#include <emmintrin.h>

static inline void test_pause(void) {
#if defined(__SSE2__)
    _mm_pause();
#elif defined(__i386__) || defined(__x86_64__)
    __asm__ __volatile__ ("pause");
#elif defined(__aarch64__)
    __asm__ __volatile__ ("wfe");
#endif
}

static atomic_int flag = 0;

// gcc -O3 -std=c11 test_pause.c -o test_pause && time ./test_pause
// gcc -O3 -std=c11 test_pause.c -o test_pause && time ./test_pause -DSPIN_HINT
int main() {
    const int n = 1000*1000*1000;
    for (int i=0; i<n; i++) {
        int v = atomic_load(&flag);
        if (v > 0) {} else if (v < 0) {}
 
#ifdef SPIN_HINT
        pin_hint();
#endif
    }
    return 0;
}
