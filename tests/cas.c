#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>

static atomic_int counter = 0;
static atomic_flag cas_lock = ATOMIC_FLAG_INIT;

const int n_threads = 6;
const int n_loops = 1000*1000;

void * runner(void *data) {
	for (int i = 0; i < n_loops; i++) {
		while (atomic_flag_test_and_set(&cas_lock)){}
		atomic_fetch_add(&counter, 1);
		atomic_flag_clear(&cas_lock);
	}
	return NULL;
}

// gcc -O3 -std=c11 -o cas tests/cas.c
int main(void) {
	pthread_t ids[n_threads];

	for (int i=0; i<n_threads; i++) {
		pthread_create(&ids[i], NULL, runner, NULL);
	}

	for (int i=0; i<n_threads; i++) {
		pthread_join(ids[i], NULL);
	}

	if (counter != n_loops * 6) {
		printf("failed\n%d\n%d\n", counter, n_loops);
	}
	return 0;
}
