#include <iostream>

// for testing
#include <chrono>
#include <iomanip>
#include <cmath>

#include <emmintrin.h>
#include <immintrin.h>

int main() {
	typedef std::chrono::high_resolution_clock hrClock;
	uint64_t N = 100000000;
	
	std::cout << std::setprecision(9);

  // calculate normally
	const auto start1 = hrClock::now();
	for (uint64_t i = 1; i < N; ++i) {
		volatile double x = sqrt(i);
	} const auto end1 = hrClock::now();

  // calculate using sse 128-bit
	const auto start2 = hrClock::now();
	bool trigger1 = false;
	for (uint64_t i = 1; i < N; ++i) {
		volatile __m128 x {};
		if (trigger1) {
			// amount of numbers left arent divisible by 4
      // volatime so compiler does not optimize
			volatile float y = sqrt(i);
			continue;
		}
		if (i % 4) {
			float y[4] __attribute__((aligned(16))) = {(float)(i-3),(float)(i-2),(float)(i-1),(float)(i)};
			_mm_store_ps(y,x);
			x = _mm_sqrt_ps(x);
		}
		if (i > N-3) trigger1 = true;
	} const auto end2 = hrClock::now();

  // calculate using avx 256-bit
	const auto start3 = hrClock::now();
	bool trigger2 = false;
	for (uint64_t i = 1; i < N; ++i) {
		volatile __m256 x {};
		if (trigger2) {
			volatile float y = sqrt(i);
			continue;
		}
		if (i % 8) {
			float y[8] __attribute__((aligned(32))) = {
				(float)(i-7),(float)(i-6),(float)(i-5),(float)(i-4),(float)(i-3),(float)(i-2),(float)(i-1),(float)(i)
			};
			_mm256_store_ps(y,x);
			x = _mm256_sqrt_ps(x);
		}
		if (i > N-7) trigger2 = true;
	} const auto end3 = hrClock::now();

  // calculate using avx 512-bit
	const auto start4 = hrClock::now();
	bool trigger3 = false;
	for (uint64_t i = 1; i < N; ++i) {
		volatile __m512 x {};
		if (trigger3) {
			volatile float y = sqrt(i);
			continue;
		}
		if (i % 16) {
			float y[16] __attribute__((aligned(64))) = {
				(float)(i-15),(float)(i-14),(float)(i-13),(float)(i-12),(float)(i-11),(float)(i-10),(float)(i-9),(float)(i-8),
				(float)(i-7) ,(float)(i-6) ,(float)(i-5) ,(float)(i-4) ,(float)(i-3) ,(float)(i-2) ,(float)(i-1),(float)(i)
			};
			_mm512_store_ps((void*)y,x);
			x = _mm512_sqrt_ps(x);
		}
		if (i > N-15) trigger3 = true;
	} const auto end4 = hrClock::now();

	const std::chrono::duration<double> t1 = end1 - start1;
	const std::chrono::duration<double> t2 = end2 - start2;
	const std::chrono::duration<double> t3 = end3 - start3;
	const std::chrono::duration<double> t4 = end4 - start4;
	
	std::cout 
		<< "calculation of sqrt(n) for n in 1 to " << N
		<< "\nNormal: " << t1 
		<< "\nSSE 128-bit: " << t2
		<< "\nAVX 256-bit: " << t3
		<< "\nAVX 512-bit: " << t4
		<< '\n';
	
	return 0;
}
