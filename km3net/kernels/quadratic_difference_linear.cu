#include <stdio.h>

#include <inttypes.h>

#ifndef tile_size_x
  #define tile_size_x 1
#endif

#ifndef block_size_x
  #define block_size_x 512
#endif

#ifndef block_size_y
  #define block_size_y 1
#endif

#ifndef window_width
#define window_width 1500
#endif

#define USE_READ_ONLY_CACHE read_only
#if USE_READ_ONLY_CACHE == 1
#define LDG(x, y) __ldg(x+y)
#elif USE_READ_ONLY_CACHE == 0
#define LDG(x, y) x[y]
#endif

#ifndef write_sums
#define write_sums 0
#endif


/*
 * This kernel computes the correlated hits of hits no more than 1500 apart.
 * It does this using a 1-dimensional mapping of threads and thread blocks.
 *
 * This kernel supports the usual set of optimizations, including tiling, partial loop unrolling, read-only cache. 
 * Tuning parameters supported are 'read_only' [0,1], 'tile_size_x' divisor of 1500, and 'block_size_x' multiple of 32.
 *
 * 'write_sums' can be set to [0,1] to enable the code that
 * produces another output, namely the number of correlated
 * hits per row. This number is later to create the sparse
 * representation of the correlations table. If not using
 * sparse respresentation set write_sums to 0.
 */
__global__ void quadratic_difference_linear(char *__restrict__ correlations, int *sums, int N, int sliding_window_width,
        const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z, const float *__restrict__ ct) {

    int tx = threadIdx.x;
    int bx = blockIdx.x * block_size_x * tile_size_x;

    __shared__ float sh_ct[block_size_x * tile_size_x + window_width];
    __shared__ float sh_x[block_size_x * tile_size_x + window_width];
    __shared__ float sh_y[block_size_x * tile_size_x + window_width];
    __shared__ float sh_z[block_size_x * tile_size_x + window_width];

    if (bx+tx < N) {

        //the loading phase
        for (int k=tx; k < block_size_x*tile_size_x+window_width; k+=block_size_x) {
            if (bx+k < N) {
                sh_ct[k] = LDG(ct,bx+k);
                sh_x[k] = LDG(x,bx+k);
                sh_y[k] = LDG(y,bx+k);
                sh_z[k] = LDG(z,bx+k);
            }
        }
        __syncthreads();

        //start of the the computations phase
        int i = tx;
        float l_ct[tile_size_x];
        float l_x[tile_size_x];
        float l_y[tile_size_x];
        float l_z[tile_size_x];
        #if write_sums == 1
        int sum[tile_size_x];
        #endif

        //keep the most often used values in registers
        for (int ti=0; ti<tile_size_x; ti++) {
            l_ct[ti] = sh_ct[i+ti*block_size_x];
            l_x[ti] = sh_x[i+ti*block_size_x];
            l_y[ti] = sh_y[i+ti*block_size_x];
            l_z[ti] = sh_z[i+ti*block_size_x];
            #if write_sums == 1
            sum[ti] = 0;
            #endif
        }

        //small optimization to eliminate bounds checks for most blocks
        if (bx+block_size_x*tile_size_x+window_width < N) {

            //unfortunately there's no better way to do this right now
            //[1, 2, 3, 4, 5, 6, 10, 12, 15]
            #if f_unroll == 2
            #pragma unroll 2
            #elif f_unroll == 3
            #pragma unroll 3
            #elif f_unroll == 4
            #pragma unroll 4
            #elif f_unroll == 5
            #pragma unroll 5
            #elif f_unroll == 6
            #pragma unroll 6
            #elif f_unroll == 10
            #pragma unroll 10
            #elif f_unroll == 12
            #pragma unroll 12
            #elif f_unroll == 15
            #pragma unroll 15
            #endif            
            for (int j=1; j < window_width+1; j++) {

                #pragma unroll
                for (int ti=0; ti<tile_size_x; ti++) {

                        float diffct = l_ct[ti] - sh_ct[i+ti*block_size_x+j];
                        float diffx  = l_x[ti] - sh_x[i+ti*block_size_x+j];
                        float diffy  = l_y[ti] - sh_y[i+ti*block_size_x+j];
                        float diffz  = l_z[ti] - sh_z[i+ti*block_size_x+j];

                        if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                            uint64_t pos = (j-1) * ((uint64_t)N) + (bx+i+ti*block_size_x);
                            correlations[pos] = 1;
                            #if write_sums == 1
                            sum[ti] += 1;
                            #endif
                        }

                }

            }

        }
        //same as above but with bounds checks for last few blocks
        else {

            //unfortunately there's no better way to do this right now
            //[1, 2, 3, 4, 5, 6, 10, 12, 15]
            #if f_unroll == 2
            #pragma unroll 2
            #elif f_unroll == 3
            #pragma unroll 3
            #elif f_unroll == 4
            #pragma unroll 4
            #elif f_unroll == 5
            #pragma unroll 5
            #elif f_unroll == 6
            #pragma unroll 6
            #elif f_unroll == 10
            #pragma unroll 10
            #elif f_unroll == 12
            #pragma unroll 12
            #elif f_unroll == 15
            #pragma unroll 15
            #endif            
            for (int j=1; j < window_width+1; j++) {

                for (int ti=0; ti<tile_size_x; ti++) {

                    if (bx+i+ti*block_size_x+j < N) {

                        float diffct = l_ct[ti] - sh_ct[i+ti*block_size_x+j];
                        float diffx  = l_x[ti] - sh_x[i+ti*block_size_x+j];
                        float diffy  = l_y[ti] - sh_y[i+ti*block_size_x+j];
                        float diffz  = l_z[ti] - sh_z[i+ti*block_size_x+j];

                        if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                            uint64_t pos = (j-1) * ((uint64_t)N) + (bx+i+ti*block_size_x);
                            correlations[pos] = 1;
                            #if write_sums == 1
                            sum[ti] += 1;
                            #endif
                        }

                    }
                }



            }

        }

        #if write_sums == 1
        for (int ti=0; ti<tile_size_x; ti++) {
            sums[bx+i+ti*block_size_x] = sum[ti];
        }
        #endif

    }
}


/*
 * This is the old kernel that uses a 2D thread block layout, mainly kept here to verify the correctness of the linear kernel
 */
__global__ void quadratic_difference(int8_t *correlations, int N, int sliding_window_width, float *x, float *y, float *z, float *ct)
{
    int i = blockIdx.x * block_size_x + threadIdx.x;
    int j = blockIdx.y * block_size_y + threadIdx.y;

    int l = i + j;
    if (i < N && j < sliding_window_width) { 

    uint64_t pos = j * (uint64_t)N + (uint64_t)i;

    if (l >= N){
        return;
    }

    float diffct = ct[i] - ct[l];
    float diffx  = x[i] - x[l];
    float diffy  = y[i] - y[l];
    float diffz  = z[i] - z[l];

    if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) { 
      correlations[pos] = 1;
    }

    }
}


