#include <stdio.h>
#include <inttypes.h>

#ifndef tile_size_x
  #define tile_size_x 1
#endif

#ifndef block_size_x
  #define block_size_x 512
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
 * This kernel computes the correlated hits of hits no more than 1500 apart in both directions.
 * It does this using a 1-dimensional mapping of threads and thread blocks to hits in this time slice.
 *
 * This kernel supports the usual set of optimizations, including tiling, partial loop unrolling, read-only cache. 
 * Tuning parameters supported are 'read_only' [0,1], 'tile_size_x' divisor of 1500, and 'block_size_x' multiple of 32.
 *
 * 'write_sums' can be set to [0,1] to enable the code that
 * produces a different output, namely the number of correlated
 * hits per row. This number is used to compute the offsets into the sparse matrix
 * representation of the correlations table.
 *
 */
__global__ void quadratic_difference_full(int *__restrict__ col_idx, const int *__restrict__ prefix_sums, int *__restrict__ sums,
        int N, int sliding_window_width, const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
        const float *__restrict__ ct) {

    int i = threadIdx.x;
    int bx = blockIdx.x * block_size_x * tile_size_x;

    __shared__ float sh_ct[block_size_x * tile_size_x + window_width];
    __shared__ float sh_x[block_size_x * tile_size_x + window_width];
    __shared__ float sh_y[block_size_x * tile_size_x + window_width];
    __shared__ float sh_z[block_size_x * tile_size_x + window_width];

        //the first loading phase
        #pragma unroll
        for (int k=i; k < block_size_x*tile_size_x+window_width; k+=block_size_x) {
            if (bx+k-window_width >= 0 && bx+k-window_width < N) {
                sh_ct[k] = LDG(ct,bx+k-window_width);
                sh_x[k] = LDG(x,bx+k-window_width);
                sh_y[k] = LDG(y,bx+k-window_width);
                sh_z[k] = LDG(z,bx+k-window_width);
            } else {
                sh_ct[k] = 0.0f;
                sh_x[k] = 0.0f;
                sh_y[k] = 0.0f;
                sh_z[k] = 0.0f;
            }
        }
        __syncthreads();

        //start of the the computations phase
        float l_ct[tile_size_x];
        float l_x[tile_size_x];
        float l_y[tile_size_x];
        float l_z[tile_size_x];
        #if write_sums == 1
        int sum[tile_size_x];
        #endif

        //keep the most often used values in registers
        for (int ti=0; ti<tile_size_x; ti++) {
            l_ct[ti] = sh_ct[i+ti*block_size_x+window_width];
            l_x[ti] = sh_x[i+ti*block_size_x+window_width];
            l_y[ti] = sh_y[i+ti*block_size_x+window_width];
            l_z[ti] = sh_z[i+ti*block_size_x+window_width];
            #if write_sums == 1
            sum[ti] = 0;
            #endif
        }


        //first loop computes correlations with earlier hits

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
            for (int j=0; j < window_width; j++) {

                #pragma unroll
                for (int ti=0; ti<tile_size_x; ti++) {

                        float diffct = l_ct[ti] - sh_ct[i+ti*block_size_x+j];
                        float diffx  = l_x[ti] - sh_x[i+ti*block_size_x+j];
                        float diffy  = l_y[ti] - sh_y[i+ti*block_size_x+j];
                        float diffz  = l_z[ti] - sh_z[i+ti*block_size_x+j];

                        if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                            //uint64_t pos = (j-1) * ((uint64_t)N) + (bx+i+ti*block_size_x);
                            //correlations[pos] = 1;
                            #if write_sums == 1
                            sum[ti] += 1;
                            #endif
                        }

                }

            }


        //make sure all threads are done with phase-1
        __syncthreads();

        //start load phase-2
        //fill the first part of shared memory with data already in registers
        #pragma unroll
        for (int ti=0; ti<tile_size_x; ti++) {
            sh_ct[i+ti*block_size_x] = l_ct[ti];
            sh_x[i+ti*block_size_x] = l_x[ti];
            sh_y[i+ti*block_size_x] = l_y[ti];
            sh_z[i+ti*block_size_x] = l_z[ti];
        }

        //the first block_size_x*tile_size_x part has already been filled
        #pragma unroll
        for (int k=block_size_x*tile_size_x+i; k < block_size_x*tile_size_x+window_width; k+=block_size_x) {
            if (bx+k < N) {
                sh_ct[k] = LDG(ct,bx+k);
                sh_x[k] = LDG(x,bx+k);
                sh_y[k] = LDG(y,bx+k);
                sh_z[k] = LDG(z,bx+k);
            } else {
                sh_ct[k] = 0.0f;
                sh_x[k] = 0.0f;
                sh_y[k] = 0.0f;
                sh_z[k] = 0.0f;
            }
        }
        __syncthreads();

        //the next loop computes correlations with hits later in time

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
                            //uint64_t pos = (j-1) * ((uint64_t)N) + (bx+i+ti*block_size_x);
                            //correlations[pos] = 1;
                            #if write_sums == 1
                            sum[ti] += 1;
                            #endif
                        }

                }

            }


        #if write_sums == 1
        for (int ti=0; ti<tile_size_x; ti++) {
            sums[bx+i+ti*block_size_x] = sum[ti];
        }
        #endif

}

