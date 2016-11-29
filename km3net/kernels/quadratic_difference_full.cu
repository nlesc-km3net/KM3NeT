#include <stdio.h>
#include <inttypes.h>
#include <float.h>

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

#ifndef write_spm
#define write_spm 0
#endif

#ifndef write_rows
#define write_rows 0
#endif

#ifndef use_shared
#define use_shared 0
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
__global__ void quadratic_difference_full(int *__restrict__ row_idx, int *__restrict__ col_idx, const int *__restrict__ prefix_sums, int *__restrict__ sums,
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
            sh_ct[k] = -FLT_MAX;  //this values ensures out-of-bound hits won't be correlated
            sh_x[k] = 0.0f;
            sh_y[k] = 0.0f;
            sh_z[k] = 0.0f;
        }
    }

    #if write_spm == 1
    int offset[tile_size_x];
    if (bx+i==0) {
        offset[0] = 0;
    }
    #pragma unroll
    for (int ti=0; ti<tile_size_x; ti++) {
        if (bx+i+ti*block_size_x-1 > 0 && bx+i+ti*block_size_x-1 < N) {
            offset[ti] = prefix_sums[bx+i+ti*block_size_x-1];
        }
    }
    #endif

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
    #pragma unroll
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
                        #if write_spm == 1
                        #if write_rows
                        row_idx[offset[ti]] = bx+i+ti*block_size_x; 
                        #endif
                        col_idx[offset[ti]] = bx+i+ti*block_size_x-window_width+j;
                        offset[ti] += 1;
                        #endif
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
                            #if write_spm == 1
                            #if write_rows
                            row_idx[offset[ti]] = bx+i+ti*block_size_x;
                            #endif
                            col_idx[offset[ti]] = bx+i+ti*block_size_x+j;
                            offset[ti] += 1;
                            #endif
                            #if write_sums == 1
                            sum[ti] += 1;
                            #endif
                        }

                }

            }


        #if write_sums == 1
        for (int ti=0; ti<tile_size_x; ti++) {
            if (bx+i+ti*block_size_x < N) {
                sums[bx+i+ti*block_size_x] = sum[ti];
            }
        }
        #endif
}


#ifndef shared_memory_size
#define shared_memory_size 10*block_size_x
#endif

/*
 * This kernel uses warp-shuffle instructions to re-use many of
 * the input values in registers and reduce the pressure on shared memory.
 * However, it does this so drastically that shared memory is hardly needed anymore.
 *
 * Tuning parameters supported are 'block_size_x', 'read_only' [0,1], 'use_if' [0,1]
 *
 */
__global__ void quadratic_difference_full_shfl(int *__restrict__ row_idx, int *__restrict__ col_idx, int *__restrict__ prefix_sums, int *sums, int N, int sliding_window_width,
        const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z, const float *__restrict__ ct) {

    int tx = threadIdx.x;
    int bx = blockIdx.x * block_size_x;

    #if write_sums == 1
    int sum = 0;
    #endif
    #if write_spm == 1
    int offset = 0;


    #if use_shared == 1
    int block_start = 0;
    __shared__ int sh_col_idx[shared_memory_size];
    if (blockIdx.x > 0) {
        block_start = prefix_sums[bx-1];
    }
    #elif write_rows == 1
    int block_start = 0;
    #endif

    #endif

    float ct_i = 0.0f;
    float x_i = 0.0f;
    float y_i = 0.0f;
    float z_i = 0.0f;

    int output = 0;
    int i = bx + tx - window_width;

    if (bx+tx < N) {
        output = 1;
        ct_i = LDG(ct,bx+tx);
        x_i = LDG(x,bx+tx);
        y_i = LDG(y,bx+tx);
        z_i = LDG(z,bx+tx);
    }

        #if write_spm == 1
        if (bx+tx > 0 && bx+tx < N) {
            offset = prefix_sums[bx+tx-1];
        }
        #if use_shared == 1
        offset -= block_start;
        #endif
        #endif


            int laneid = tx & (32-1);
            if (output) {
            for (int j=0; j < 32-laneid && output; j++) {
                if (i+j >= 0 && i+j<N) {

                float diffct = ct_i - LDG(ct,i+j);
                float diffx = x_i - LDG(x,i+j);
                float diffy = y_i - LDG(y,i+j);
                float diffz = z_i - LDG(z,i+j);

                if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                    #if write_sums == 1
                    sum++;
                    #endif
                    #if write_spm == 1
                        #if write_rows
                        row_idx[offset + block_start] = bx+tx;
                        #endif

                        #if use_shared == 1
                            sh_col_idx[offset++] = i+j;
                        #else
                        col_idx[offset++] = i+j;
                        #endif
                    #endif

                }

                }
            }
            }//end of if output

            int j;
                
            #if f_unroll == 2
            #pragma unroll 2
            #elif f_unroll == 4
            #pragma unroll 4
            #endif
            for (j=32; j < window_width*2-32; j+=32) {

                float ct_j = 0.0f;
                float x_j = 0.0f;
                float y_j = 0.0f;
                float z_j = 0.0f;

                if (i+j >= 0 && i+j<N) {
                    ct_j = LDG(ct,i+j);
                    x_j = LDG(x,i+j);
                    y_j = LDG(y,i+j);
                    z_j = LDG(z,i+j);
                }

                for (int d=1; d<33; d++) {
                    ct_j = __shfl(ct_j, laneid+1);
                    x_j = __shfl(x_j, laneid+1);
                    y_j = __shfl(y_j, laneid+1);
                    z_j = __shfl(z_j, laneid+1);

                    float diffct = ct_i - ct_j;
                    float diffx  = x_i - x_j;
                    float diffy  = y_i - y_j;
                    float diffz  = z_i - z_j;

                    if (i+j >= 0 && i+j<N && output && (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz)) {
                        #if write_sums == 1
                        sum++;
                        #endif
                        #if write_spm == 1
                            #if write_rows
                            row_idx[offset + block_start] = bx+tx;
                            #endif

                            int c = laneid+d > 31 ? -32 : 0;
                            #if use_shared == 1
                                sh_col_idx[offset++] = i+j+d+c;
                            #else
                            col_idx[offset++] = i+j+d+c;
                            #endif
                        #endif
                    }

                }

            }

            if (output) {
            j-=laneid;
            for (; j < window_width*2+1; j++) {
                if (i+j >= 0 && i+j<N) {

                float diffct = ct_i - LDG(ct,i+j);
                float diffx = x_i - LDG(x,i+j);
                float diffy = y_i - LDG(y,i+j);
                float diffz = z_i - LDG(z,i+j);

                if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                    #if write_sums == 1
                    sum++;
                    #endif
                    #if write_spm == 1
                        #if write_rows
                        row_idx[offset + block_start] = bx+tx;
                        #endif
                        #if use_shared == 1
                            sh_col_idx[offset++] = i+j;
                        #else
                        col_idx[offset++] = i+j;
                        #endif
                    #endif
                }

                }
            }
            } // end of if output

    #if write_sums == 1
    if (bx+tx < N) {
        sums[bx+tx] = sum;
    }
    #endif

    //collaboratively write back the output collected in shared memory to global memory

    #if use_shared == 1 && write_spm == 1
    int block_stop = 0;
    int last_i = bx + block_size_x-1;
    if (last_i < N) {
        block_stop = prefix_sums[last_i];
    } else {
        block_stop = prefix_sums[N-1];
    }
    __syncthreads(); //ensure all threads are done writing shared memory
    for (int k=block_start+tx; k<block_stop; k+=block_size_x) {
        if (k-block_start >= 0 && k-block_start < shared_memory_size-1)
            col_idx[k] = sh_col_idx[k-block_start];
    }
    #endif

}


