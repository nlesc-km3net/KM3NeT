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

extern "C" {
__global__ void quadratic_difference_full(int *__restrict__ row_idx, int *__restrict__ col_idx, const int *__restrict__ prefix_sums, int *__restrict__ sums,
        int N, int sliding_window_width, const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
        const float *__restrict__ ct);

__global__ void match3b_full(int *__restrict__ row_idx, int *__restrict__ col_idx, const int *__restrict__ prefix_sums, int *__restrict__ sums,
        int N, int sliding_window_width, const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
        const float *__restrict__ ct);
}

template<typename F>
__device__ void correlate_full(int *__restrict__ row_idx, int *__restrict__ col_idx, const int *__restrict__ prefix_sums, int *__restrict__ sums,
        int N, const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
        const float *__restrict__ ct, F criterion);

__forceinline__ __device__ bool match3b(float x1, float y1, float z1, float t1, float x2, float y2, float z2, float t2);
__forceinline__ __device__ bool quadratic_difference(float x1, float y1, float z1, float ct1, float x2, float y2, float z2, float ct2);



/*
 * This is the kernel used for computing correlations in both directions using the quadratic difference criterion
 */
__global__ void quadratic_difference_full(int *__restrict__ row_idx, int *__restrict__ col_idx, const int *__restrict__ prefix_sums, int *__restrict__ sums,
        int N, int sliding_window_width, const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
        const float *__restrict__ ct) {

    correlate_full(row_idx, col_idx, prefix_sums, sums, N, x, y, z, ct, quadratic_difference);

}

/*
 * This is the kernel used for computing correlations in both directions using the match 3b criterion
 */
__global__ void match3b_full(int *__restrict__ row_idx, int *__restrict__ col_idx, const int *__restrict__ prefix_sums, int *__restrict__ sums,
        int N, int sliding_window_width, const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
        const float *__restrict__ ct) {

    correlate_full(row_idx, col_idx, prefix_sums, sums, N, x, y, z, ct, match3b);

}


/*
 * This function fills shared memory will values from global memory
 *
 * The area loaded is equal to the working set of this thread block (block_size_x * tile_size_x) plus the window_width
 *
 * The threads in a thread block will load values in global memory from their global index 'i' up to block_size_x*tile_size_x+window_width
 * It is possible to modify which values from global memory are loaded by using the parameter 'offset'
 * The threads can skip the first x elements of shared memory by using a non zero value for 'start'
 * N is the total number of hits in the input, used to guard out-of-bound accesses
 *
 * first loading phase, start=0, offset=bx-window_width
 * second loading phase, start=block_size_x*tile_size_x, offset=bx
 */
__forceinline__ __device__ void fill_shared_memory(float *sh_ct, float *sh_x, float *sh_y, float* sh_z,
                                       const float *ct, const float *x, const float *y, const float *z,
                                                         int bx, int i, int start, int offset, int N) {
    #pragma unroll
    for (int k=start+i; k < block_size_x*tile_size_x+window_width; k+=block_size_x) {
        if (k+offset >= 0 && k+offset < N) {
            sh_ct[k] = LDG(ct,k+offset);
            sh_x[k] = LDG(x,k+offset);
            sh_y[k] = LDG(y,k+offset);
            sh_z[k] = LDG(z,k+offset);
        } else {
            sh_ct[k] = (float) NAN;  //this value ensures out-of-bound hits won't be correlated
            sh_x[k] = 0.0f;
            sh_y[k] = 0.0f;
            sh_z[k] = 0.0f;
        }
    }
}


/*
 * This function is responsible for looping over the iteration space of each thread
 * For each correlation to be computed it will call the criterion and either
 * store the number of correlations or the coordinates of the correlated hit.
 */
template<typename F>
__forceinline__ __device__ void correlate(int *row_idx, int *col_idx, int *sum, int *offset, int bx, int i,
                float *l_x, float *l_y, float *l_z, float *l_ct, float *sh_x, float *sh_y, float *sh_z, float *sh_ct, int col_offset, int it_offset, F criterion) {
    for (int j=it_offset; j < window_width+it_offset; j++) {

        #pragma unroll
        for (int ti=0; ti<tile_size_x; ti++) {

            bool condition = criterion(l_x[ti], l_y[ti], l_z[ti], l_ct[ti],
                    sh_x[i+ti*block_size_x+j], sh_y[i+ti*block_size_x+j],
                    sh_z[i+ti*block_size_x+j], sh_ct[i+ti*block_size_x+j]);

            if (condition) {
                #if write_spm == 1
                #if write_rows
                row_idx[offset[ti]] = bx+i+ti*block_size_x; 
                #endif
                col_idx[offset[ti]] = bx+i+ti*block_size_x+j+col_offset;
                offset[ti] += 1;
                #endif
                #if write_sums == 1
                sum[ti] += 1;
                #endif
            }

        }

    }

}



/*
 * This function computes the correlated hits of hits no more than 'window_width' apart in both directions.
 * It does this using a 1-dimensional mapping of threads and thread blocks to hits in this time slice.
 *
 * This function supports the usual set of optimizations, including tiling, read-only cache. 
 * Tuning parameters supported are 'read_only' [0,1], 'tile_size_x' any low number, and 'block_size_x' multiple of 32.
 *
 * 'write_sums' can be set to [0,1] to enable the code that outputs the number of correlated hits per hit
 * This number is used to compute the offsets into the sparse matrix representation of the correlations table.
 * 
 * 'write_spm' can be set to [0,1] to enable the code that outputs the sparse matrix
 * 'write_rows' can be set to [0,1] to enable also writing the row_idx, only effective when write_spm=1
 *
 */
template<typename F>
__device__ void correlate_full(int *__restrict__ row_idx, int *__restrict__ col_idx, const int *__restrict__ prefix_sums, int *__restrict__ sums,
        int N, const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
        const float *__restrict__ ct, F criterion) {

    int i = threadIdx.x;
    int bx = blockIdx.x * block_size_x * tile_size_x;

    __shared__ float sh_ct[block_size_x * tile_size_x + window_width];
    __shared__ float sh_x[block_size_x * tile_size_x + window_width];
    __shared__ float sh_y[block_size_x * tile_size_x + window_width];
    __shared__ float sh_z[block_size_x * tile_size_x + window_width];

    //the first loading phase
    fill_shared_memory(sh_ct, sh_x, sh_y, sh_z, ct, x, y, z, bx, i, 0, bx-window_width, N);

    #if write_spm == 1
    int offset[tile_size_x];
    if (bx+i==0) {
        offset[0] = 0;
    }
    #pragma unroll
    for (int ti=0; ti<tile_size_x; ti++) {
        if (bx+i+ti*block_size_x-1 >= 0 && bx+i+ti*block_size_x-1 < N) {
            offset[ti] = prefix_sums[bx+i+ti*block_size_x-1];
        }
    }
    #else
    int *offset = (int *)0;
    #endif

    __syncthreads();

    //start of the the computations phase
    float l_ct[tile_size_x];
    float l_x[tile_size_x];
    float l_y[tile_size_x];
    float l_z[tile_size_x];
    #if write_sums == 1
    int sum[tile_size_x];
    #else
    int *sum = (int *)0;
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
    correlate(row_idx, col_idx, sum, offset, bx, i, l_x, l_y, l_z, l_ct,
                    sh_x, sh_y, sh_z, sh_ct, -window_width, 0, criterion);

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
    fill_shared_memory(sh_ct, sh_x, sh_y, sh_z, ct, x, y, z, bx, i, block_size_x*tile_size_x, bx, N);

    __syncthreads();

    //the next loop computes correlations with hits later in time
    correlate(row_idx, col_idx, sum, offset, bx, i, l_x, l_y, l_z, l_ct,
                    sh_x, sh_y, sh_z, sh_ct, 0, 1, criterion);

    #if write_sums == 1
    for (int ti=0; ti<tile_size_x; ti++) {
        if (bx+i+ti*block_size_x < N) {
            sums[bx+i+ti*block_size_x] = sum[ti];
        }
    }
    #endif
}


//constants needed by Match 3B criterion
//the reason that we hard code the constants like this is that we can't use sqrt in device constants sadly
#define roadwidth 90.0f
#define speed_of_light  0.299792458f                // m/ns
#define inverse_c       (1.0f/speed_of_light)
#define index_of_refrac 1.3800851282f               // average index of refraction of water
#define D0 (roadwidth)
#define D1 (roadwidth * 2.0f)
#define D02 (D0 * D0)
#define D12 (D1 * D1)
#define R2 (roadwidth * roadwidth)
#define Rs2 3847.2165714f
#define Rst 58.9942930573f
#define D22 42228.1334918f
#define Rt  85.6010699976f


/*
 * This function implements the Match 3B criterion
 */
__forceinline__ __device__ bool match3b(float x1, float y1, float z1, float t1, float x2, float y2, float z2, float t2) {

    float difft = fabsf(t1 - t2);
    if (isnan(difft)) {
        return false;
    }

    float d2 = ((x1-x2)*(x1-x2)) + ((y1-y2)*(y1-y2)) + ((z1-z2)*(z1-z2));

    float dmax = 0.0f;
    if (d2 < D02) {
        dmax = sqrtf(d2) * index_of_refrac;
    } else {
        dmax = sqrtf(d2 - Rs2) + Rst;
    }
    if (difft > (dmax * inverse_c)) {
        return false;
    } 

    float dmin = 0.0f;
    if (d2 > D22) {
        dmin = sqrtf(d2 - R2) - Rt;
    } else if (d2 > D12) {
        dmin = sqrtf(d2 - D12);
    } else {
        return true;
    }

    return (difft >= (dmin*inverse_c));
}



/*
 * This function implements the quadratic differnce criterion
 */
__forceinline__ __device__ bool quadratic_difference(float x1, float y1, float z1, float ct1, float x2, float y2, float z2, float ct2) {
    float diffct = ct1 - ct2;
    float diffx  = x1 - x2;
    float diffy  = y1 - y2;
    float diffz  = z1 - z2;

    return (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz);
}





/*
 * This is a small helper kernel to ready the input for the Match 3B kernel, currently not in use
 */
__global__ void convert_ct_to_t(float *ct, int n) {
    int i = threadIdx.x + blockIdx.x * block_size_x;
    if (i<n) {
        ct[i] = ct[i] * inverse_c;
    }
}




















#ifndef shared_memory_size
#define shared_memory_size 10*block_size_x
#endif

/*
 * This kernel is an experimental version of the above quadratic_difference_full kernel.
 * It is not production ready and needs more work.
 *
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





