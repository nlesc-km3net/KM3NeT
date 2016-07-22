
/*
 * This kernel computes the correlated hits of hits no more than 1500 apart.
 * It does this using a 1-dimensional mapping of threads and thread blocks.
 *
 * This kernel supports the usual set of optimizations, including tiling, partial loop unrolling, read-only cache. 
 * Tuning parameters supported are 'read_only' [0,1], 'tile_size_x_qd' divisor of 1500, and 'block_size_x_qd' multiple of 32.
 *
 * 'write_sums_qd' can be set to [0,1] to enable the code that
 * produces another output, namely the number of correlated
 * hits per row. This number is later to create the sparse
 * representation of the correlations table. If not using
 * sparse respresentation set write_sums_qd to 0.
 */
__global__ void quadratic_difference_linear(char *__restrict__ correlations, int *sums, int N, int sliding_window_width,
        const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z, const float *__restrict__ ct) {

    int tx = threadIdx.x;
    int bx = blockIdx.x * block_size_x_qd * tile_size_x_qd;

    __shared__ float sh_ct[block_size_x_qd * tile_size_x_qd + window_width];
    __shared__ float sh_x[block_size_x_qd * tile_size_x_qd + window_width];
    __shared__ float sh_y[block_size_x_qd * tile_size_x_qd + window_width];
    __shared__ float sh_z[block_size_x_qd * tile_size_x_qd + window_width];

    if (bx+tx < N) {

        //the loading phase
        for (int k=tx; k < block_size_x_qd*tile_size_x_qd+window_width; k+=block_size_x_qd) {
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
        float l_ct[tile_size_x_qd];
        float l_x[tile_size_x_qd];
        float l_y[tile_size_x_qd];
        float l_z[tile_size_x_qd];
        #if write_sums_qd == 1
        int sum[tile_size_x_qd];
        #endif

        //keep the most often used values in registers
        for (int ti=0; ti<tile_size_x_qd; ti++) {
            l_ct[ti] = sh_ct[i+ti*block_size_x_qd];
            l_x[ti] = sh_x[i+ti*block_size_x_qd];
            l_y[ti] = sh_y[i+ti*block_size_x_qd];
            l_z[ti] = sh_z[i+ti*block_size_x_qd];
            #if write_sums_qd == 1
            sum[ti] = 0;
            #endif
        }

        //small optimization to eliminate bounds checks for most blocks
        if (bx+block_size_x_qd*tile_size_x_qd+window_width < N) {

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
                for (int ti=0; ti<tile_size_x_qd; ti++) {

                        float diffct = l_ct[ti] - sh_ct[i+ti*block_size_x_qd+j];
                        float diffx  = l_x[ti] - sh_x[i+ti*block_size_x_qd+j];
                        float diffy  = l_y[ti] - sh_y[i+ti*block_size_x_qd+j];
                        float diffz  = l_z[ti] - sh_z[i+ti*block_size_x_qd+j];

                        if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                            uint64_t pos = (j-1) * ((uint64_t)N) + (bx+i+ti*block_size_x_qd);
                            correlations[pos] = 1;
                            #if write_sums_qd == 1
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

                for (int ti=0; ti<tile_size_x_qd; ti++) {

                    if (bx+i+ti*block_size_x_qd+j < N) {

                        float diffct = l_ct[ti] - sh_ct[i+ti*block_size_x_qd+j];
                        float diffx  = l_x[ti] - sh_x[i+ti*block_size_x_qd+j];
                        float diffy  = l_y[ti] - sh_y[i+ti*block_size_x_qd+j];
                        float diffz  = l_z[ti] - sh_z[i+ti*block_size_x_qd+j];

                        if (diffct * diffct < diffx * diffx + diffy * diffy + diffz * diffz) {
                            uint64_t pos = (j-1) * ((uint64_t)N) + (bx+i+ti*block_size_x_qd);
                            correlations[pos] = 1;
                            #if write_sums_qd == 1
                            sum[ti] += 1;
                            #endif
                        }

                    }
                }



            }

        }

        #if write_sums_qd == 1
        for (int ti=0; ti<tile_size_x_qd; ti++) {
            sums[bx+i+ti*block_size_x_qd] = sum[ti];
        }
        #endif

    }
}


/*
 * This is the old kernel that uses a 2D thread block layout, mainly kept here to verify the correctness of the linear kernel
 */
__global__ void quadratic_difference(int8_t *correlations, int N, int sliding_window_width, float *x, float *y, float *z, float *ct)
{
    int i = blockIdx.x * block_size_x_qd + threadIdx.x;
    int j = blockIdx.y * block_size_y_qd + threadIdx.y;

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


