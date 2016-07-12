#include <stdio.h>

#ifndef block_size_x
#define block_size_x 256
#endif


/*
 * This file contains the CUDA implementation for a parallel prefix-sum kernel.
 *
 * Prefix sums are a common parallel algorithm mainly used in GPU Computing when
 * the number of outputs per thread is not constant. The prefix sum of an array
 * is an array that contains the sum of all preceeding elements in the original
 * array. If the value itself is included the prefix sum is considered to be
 * inclusive, and exclusive if the value itself is not included. As such, for an
 * exclusive prefix sum the first element is of the prefix sum array is always 0.
 * For an inclusive sum the last element of the prefix sum array contains the sum
 * of the input array.
 *
 */


/*
 * This is a small device function that will be inlined by the compiler in the CUDA kernel.
 *
 * This CUDA device function includes some in-line PTX code, for more info about inlining PTX see: 
 * http://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
 *
 * The reason to use predicated warp-shuffle instructions is because all threads in a warp need
 * to compute a sum of all preceeding elements. This is basically done in the same manner as for a
 * regular sum, where all threads in the end have the same sum value. However, by using predication we can
 * disable certain threads during the add instruction and ensure that threads only add values
 * previously owned by threads with a lower thread id than their own.
 *
 * This function operates at the warp-level and only computes the prefix sum within a 32-thread warp.
 *
 */
__device__ __forceinline__ int prefix_sum_warp(int v, int end) {
    int x = v;

    asm("{                  \n\t"
        "  .reg .s32  t;    \n\t"
        "  .reg .pred p;    \n\t");

    #pragma unroll
    for (unsigned int d=1; d<end; d*=2) {
        asm("shfl.up.b32 t|p, %0, %1, 0x0;  \n\t"
            "@p add.s32 %0, %0, t;          \n\t" : "+r"(x) : "r"(d));
    }

    asm("}");

    return x;
}


/*
 * This kernel computes a block-wide prefix sum using the prefix_sum_warp function within warps.
 * The last thread of each warp writes the value that should carry over to the next warp into
 * the shared memory array warp_carry. Then within a single warp the prefix_sum of the warp carry
 * is computed, after which the warp carries are propagated to all warps. The kernel also writes
 * the block carry of the thread block into the global memory array block_carry.

 * After first kernel execution, which should use a zerod block_carray array as input, this same
 * kernel can be called again using the previously written block_carry array as input array,
 * to compute the prefix sum of all block carries. Finally, the propagate_block_carry kernel
 * can be used to propagate the block carry values to all elements.
 */
__global__ void prefix_sum_block(int *prefix_sums, int *block_carry, int *input, int n) {
    int tx = threadIdx.x;
    int x = blockIdx.x * block_size_x + tx;
    int v = 0;
    if (x < n) {
        v = input[blockIdx.x * block_size_x + tx];
    }

    v = prefix_sum_warp(v, 32);

    #if block_size_x > 32
    int laneid = tx & (32-1);
    int warpid = tx / 32;

    __shared__ int warp_carry[block_size_x/32];
    if (laneid == 31) {
        warp_carry[warpid] = v;
    }
    __syncthreads();

    if (tx < block_size_x/32) {
        int temp = warp_carry[tx];
        temp = prefix_sum_warp(temp, block_size_x/32);
        warp_carry[tx] = temp;
    }
    __syncthreads();

    if (warpid>0) {
        v += warp_carry[warpid-1];
    }
    #endif

    if (x < n) {
        prefix_sums[x] = v;
    }

    if (tx == block_size_x-1) {
        block_carry[blockIdx.x] = v;
    }
}


/*
 * This is a simple kernel that can be used to propagate block carry
 * values to all previously computed block-wide prefix sums.
 */
__global__ void propagate_block_carry(int *prefix_sums, int *block_carry, int n) {
    int x = blockIdx.x * block_size_x + threadIdx.x;
    if (blockIdx.x > 0 && x < n) {
        prefix_sums[x] = prefix_sums[x] + block_carry[blockIdx.x-1];
    }
}
