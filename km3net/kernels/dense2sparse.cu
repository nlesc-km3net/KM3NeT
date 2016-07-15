#include <inttypes.h>

#ifndef block_size_x
#define block_size_x 256
#endif

#ifndef window_width
#define window_width 1500
#endif

/*
 * This kernel creates a sparse representation of the densely stored correlations table.
 *
 * In addition to the correlations table, this kernel needs a precomputed prefix_sums array. This
 * array contains the inclusive prefix sums of the degrees of the nodes in the correlations table.
 * In other words, it is an array with one element per hit, containing the sum over the total
 * number of hits correlated with the hits up to and including that hit in the correlations table.
 *
 * Output arguments are row_idx and col_idx, which contain the (hit id, hit id) pairs that describe
 * the correlated hits.
 *
 */
__global__ void dense2sparse_kernel(int *row_idx, int *col_idx, int *prefix_sums, uint8_t *correlations, int n) {
    int i = blockIdx.x * block_size_x + threadIdx.x;

    if (i<n) {

        //get the offset to where output should be written
        int offset = 0;
        if (i>0) {
            offset = prefix_sums[i-1];
        }

        //see how much work there is on this row
        //int end = prefix_sums[i];

        //collect the edges to nodes with lower id
        for (int j=window_width-1; j>=0; j--) {
            int col = i - j - 1;
            uint64_t pos = (j * (uint64_t)n) + (uint64_t)col;
            if (col >= 0 && correlations[pos] == 1) {
                row_idx[offset] = i;
                col_idx[offset] = col;
                offset += 1;
            }
        }

        //collect the edges to nodes with higher id
        for (int j=0; j<window_width; j++) {
            uint64_t pos = (j * (uint64_t)n) + (uint64_t)i;
            if (correlations[pos] == 1) {
                row_idx[offset] = i;
                col_idx[offset] = i+j+1;
                offset += 1;
            }
        }


    }
}
