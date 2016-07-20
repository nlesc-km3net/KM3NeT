


#ifndef block_size_x
#define block_size_x 128
#endif



/*
 * This kernel removes nodes with degree less than or equal to minimum.
 * For the remaining nodes this kernel removes edges to nodes that have been removed.
 *
 * To remove a node we need to set its degree to zero
 * To remove an edge we need to set its col_idx to -1
 */
__global__ void remove_nodes(int *degrees, int *row_idx, int *col_idx, int *prefix_sum, const int *__restrict__ minimum, int n) {
    int i = blockIdx.x * block_size_x + threadIdx.x;

    if (i<n) {
        int my_degree = degrees[i];
        int min = minimum[0];

        //if my node needs to be removed, remove it
        if (my_degree > 0 && my_degree <= min) {
            degrees[i] = 0;
        }
        
        //if my node remains, update my edges
        if (my_degree > min) {

            //obtain indices to iterate over my edges
            int start = 0;
            if (i>0) {
                start = prefix_sum[i-1];
            }
            int end = prefix_sum[i];

            //remove edges to nodes with degree less than or equal to min
            for (int k=start; k<end; k++) {
                int col = col_idx[k];
                if (col != -1 && degrees[col] <= min) {
                    col_idx[k] = -1;
                }

            }

        }


    }
}


