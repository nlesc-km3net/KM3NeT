
#ifndef block_size_x
#define block_size_x 256
#endif


/*
 * This kernel recomputes the degree of each node in the graph and computes
 * the minimum degree of all nodes with at least one edge.
 *
 * output arguments:
 *
 *  minimum an array which will contain a per-thread block minimum degree
 *
 *  num_nodes will contain the per-thread block sum of the number of nodes
 *      with at least 1 edge
 *
 *  degrees contains the degree of all nodes and is updated with the current degree
 *
 * input arguments:
 *
 *  row_idx is the index of the node
 *  col_idx is the index of the node to which this node has an edge, this index
 *      can be -1 if the edge has been removed
 *
 *  prefix_sum contains the start index of each row, because elements can be removed
 *          this subtracting two consecutive numbers no longer indicates the degree
 *
 *  n is the number of nodes in the graph
 */
__global__ void minimum_degree(int *minimum, int *num_nodes, int *degrees, int *row_idx, int *col_idx, int *prefix_sum, int n) {

    int ti = threadIdx.x;
    int i = blockIdx.x * block_size_x + ti;

    if (i<n) {

        //obtain indices for reading col_idx
        int start = 0;
        if (i>0) {
            start = prefix_sum[i-1];
        }
        int end = prefix_sum[i];

        //get the degree of this node
        int degree = 0;
        for (int k=start; k<end; k++) {
            if (col_idx[k] != -1) {
                degree++;
            }
        }

        //update degrees array
        degrees[i] = degree;

        //start the reduce
        //get the minimum value larger than 0
        //and the total number of nodes with degree > 0 (at least 1 edge)
        __shared__ int sh_min[block_size_x];
        __shared__ int sh_sum[block_size_x];

        sh_min[ti] = degree;
        sh_sum[ti] = degree > 0 ? 1 : 0;
        __syncthreads();
        #pragma unroll
        for (unsigned int s=block_size_x/2; s>0; s>>=1) {
            if (ti < s) {
                int self = sh_min[ti];
                int other = sh_min[ti+s];
                if (other > 0) {
                    if (self == 0 || other < self) {
                        sh_min[ti] = other;
                    }
                }
                sh_sum[ti] += sh_sum[ti+s];
            }
            __syncthreads();
        }

        //write output
        if (ti == 0) {
            minimum[blockIdx.x] = sh_min[0];
            num_nodes[blockIdx.x] = sh_sum[0];
        }


    }
}
