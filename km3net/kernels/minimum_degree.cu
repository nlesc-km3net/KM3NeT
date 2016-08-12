
#ifndef block_size_x
#define block_size_x 128
#endif

#ifndef threshold
#define threshold 3
#endif



/*
 * Helper function that does the reduction step of the algorithm
 *
 * This function reduces the values in a thread block to a single value
 *
 */
__device__ __forceinline__ void reduce_min_num(int *sh_min, int *sh_sum, int lmin, int lnum, int ti) {
    sh_min[ti] = lmin;
    sh_sum[ti] = lnum;
    __syncthreads();
    #pragma unroll
    for (unsigned int s=block_size_x/2; s>0; s>>=1) {
        if (ti < s) {
            int self = sh_min[ti];
            int other = sh_min[ti+s];
            if (other >= threshold) {
                if (self < threshold || other < self) {
                    sh_min[ti] = other;
                }
            }
            sh_sum[ti] += sh_sum[ti + s];
        }
        __syncthreads();
    }
}





/*
 * This kernel recomputes the degree of each node in the graph and computes
 * the minimum degree of all nodes with at least one edge.
 *
 * output arguments:
 *
 *  minimum an array containing a per-thread block minimum degree
 *
 *  num_nodes will contain the per-thread block sum of the number of nodes
 *      with at least 'threshold' edges
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

    int degree = 0;

    if (i<n) {

        
        //obtain indices for reading col_idx
        int start = 0;
        if (i>0) {
            start = prefix_sum[i-1];
        }
        int end = prefix_sum[i];

        int max_degree = degrees[i];

        //get the degree of this node
        for (int k=start; k<end && degree < max_degree; k++) {
            if (col_idx[k] != -1) {
                degree++;
            }
        }

        //update degrees array
        degrees[i] = degree;

    }

    //start the reduce
    //get the minimum value larger than 0
    //and the total number of nodes with degree >= theshold (at least 'threshold' edges)
    __shared__ int sh_min[block_size_x];
    __shared__ int sh_sum[block_size_x];
    int lnum = 0;
    if (degree >= threshold) {
        lnum = 1;
    }
    reduce_min_num(sh_min, sh_sum, degree, lnum, ti);

    //write output
    if (ti == 0) {
        minimum[blockIdx.x] = sh_min[0];
        num_nodes[blockIdx.x] = sh_sum[0];
    }

}





/*
 * Helper kernel to combine per-thread block results into single values
 *
 * call with 1 thread block, block_size_x should be sufficiently large
 */
__global__ void combine_blocked_min_num(int *minimum, int *num_nodes, int n) {
    int ti = threadIdx.x;

    __shared__ int sh_min[block_size_x];
    __shared__ int sh_sum[block_size_x];

    int lmin = 0;
    int lnum = 0;

    if (ti < n) {
        lmin = minimum[ti];
        lnum = num_nodes[ti];
    }

    reduce_min_num(sh_min, sh_sum, lmin, lnum, ti);

    if (ti==0) {
        minimum[0] = sh_min[0];
        num_nodes[0] = sh_sum[0];
    }

}
