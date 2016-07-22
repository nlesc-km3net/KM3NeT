/* 
 * This kernel is meant to find the degree of a node in the correlations
 * table, which is stored in the (window_width_d x N) dense format
 *
 * The number of edges to nodes with a higher id is simply the number of entries
 * in the graph for a given node.
 *
 * To get the full degree of a node, it is necessary to look at the
 * edges of preceeding nodes to find any edges to itself. Fortunately, there
 * is a limit of at most 1500. That is to say, two nodes with ids more than 1500 apart
 * can never have an edge.
 *
 * The degree array is assumed to already contain the out-degree as computed by
 * the quadratic_difference_linear kernel, this kernel adds the in-degree of each node to that number
 * to obtain the total degree of the node.
 */

__global__ void degrees_dense(int *degree, uint8_t *correlations, int n) {

    //node id for which this thread is responsible
    int i = blockIdx.x * block_size_x_d + threadIdx.x;

    if (i < n) {
        int in_degree = 0;

        for (int j=window_width-1; j>=0; j--) {
            int col = i - j -1;
            uint64_t pos = (j * (uint64_t)n) + (uint64_t)col;
            if (col >= 0 && correlations[pos] == 1) {
                in_degree++;
            }

           // perhaps this leads to more irregular access patterns than looping separately like below?
           // int pos1 = j * n + i;
           // if(correlations[pos1] == 1){
           //     in_degree++;
           // }
        }   

        for (int j=0; j<window_width; j++){
            uint64_t pos = (j * (uint64_t)n) + (uint64_t)i;
            if(correlations[pos] == 1){
                in_degree++;
            }
        }
        //could implement a cutoff here to remove all nodes with degree less than some threshold

        degree[i] += in_degree;   //already contains the out-degree, simply add the in-degree
    }
}

