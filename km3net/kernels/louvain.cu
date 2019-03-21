#ifndef block_size_x
#define block_size_x 128
#endif

// #define DEBUG_MODE 1

#ifndef gamma_mod
#define gamma_mod 1
#endif

#include <stdio.h>
#include <cmath>

__global__ void move_nodes(int *n_tot, int *m_tot, int *col_idx, int *prefix_sum, int *degrees, int *community_idx, int *community_degrees, int *tmp_community_idx) {
    const int graph_ord = *n_tot;
    const int m = *m_tot;
    int ti = blockIdx.x * block_size_x + threadIdx.x;


    for(int i = ti; i < graph_ord; i += block_size_x ) {

        //define neighbour range
        int start = 0;
        if (i>0) {
            start = prefix_sum[i-1];
        }
        int end = prefix_sum[i];

        //modularity
        int current_comm = community_idx[i];
        int new_comm = current_comm;
        float local_q = 0;
        float max_q = 0;

        //iterate over neighbours of i 
        for(int j = start; j < end; j++) {

            int col = col_idx[j];
            

            //get community of neighbour
            int col_comm = community_idx[col];

            int l_i_comm = 0;   //number of edges joining i with community
            int k_comm = community_degrees[col_comm];     //degree of community

            // The singlet minimum HEURISTIC
            if(i == current_comm && degrees[i] == community_degrees[current_comm] && col == col_comm && degrees[col] == k_comm && col_comm > current_comm) {
                #ifdef DEBUG_MODE
                if(ti == 0) {
                    printf("$$$");
                    printf("SKIP CHANGE %d to %d \n", i, col);
                    printf("$$$");
                }
                #endif

                continue;
            }

            //search for other neighbors from this community
            for(int n = start; n < end; n++) {
                int col_n = col_idx[n];
                //check if its from the same community
                if(community_idx[col_n] != col_comm) {
                    continue;
                }
                l_i_comm++;
            }

            // local_q = (1.0 / (float)graph_ord) * ((float)l_i_comm - ((float)degrees[i] * (float)k_comm / (2.0 * (float)graph_ord)));
            local_q = (1.0 / (float)m) * ((float)l_i_comm - ((float)degrees[i] * (float)k_comm / (2.0 * (float)m)));
            // local_q = (1 / (2* (float)m_tot)) * ( l_i_comm - (k_comm * (float)degrees[i] / (float)m_tot) );

            #ifdef DEBUG_MODE
            if(ti == 0) {
                printf("=============== \n");
                printf("migrate %d to %d \n", i, col_comm);
                printf("m_tot = %d \n", m);
                printf("l_i_comm = %d \n", l_i_comm);
                printf("degrees[i] = %d \n", degrees[i]);
                printf("k_comm = %d \n", k_comm);
                printf("local_q = %f \n", local_q);
            }
            #endif

            if(local_q >= max_q) {
                if(local_q == max_q && new_comm < col_comm) {
                    //do nothing
                } else {
                    #ifdef DEBUG_MODE
                    if(ti ==0) {
                        printf("$$$$$ \n");
                        printf("migrated [%d] from %d to %d \n", i, new_comm, col_comm);
                        printf("previous q: %f , current q: %f \n", max_q, local_q);
                        printf("$$$$$ \n");
                    }
                    #endif

                    new_comm = col_comm;
                    max_q = local_q;
                }

                
                
            }
        }

        tmp_community_idx[i] = new_comm;
    }
}

__global__ void calculate_community_degrees(int *n_tot, int *community_idx, int *degrees, int *community_degrees) {
    const int graph_ord = *n_tot;
    int ti = blockIdx.x * block_size_x + threadIdx.x;

    for(int i = ti; i < graph_ord; i += block_size_x ) {
        int com_deg = 0;
        for(int j = 0; j < graph_ord; j++) {
            if(community_idx[j] == i) {
                com_deg += degrees[j];
            }
        }
        community_degrees[i] = com_deg;
    }
}

__global__ void calculate_community_internal_edges(int *n_tot, int *col_idx, int *prefix_sum, int *community_idx, int *community_int_edg) {
    const int graph_ord = *n_tot;
    int ti = blockIdx.x * block_size_x + threadIdx.x;

    for(int i = ti; i < graph_ord; i += block_size_x ) {
        int inter_count = 0;
       
        //define neighbour range
        int start = 0;
        if (i>0) {
            start = prefix_sum[i-1];
        }
        int end = prefix_sum[i];
        int current_comm = community_idx[i];

        //iterate over neighbours of i 
        for (int j = start; j < end; j++) {
            int col = col_idx[j];
            if (community_idx[col] == current_comm) {
                inter_count++;
            }
        }

        community_int_edg[i] = inter_count;
    }
}

__global__ void calculate_community_internal_sum(int *n_tot, int *community_idx, int *community_int_edg, int *community_internal_sum) {
    const int graph_ord = *n_tot;
    int ti = blockIdx.x * block_size_x + threadIdx.x;

    for(int i = ti; i < graph_ord; i += block_size_x ) {
        int comm_sum = 0;
        for(int j = 0; j < graph_ord; j++) {
            if(community_idx[j] == i) {
                comm_sum += community_int_edg[j];
            }
        }

        // edges are bidirectional
        community_internal_sum[i] = comm_sum / 2;
    }
}

__global__ void calc_part_modularity(int *n_tot, int *m_tot, int *inter_comm_deg, int *community_degrees, float *part_modularities) {
    const int graph_ord = *n_tot;
    const int m_ = *m_tot;
    int ti = blockIdx.x * block_size_x + threadIdx.x;

    for(int i = ti; i < graph_ord; i += block_size_x) {
        float lc = (float)inter_comm_deg[i];
        float kc = (float)community_degrees[i];
        float m = (float)m_;
        part_modularities[i] = ( ( lc/m ) - (pow(kc/(2*m), 2.0f)) );
    }
}