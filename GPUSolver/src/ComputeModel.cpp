
#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <array>
#include <algorithm>
#include <map>

#include "include/model.hpp"
#include "include/ComputeModel.hpp"

#include <omp.h>

#define alpha f
#define beta 1.0f


/*************************************

Problem: 3d Poisson:  -∇²φ = f  in Ω
                         φ = g  on ∂Ω

            φ = ALPHA * (BETA - x² - y² - z²) on ∂Ω
            f = FORCING_TERM  in Ω
            g = ALPHA * (BETA - x² - y² - z²)

            ∂Ω is the outside surface of complex volumetric mesh
            Ω is the inner volume of the mesh

*******************************************/


/**************************************************
 * FEM first order tetrahedral elements derivation
 * 
 * *************************************************/

#define FORCING_TERM 7.2f
#define ALPHA 0.00048f
#define BETA 25.0f

template<typename T_value>
T_value analytical_solution(T_value x, T_value y, T_value z) {
    return 1.2 * (25.0 - x*x - y*y - z*z);
}


template<typename T_index, typename T_value>
std::vector<Edge<T_index>> get_mesh_edges(const Model<T_index, T_value>& model) {
    std::set<Edge<T_index>> edge_set;

    for (const auto& elem : model.elements) {
        T_index v[4] = {elem.v1, elem.v2, elem.v3, elem.v4};

        Edge<T_index> e1(v[0], v[1]);
        Edge<T_index> e2(v[0], v[2]);
        Edge<T_index> e3(v[0], v[3]);
        Edge<T_index> e4(v[1], v[2]);
        Edge<T_index> e5(v[1], v[3]);
        Edge<T_index> e6(v[2], v[3]);

        e1.sort_edges();
        e2.sort_edges();
        e3.sort_edges();
        e4.sort_edges();
        e5.sort_edges();
        e6.sort_edges();

        edge_set.insert(e1);
        edge_set.insert(e2);
        edge_set.insert(e3);
        edge_set.insert(e4);
        edge_set.insert(e5);
        edge_set.insert(e6);
    }

    return std::vector<Edge<T_index>>(edge_set.begin(), edge_set.end());

}

// I want a function given a model and list of edges to compute maximum edge length
template<typename T_index, typename T_value>
T_value compute_max_edge_length(const Model<T_index, T_value>& model, const std::vector<Edge<T_index>>& edges) {
    
    T_value max_length{0.0};

    for (const auto& edge : edges) {
        const auto& v1 = model.vertices[edge.v1];
        const auto& v2 = model.vertices[edge.v2];

        T_value length = std::sqrt(
            (v1.x - v2.x) * (v1.x - v2.x) +
            (v1.y - v2.y) * (v1.y - v2.y) +
            (v1.z - v2.z) * (v1.z - v2.z)
        );

        if (length > max_length) {
            max_length = length;
        }
    }

    return max_length;
}


template<typename T_index, typename T_value>
std::vector<T_index> extract_boundary_nodes(const Model<T_index, T_value>& model) {
    
    std::map<Face<T_index>, T_index> face_counts;

    for (const auto& elem : model.elements) {


        T_index element_v[4] = {elem.v1, elem.v2, elem.v3, elem.v4};
        

        //    For reference all possible face combinations in a tetrahedron (indexing starts at 1)
        //    {1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4}

        Face<T_index> f1 = Face<T_index>(elem.v1, elem.v2, elem.v3);
        Face<T_index> f2 = Face<T_index>(elem.v1, elem.v2, elem.v4);
        Face<T_index> f3 = Face<T_index>(elem.v1, elem.v3, elem.v4);
        Face<T_index> f4 = Face<T_index>(elem.v2, elem.v3, elem.v4);

        f1.sort_faces();
        f2.sort_faces();
        f3.sort_faces();
        f4.sort_faces();

        face_counts[f1]++;
        face_counts[f2]++;
        face_counts[f3]++;
        face_counts[f4]++;

    }

    std::set<T_index> boundary_set;
    for (const auto& face_frequency_pair : face_counts) {
        if (face_frequency_pair.second == 1) {
            boundary_set.insert(face_frequency_pair.first.v1);
            boundary_set.insert(face_frequency_pair.first.v2);
            boundary_set.insert(face_frequency_pair.first.v3);
        }
    }

    return std::vector<T_index>(boundary_set.begin(), boundary_set.end());
}

template<typename T_index, typename T_value>
void initialize_CSR_indices(const Model<T_index, T_value> &model, CSR_matrix<T_index, T_value> &A) {

    T_index N = model.n_vertices;

    std::vector<T_index> boundary_nodes = extract_boundary_nodes<T_index, T_value>(model);
    

    std::vector<bool> is_boundary(N, false);
    for (T_index idx : boundary_nodes) {
        is_boundary[idx] = true;
    }

    // Should be faster than searching each time
    // just initialize a mask: if not on boundary, keep node, else -1
    // time: 2N, instead of NlogNb (Nb ==> number of boundary nodes)
    std::vector<T_index> is_boundary_mask(N);
    T_index internal_count = 0;

    for (T_index i = 0; i < N; ++i) {
        if (!is_boundary[i]) {
            is_boundary_mask[i] = internal_count;
            internal_count++;
        } else {
            is_boundary_mask[i] = -1;
        }
    }

    //matrix is symmetric and adjacency is pairwise :)
    A.n_rows = internal_count;
    A.n_cols = internal_count;

    std::vector<std::set<T_index>> adjacency(internal_count);


    for (const auto& elem : model.elements) {
        T_index nodes[4] = {elem.v1, elem.v2, elem.v3, elem.v4};
        
        for (T_index row_node_global : nodes) {
            T_index row_node_reduced = is_boundary_mask[row_node_global];


            if (row_node_reduced == -1) continue;

            for (T_index col_node_global : nodes) {
                T_index col_node_reduced = is_boundary_mask[col_node_global];

                
                if (col_node_reduced != -1) {
                    adjacency[row_node_reduced].insert(col_node_reduced);
                }
            }
        }
    }

    T_index total_nnz = 0;
    for (const auto& row_set : adjacency) {
        total_nnz += row_set.size();
    }


    A.n_nonzero = total_nnz;
    A.row_ptr.resize(internal_count + 1);
    A.col_ind.resize(total_nnz);
    A.values.resize(total_nnz, 0.0f);

    T_index current_nnz_index = 0;
    A.row_ptr[0] = 0;

    for (T_index i = 0; i < internal_count; ++i) {
        for (T_index col : adjacency[i]) {
            A.col_ind[current_nnz_index] = col;
            current_nnz_index++;
        }
        A.row_ptr[i + 1] = current_nnz_index;
    }
}


template<typename T_index>
T_index get_csr_index(const std::vector<T_index>& row_ptr, 
                      const std::vector<T_index>& col_ind, 
                      T_index row, T_index col) {
    
    T_index start = row_ptr[row];
    T_index end = row_ptr[row + 1];
    
    // col_ind is sorted because it was built using std::set iteration
    auto it = std::lower_bound(col_ind.begin() + start, col_ind.begin() + end, col);
    
    if (it != col_ind.begin() + end && *it == col) {
        return (T_index)(it - col_ind.begin());
    }
    return -1; // Should not happen if sparsity pattern is correct
}

template<typename T_index, typename T_value>
void fill_FEM_CSR(const Model<T_index, T_value> &model,
                  CSR_matrix<T_index, T_value> &A,
                  std::vector<T_value> &b) {

    T_index N = model.n_vertices;

    // 1. Re-build Mapping (Global -> Reduced)
    std::vector<T_index> boundary_nodes = extract_boundary_nodes<T_index, T_value>(model);
    std::vector<bool> is_boundary(N, false);
    for (T_index idx : boundary_nodes) is_boundary[idx] = true;

    std::vector<T_index> global_to_reduced(N);
    T_index internal_count = 0;
    for (T_index i = 0; i < N; ++i) {
        if (!is_boundary[i]) {
            global_to_reduced[i] = internal_count++;
        } else {
            global_to_reduced[i] = -1;
        }
    }

    // Resize RHS vector to match internal nodes
    b.assign(internal_count, 0.0f);

    // 2. ASSEMBLE STIFFNESS MATRIX AND LOAD VECTOR
    for (const auto& elem : model.elements) {

        T_index nodes[4] = {elem.v1, elem.v2, elem.v3, elem.v4};

        // Cache coordinates
        T_value x[4], y[4], z[4];
        for (int i = 0; i < 4; ++i) {
            x[i] = model.vertices[nodes[i]].x;
            y[i] = model.vertices[nodes[i]].y;
            z[i] = model.vertices[nodes[i]].z;
        }

        T_value B[4], C[4], D[4];

        // Geometric calculations
        T_value x21 = x[1] - x[0], x31 = x[2] - x[0], x41 = x[3] - x[0];
        T_value y21 = y[1] - y[0], y31 = y[2] - y[0], y41 = y[3] - y[0];
        T_value z21 = z[1] - z[0], z31 = z[2] - z[0], z41 = z[3] - z[0];

        T_value detJ = x21 * (y31 * z41 - y41 * z31) -
                       x31 * (y21 * z41 - y41 * z21) +
                       x41 * (y21 * z31 - y31 * z21);

        T_value vol = std::abs(detJ) / 6.0f;

        if (vol < 1e-12) continue;

        // Gradients
        B[0] =   y[1]*(z[2]-z[3]) + y[2]*(z[3]-z[1]) + y[3]*(z[1]-z[2]);
        C[0] = -(x[1]*(z[2]-z[3]) + x[2]*(z[3]-z[1]) + x[3]*(z[1]-z[2]));
        D[0] =   x[1]*(y[2]-y[3]) + x[2]*(y[3]-y[1]) + x[3]*(y[1]-y[2]);

        B[1] =   y[0]*(z[3]-z[2]) + y[2]*(z[0]-z[3]) + y[3]*(z[2]-z[0]);
        C[1] = -(x[0]*(z[3]-z[2]) + x[2]*(z[0]-z[3]) + x[3]*(z[2]-z[0]));
        D[1] =   x[0]*(y[3]-y[2]) + x[2]*(y[0]-y[3]) + x[3]*(y[2]-y[0]);

        B[2] =   y[0]*(z[1]-z[3]) + y[1]*(z[3]-z[0]) + y[3]*(z[0]-z[1]);
        C[2] = -(x[0]*(z[1]-z[3]) + x[1]*(z[3]-z[0]) + x[3]*(z[0]-z[1]));
        D[2] =   x[0]*(y[1]-y[3]) + x[1]*(y[3]-y[0]) + x[3]*(y[0]-y[1]);

        B[3] =   y[0]*(z[2]-z[1]) + y[1]*(z[0]-z[2]) + y[2]*(z[1]-z[0]);
        C[3] = -(x[0]*(z[2]-z[1]) + x[1]*(z[0]-z[2]) + x[2]*(z[1]-z[0]));
        D[3] =   x[0]*(y[2]-y[1]) + x[1]*(y[0]-y[2]) + x[2]*(y[1]-y[0]);

        // Element Load
        T_value f_e = 7.2f;
        T_value load_val = f_e * vol / 4.0f;

        // Assembly
        for (int i = 0; i < 4; ++i) {
            
            T_index row_global = nodes[i];
            T_index row_reduced = global_to_reduced[row_global];

            // If this node is a boundary node, we do not solve for it. Skip.
            if (row_reduced == -1) continue;

            // 1. Add standard volume load to Internal Node
            b[row_reduced] += load_val;

            for (int j = 0; j < 4; ++j) {
                
                T_index col_global = nodes[j];
                T_index col_reduced = global_to_reduced[col_global];

                // Calculate Stiffness Coefficient K_ij
                T_value k_val = (B[i]*B[j] + C[i]*C[j] + D[i]*D[j]) / (6.0f * std::abs(detJ));

                if (col_reduced != -1) {
                    // CASE A: Internal Node connected to Internal Node
                    // Add to Matrix A
                    // Note: We use reduced indices for CSR lookup
                    T_index idx = get_csr_index(A.row_ptr, A.col_ind, row_reduced, col_reduced);
                    if (idx != -1) {
                        A.values[idx] += k_val;
                    }
                } else {
                    // CASE B: Internal Node connected to Boundary Node
                    // Move to RHS: F_effective = F_load - K_ij * u_known
                    
                    T_value bx_coord = model.vertices[col_global].x;
                    T_value by_coord = model.vertices[col_global].y;
                    T_value bz_coord = model.vertices[col_global].z;
                    
                    T_value g_val = analytical_solution(bx_coord, by_coord, bz_coord);

                    b[row_reduced] -= k_val * g_val;
                }
            }
        }
    }
}


// Explicit template instantiations to ensure code is emitted for the types we use.

template void initialize_CSR_indices<int, float>(const Model<int, float> &model, CSR_matrix<int, float> &A);
template void initialize_CSR_indices<int, double>(const Model<int, double> &model, CSR_matrix<int, double> &A);

template void fill_FEM_CSR<int, float>(const Model<int, float> &model,
                                      CSR_matrix<int, float> &A,
                                      std::vector<float> &b);
template void fill_FEM_CSR<int, double>(const Model<int, double> &model,
                                       CSR_matrix<int, double> &A,
                                       std::vector<double> &b);

template float analytical_solution<float>(float x, float y, float z);
template double analytical_solution<double>(double x, double y, double z);

template std::vector<int> extract_boundary_nodes<int, float>(const Model<int, float>& model);
template std::vector<int> extract_boundary_nodes<int, double>(const Model<int, double>& model);

template int get_csr_index(const std::vector<int>& row_ptr, 
                      const std::vector<int>& col_ind, 
                      int row, int col);


template std::vector<Edge<int>> get_mesh_edges(const Model<int, float>& model);
template std::vector<Edge<int>> get_mesh_edges(const Model<int, double>& model);

template float compute_max_edge_length(const Model<int, float>& model, const std::vector<Edge<int>>& edges);
template double compute_max_edge_length(const Model<int, double>& model, const std::vector<Edge<int>>& edges);
