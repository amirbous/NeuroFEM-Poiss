#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

#include <CLI/CLI.hpp>

#include "include/IO.hpp"
#include "include/model.hpp"
#include "include/ComputeModel.hpp"
#include "include/Solver.hpp"

#include <ginkgo/ginkgo.hpp>

int main(int argc, char* argv[])  {

    CLI::App app{"APU FEM solver with CLI"};
    
    using index_type = int;
    using value_type = float;
    
    Model<index_type, value_type> poissfem_model;
    CSR_matrix<index_type, value_type> A;

    std::vector<value_type> b; 
    std::vector<value_type> x0; // solution vector

    std::string model_name = "Sphere_00";
    app.add_option("model_name", model_name, "Problem name")
       ->capture_default_str()
       ->expected(1);

    int mode = 0;
    app.add_option("-m,--mode", mode, "Mode flag (0: assemble from mesh, 1: read from file)")
       ->capture_default_str()
       ->expected(1);
    int use_gpu = 0;

    app.add_option("-g,--gpu", use_gpu, "Flag to enable GPU acceleration (0: CPU, 1: GPU)")
       ->capture_default_str()
       ->expected(1);

    CLI11_PARSE(app, argc, argv);

    // Variables for the final report
    long assemble_time = 0;
    long transfer_time = 0;
    long solve_time = 0;
    float energy_joules = 0.0f;
    value_type max_edge_length = 0.0;
    value_type l2_error = 0.0;
    index_type report_vertices = 0;

    if (mode == 1) {
        std::cout << "Matrix prefetching mode enabled: Reading files directly." << std::endl;

        // 1. Read Matrix and RHS
        std::string mtx_file = model_name + "_mtx.txt";
        readCOOMatrix<index_type, value_type>(mtx_file, A);
        b = ReadVector<index_type, value_type>(model_name, "rhs");
        
        // Ensure size is correct for x0
        x0.assign(A.n_rows, 0.0f);
        report_vertices = A.n_rows; // Use matrix rows as the "vertices" count for the report

        // 2. Solve in Ginkgo
        simple_logger custom_simple_logger = solveGinkgo<index_type, value_type>(poissfem_model, A, b, x0, use_gpu);

        transfer_time = custom_simple_logger.transfer_duration;
        solve_time = custom_simple_logger.solve_duration;
        energy_joules = custom_simple_logger.energy_joules;

    } else {
        std::cout << "Mesh assembly mode enabled: Reading VTK and building system." << std::endl;

        // 1. Read input
        ReadVTK(model_name, poissfem_model);

        // 2. Assembly part
        auto assemble_begin = std::chrono::steady_clock::now();
        initialize_CSR_indices<index_type, value_type>(poissfem_model, A);
        fill_FEM_CSR<index_type, value_type>(poissfem_model, A, b);
        auto assemble_end = std::chrono::steady_clock::now();
        assemble_time = std::chrono::duration_cast<std::chrono::microseconds>(assemble_end - assemble_begin).count();
        
        // 3. Solver part
        x0.assign(A.n_rows, 0.0f);
        simple_logger custom_simple_logger = solveGinkgo<index_type, value_type>(poissfem_model, A, b, x0, use_gpu);

        transfer_time = custom_simple_logger.transfer_duration;
        solve_time = custom_simple_logger.solve_duration;
        energy_joules = custom_simple_logger.energy_joules;
        report_vertices = poissfem_model.n_vertices;

        // 4. Post-processing (Mesh-dependent)
        std::vector<index_type> boundary_nodes = extract_boundary_nodes<index_type, value_type>(poissfem_model);
        std::vector<bool> is_boundary(poissfem_model.n_vertices, false);
        
        for (index_type idx : boundary_nodes) {
            is_boundary[idx] = true;
        }

        index_type internal_idx = 0; 
        for (int i = 0; i < poissfem_model.n_vertices; ++i) {
            value_type vx = poissfem_model.vertices[i].x;
            value_type vy = poissfem_model.vertices[i].y;
            value_type vz = poissfem_model.vertices[i].z;

            if (!is_boundary[i]) {
                poissfem_model.vertices[i].potential = x0[internal_idx];
                internal_idx++;
            } else {
                poissfem_model.vertices[i].potential = analytical_solution(vx, vy, vz);
            }
            poissfem_model.vertices[i].density = analytical_solution<value_type>(vx, vy, vz);
        }

        // 5. Error Calculation
        value_type total_l2_error_sq{0.0};
        std::vector<Edge<index_type>> edges = get_mesh_edges<index_type, value_type>(poissfem_model);
        max_edge_length = compute_max_edge_length<index_type, value_type>(poissfem_model, edges);

        for (int i = 0; i < poissfem_model.n_elements; ++i) {
            index_type v1 = poissfem_model.elements[i].v1;
            index_type v2 = poissfem_model.elements[i].v2;
            index_type v3 = poissfem_model.elements[i].v3;
            index_type v4 = poissfem_model.elements[i].v4;

            value_type x1 = poissfem_model.vertices[v1].x, y1 = poissfem_model.vertices[v1].y, z1 = poissfem_model.vertices[v1].z;
            value_type x2 = poissfem_model.vertices[v2].x, y2 = poissfem_model.vertices[v2].y, z2 = poissfem_model.vertices[v2].z;
            value_type x3 = poissfem_model.vertices[v3].x, y3 = poissfem_model.vertices[v3].y, z3 = poissfem_model.vertices[v3].z;
            value_type x4 = poissfem_model.vertices[v4].x, y4 = poissfem_model.vertices[v4].y, z4 = poissfem_model.vertices[v4].z;

            value_type vol = std::abs(
                (x2 - x1) * ((y3 - y1) * (z4 - z1) - (y4 - y1) * (z3 - z1)) -
                (x3 - x1) * ((y2 - y1) * (z4 - z1) - (y4 - y1) * (z2 - z1)) +
                (x4 - x1) * ((y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1))
            ) / 6.0f;

            value_type cx = (x1 + x2 + x3 + x4) * 0.25f;
            value_type cy = (y1 + y2 + y3 + y4) * 0.25f;
            value_type cz = (z1 + z2 + z3 + z4) * 0.25f;

            value_type u_fem_centroid = (poissfem_model.vertices[v1].potential + poissfem_model.vertices[v2].potential + 
                                         poissfem_model.vertices[v3].potential + poissfem_model.vertices[v4].potential) * 0.25f;
            value_type u_exact_centroid = analytical_solution(cx, cy, cz);
            value_type diff = u_fem_centroid - u_exact_centroid;
            
            total_l2_error_sq += (diff * diff) * vol;
        }

        l2_error = std::sqrt(total_l2_error_sq);
    }

    // Produce unified report
    Report<index_type, value_type> run_report{
        report_vertices, A.n_nonzero, max_edge_length, 
        l2_error, energy_joules, assemble_time, transfer_time, solve_time
    };

    print_report<index_type, value_type>(model_name, run_report, "");

    return 0;
}