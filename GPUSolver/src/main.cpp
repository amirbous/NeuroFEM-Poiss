#include <iostream>
#include <string>
#include <set>
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>


#include "include/IO.hpp"
#include "include/model.hpp"
#include "include/ComputeModel.hpp"

#include <ginkgo/ginkgo.hpp>


int main(int argc, char* argv[])  {

    using index_type = int;
    using value_type = float;

    bool write_matrix = false;

    std::string model_name{};

    Model<index_type, value_type> poissfem_model;
    CSR_matrix<index_type, value_type> A;

    std::vector<value_type> b; // we still don't know the size, given the sparsity pattern but will be decided in initialize_CSR_indices
    std::vector<value_type> x0; // solution vector
    model_name = (argc > 1 ? argv[1] : "Sphere_00");
    write_matrix = (argc > 2 ? (std::string(argv[2]) == "1" ? true : false) : false);

    ReadVTK(model_name, poissfem_model);


    auto assemble_begin = std::chrono::steady_clock::now();
    initialize_CSR_indices<index_type, value_type>(poissfem_model, A);
    fill_FEM_CSR<index_type, value_type>(poissfem_model, A, b);
    auto assemble_end = std::chrono::steady_clock::now();
    long assemble_time = std::chrono::duration_cast<std::chrono::microseconds>(assemble_end - assemble_begin).count();
    //******************************************
    //*  Solver part in ginkgo :)
    //******************************************

    std::shared_ptr<gko::Executor> exec;
    if (gko::CudaExecutor::get_num_devices() > 0) {
        exec = gko::CudaExecutor::create(0, gko::ReferenceExecutor::create());
    } else {
        exec = gko::ReferenceExecutor::create();
    }

    auto logger = gko::share(
        gko::log::Stream<value_type>::create(
            gko::log::Logger::all_events_mask,
            std::cout
        )
    );


    exec->synchronize();

    auto transfer_start = std::chrono::steady_clock::now();

    auto gko_A = gko::share(
        gko::matrix::Csr<value_type, index_type>::create(
            exec,
            gko::dim<2>{A.n_rows, A.n_cols},
            gko::array<value_type>::view(exec, A.n_nonzero, A.values.data()),
            gko::array<index_type>::view(exec, A.n_nonzero, A.col_ind.data()),
            gko::array<index_type>::view(exec, A.n_rows + 1, A.row_ptr.data())
        )
    );

    exec->synchronize();
    auto transfer_end = std::chrono::steady_clock::now();


    x0.assign(A.n_rows, 0.0);

    auto gko_b = gko::share(
        gko::matrix::Dense<value_type>::create(
            exec,
            gko::dim<2>{A.n_rows, 1},
            gko::array<value_type>::view(exec, b.size(), b.data()),
            1
        )
    );

    auto gko_x = gko::share(
        gko::matrix::Dense<value_type>::create(
            exec,
            gko::dim<2>{A.n_rows, 1},
            gko::array<value_type>::view(exec, x0.size(), x0.data()),
            1
        )
    );

    auto solver_gen = gko::solver::Cg<value_type>::build()
        .with_criteria(
            gko::stop::Iteration::build()
                .with_max_iters(1000)
                .on(exec),
            gko::stop::ResidualNorm<value_type>::build()
                .with_reduction_factor(1e-6)
                .on(exec)
        )
        .with_preconditioner(
            gko::preconditioner::Jacobi<value_type>::build().on(exec)
        )
        .on(exec);


    auto solver = solver_gen->generate(gko_A);


    exec->synchronize();

    auto solve_start = std::chrono::steady_clock::now();
    solver->apply(gko_b, gko_x);
    exec->synchronize();
    auto solve_end = std::chrono::steady_clock::now();


    long transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(transfer_end - transfer_start).count();
    long solve_time = std::chrono::duration_cast<std::chrono::microseconds>(solve_end - solve_start).count();


    std::vector<index_type> boundary_nodes = extract_boundary_nodes<index_type, value_type>(poissfem_model);

    std::vector<value_type> x_analytical;
    x_analytical.assign(x0.size(), 0);


    std::vector<bool> is_boundary(poissfem_model.n_vertices, false);
    for (index_type idx : boundary_nodes) {
        is_boundary[idx] = true;
    }



    index_type internal_idx = 0; 


    for (int i = 0; i < poissfem_model.n_vertices; ++i) {
            // Get coordinates for the current node
        value_type vx = poissfem_model.vertices[i].x;
        value_type vy = poissfem_model.vertices[i].y;
        value_type vz = poissfem_model.vertices[i].z;

        if (!is_boundary[i]) {
            // Assign computed solution to model
            poissfem_model.vertices[i].potential = x0[internal_idx];
                
            // 2. Compute and store analytical solution for this internal node
            value_type exact_val = analytical_solution<value_type>(vx, vy, vz);
            x_analytical[internal_idx] = exact_val;

            internal_idx++;
        } else {
            // Boundary nodes: set potential directly from analytical solution
            poissfem_model.vertices[i].potential = analytical_solution(vx, vy, vz);
        }
    }

    for (int i = 0; i < poissfem_model.n_vertices; ++i) {
        value_type x = poissfem_model.vertices[i].x;
        value_type y = poissfem_model.vertices[i].y;
        value_type z = poissfem_model.vertices[i].z;
        poissfem_model.vertices[i].density = analytical_solution<value_type>(x, y, z);
    }
        



    std::vector ana_sol = std::vector<value_type>(poissfem_model.n_vertices);
    for (int i = 0; i < poissfem_model.n_vertices; ++i) {
        value_type x = poissfem_model.vertices[i].x;
        value_type y = poissfem_model.vertices[i].y;
        value_type z = poissfem_model.vertices[i].z;
        ana_sol[i] = analytical_solution<value_type>(x, y, z);
    }

    value_type total_l2_error_sq{0.0};
    value_type max_edge_length{0.0};

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

        
    value_type l2_error = std::sqrt(total_l2_error_sq);

    if (write_matrix) {
        WriteCSRMatrix<index_type, value_type>(A, model_name);
        WriteVector<index_type, value_type>(b, model_name, "rhs");
        WriteVector<index_type, value_type>(x0, model_name, "x0");
    }

    Report<index_type, value_type> run_report{poissfem_model.n_vertices, A.n_nonzero, max_edge_length, l2_error, assemble_time, transfer_time, solve_time};


    print_report<index_type, value_type>(model_name, run_report, "");

    write_vtu<index_type, value_type>(model_name, poissfem_model);
    


    return 0;
}
