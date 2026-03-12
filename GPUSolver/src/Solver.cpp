#include "include/Solver.hpp"
#include "include/model.hpp"
#include "include/IO.hpp"

#include <iostream>
#include <string>
#include <set>
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>

#include <nvml.h>

#include <ginkgo/ginkgo.hpp>


template<typename T_index, typename T_value>
simple_logger solveGinkgo(struct Model<T_index, T_value>& model, 
                         struct CSR_matrix<T_index, T_value>& A, 
                         std::vector<T_value>& b, 
                         std::vector<T_value>& x0, 
                         int use_gpu) {
    
    nvmlDevice_t device;
    unsigned long long energy_start{0}, energy_end{0};

    if (use_gpu) {
        nvmlInit();
        nvmlDeviceGetHandleByIndex(0, &device);
    }

    std::shared_ptr<gko::Executor> exec;

    if (use_gpu && gko::CudaExecutor::get_num_devices() > 0) {
        std::cout << "Using CUDA executor." << std::endl;
        exec = gko::CudaExecutor::create(0, gko::ReferenceExecutor::create());
    } else {
        std::cout << "Using Reference (CPU) executor." << std::endl;
        exec = gko::ReferenceExecutor::create();
    }

    auto host = gko::ReferenceExecutor::create();

    auto logger = gko::share(
        gko::log::Stream<T_value>::create(
            gko::log::Logger::all_events_mask,
            std::cout
        )
    );

    exec->synchronize();
    auto transfer_start = std::chrono::steady_clock::now();

    /* ---- FIX: construct matrix on host then clone to executor ---- */

auto gko_A_host = gko::matrix::Csr<T_value, T_index>::create(
    host,
    gko::dim<2>{A.n_rows, A.n_cols},
    gko::array<T_value>::view(host, A.n_nonzero, A.values.data()),
    gko::array<T_index>::view(host, A.n_nonzero, A.col_ind.data()),
    gko::array<T_index>::view(host, A.n_rows + 1, A.row_ptr.data())
);

auto gko_A = gko::share(gko::clone(exec, gko_A_host));

auto gko_b_host = gko::matrix::Dense<T_value>::create(
    host,
    gko::dim<2>{A.n_rows, 1},
    gko::array<T_value>::view(host, b.size(), b.data()),
    1
);

auto gko_x_host = gko::matrix::Dense<T_value>::create(
    host,
    gko::dim<2>{A.n_rows, 1},
    gko::array<T_value>::view(host, x0.size(), x0.data()),
    1
);

auto gko_b = gko::share(gko::clone(exec, gko_b_host));
auto gko_x = gko::share(gko::clone(exec, gko_x_host));


    exec->synchronize();
    auto transfer_end = std::chrono::steady_clock::now();

    /* -------------------------------------------------------------- */

    auto solver_gen = gko::solver::Cg<T_value>::build()
        .with_criteria(
            gko::stop::Iteration::build()
                .with_max_iters(1000)
                .on(exec),
            gko::stop::ResidualNorm<T_value>::build()
                .with_reduction_factor(1e-6)
                .on(exec)
        )
        .with_preconditioner(
            gko::preconditioner::Jacobi<T_value>::build().on(exec)
        )
        .on(exec);

    auto solver = solver_gen->generate(gko_A);

    exec->synchronize();

    if (use_gpu) {
        nvmlDeviceGetTotalEnergyConsumption(device, &energy_start);
    }

    auto solve_start = std::chrono::steady_clock::now();

    solver->apply(gko_b, gko_x);

    exec->synchronize();
    auto solve_end = std::chrono::steady_clock::now();

    if (use_gpu) {
        nvmlDeviceGetTotalEnergyConsumption(device, &energy_end);
        nvmlShutdown();
    }

    /* ---- copy solution back to host vector x0 ---- */

    auto gko_x_result = gko::clone(host, gko_x);
    auto vals = gko_x_result->get_values();

    for (size_t i = 0; i < x0.size(); ++i) {
        x0[i] = vals[i];
    }

    /* ---------------------------------------------- */

    double energy_mJ = (use_gpu) ? static_cast<double>(energy_end - energy_start) : 0.0;
    double solve_time_s = std::chrono::duration<double>(solve_end - solve_start).count();

    if (use_gpu) {
        std::cout << "Solver finished. Energy consumed: " << energy_mJ << " mJ" << std::endl;
        std::cout << "Power consumption: " << energy_mJ / solve_time_s << " mW" << std::endl;
    }

    auto solve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - solve_start).count();
    auto transfer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(transfer_end - transfer_start).count();
    
    return {
        static_cast<float>(transfer_duration), 
        static_cast<float>(solve_duration), 
        static_cast<float>(energy_mJ)
    };
}

template simple_logger
solveGinkgo<int, float>(
    Model<int, float>&,
    CSR_matrix<int, float>&,
    std::vector<float>&,
    std::vector<float>&, 
    int use_gpu
);

template simple_logger
solveGinkgo<int, double>(
    Model<int, double>&,
    CSR_matrix<int, double>&,
    std::vector<double>&,
    std::vector<double>&, 
    int use_gpu
);
