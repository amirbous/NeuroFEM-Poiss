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
simple_logger solveGinkgo(struct Model<T_index, T_value>& model, struct CSR_matrix<T_index, T_value>& A, std::vector<T_value>& b, std::vector<T_value>& x0) {
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);
    unsigned long long energy_start{0}, energy_end{0};


        std::shared_ptr<gko::Executor> exec;
    if (gko::CudaExecutor::get_num_devices() > 0) {
        exec = gko::CudaExecutor::create(0, gko::ReferenceExecutor::create());
    } else {
        exec = gko::ReferenceExecutor::create();
    }

    auto logger = gko::share(
        gko::log::Stream<T_value>::create(
            gko::log::Logger::all_events_mask,
            std::cout
        )
    );


    exec->synchronize();

    auto transfer_start = std::chrono::steady_clock::now();

    auto gko_A = gko::share(
        gko::matrix::Csr<T_value, T_index>::create(
            exec,
            gko::dim<2>{A.n_rows, A.n_cols},
            gko::array<T_value>::view(exec, A.n_nonzero, A.values.data()),
            gko::array<T_index>::view(exec, A.n_nonzero, A.col_ind.data()),
            gko::array<T_index>::view(exec, A.n_rows + 1, A.row_ptr.data())
        )
    );

    exec->synchronize();
    auto transfer_end = std::chrono::steady_clock::now();




    auto gko_b = gko::share(
        gko::matrix::Dense<T_value>::create(
            exec,
            gko::dim<2>{A.n_rows, 1},
            gko::array<T_value>::view(exec, b.size(), b.data()),
            1
        )
    );

    auto gko_x = gko::share(
        gko::matrix::Dense<T_value>::create(
            exec,
            gko::dim<2>{A.n_rows, 1},
            gko::array<T_value>::view(exec, x0.size(), x0.data()),
            1
        )
    );

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

    nvmlDeviceGetTotalEnergyConsumption(device, &energy_start);
    auto solve_start = std::chrono::steady_clock::now();
    solver->apply(gko_b, gko_x);
    exec->synchronize();
    auto solve_end = std::chrono::steady_clock::now();
    nvmlDeviceGetTotalEnergyConsumption(device, &energy_end);

    auto solve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(solve_end - solve_start).count();
    auto transfer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(transfer_end - transfer_start).count();
    return {transfer_duration, solve_duration, static_cast<float>(energy_end - energy_start)};
}

template simple_logger
solveGinkgo<int, float>(
    Model<int, float>&,
    CSR_matrix<int, float>&,
    std::vector<float>&,
    std::vector<float>&
);

template simple_logger
solveGinkgo<int, double>(
    Model<int, double>&,
    CSR_matrix<int, double>&,
    std::vector<double>&,
    std::vector<double>&
);
