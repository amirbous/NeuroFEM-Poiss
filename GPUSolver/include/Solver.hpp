#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "include/model.hpp"
#include "include/IO.hpp"


template<typename T_index, typename T_value>
simple_logger solveGinkgo(struct Model<T_index, T_value>& model, 
    struct CSR_matrix<T_index, T_value>& A, std::vector<T_value>& b,
    std::vector<T_value>& x0,
    int use_gpu);

#endif // SOLVER_HPP