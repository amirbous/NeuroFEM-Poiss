#ifndef IO_HPP
#define IO_HPP

#include "include/model.hpp"

#include <stdio.h>
#include <string>
#include <vector>


void DisplayWelcomeHeader();
template<typename T_index, typename T_value>
void ReadVTK(std::string model_name, Model<T_index, T_value> &model);


template<typename T_index, typename T_value>
void write_vtu(const std::string problem_name,
               Model<T_index, T_value> &model);

template<typename T_index, typename T_value>
void WriteCSRMatrix(const CSR_matrix<T_index, T_value> &A, std::string model_name);

template<typename T_index, typename T_value>
void WriteVector(const std::vector<T_value> &b, const std::string problem_name, std::string vector_file_desc);


template<typename T_index, typename T_value>
std::vector<T_value> ReadVector(const std::string model_name, const std::string vector_file_desc);

template<typename T_index, typename T_value>
void print_report(std::string problem_name, Report<T_index, T_value>, std::string log_file);


template<typename T_index, typename T_value>
void readCOOMatrix(const std::string filename, CSR_matrix<T_index, T_value> &A);

#endif
