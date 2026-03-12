#include "include/IO.hpp"
#include "include/model.hpp"

#include <iostream>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <fstream>
#include <string>
#include <iomanip>

void DisplayWelcomeHeader()
{

        std::cout << "**************************************" << std::endl;
        std::cout << "*          3D FEM POISSON            *" << std::endl;
        std::cout << "**************************************" << std::endl;
        std::cout << std::endl;
}


template<typename T_index, typename T_value>
void print_report(std::string problem_name, const Report<T_index, T_value> report, std::string log_file)
{
    if (log_file.compare("") == 0) {
        // print to screen
        std::cout << std::right << std::setw(15) << problem_name <<",";
        std::cout << std::right << std::setw(12) << report.n_vertices <<",";
        std::cout << std::right << std::setw(15) << report.n_nonzero <<",";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(6) << report.max_edge_length <<",";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(6) << report.residual_norm << ",";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(6) << report.energy_joules << ",";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(6) << report.assemble_time << ",";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(6) << report.transfer_time << ",";
        std::cout << std::right << std::setw(15) << std::fixed << std::setprecision(6) << report.solve_time;

        std::cout << std::endl;
    }
    else {
        // print to file
        std::ofstream fstream;
        fstream.open(log_file, std::ios_base::app); 

        fstream << std::right << std::setw(15) << problem_name <<",";
        fstream << std::right << std::setw(12) << report.n_vertices <<",";
        fstream << std::right << std::setw(15) << report.n_nonzero <<",";
        fstream << std::right << std::setw(15) << std::fixed << std::setprecision(6) << report.max_edge_length <<",";
        fstream << std::right << std::setw(15) << std::fixed << std::setprecision(6) << report.residual_norm << ",";
        fstream << std::right << std::setw(15) << std::fixed << std::setprecision(6) << report.assemble_time << ",";
        fstream << std::right << std::setw(15) << std::fixed << std::setprecision(6) << report.transfer_time << ",";
        fstream << std::right << std::setw(15) << std::fixed << std::setprecision(6) << report.solve_time;

        fstream << std::endl;
        fstream.close();

    }

}


// TODO: adapt for tetrahedral elements - DONE
template<typename T_index, typename T_value>
void ReadVTK(std::string model_name, Model<T_index, T_value> &model)
{

        std::string vtk_filename = model_name + ".vtk";
        std::string vtk_file = "data/mesh/" + vtk_filename;
        std::ifstream vtk_file_stream(vtk_file);


        std::vector<Vertex<T_index, T_value>>& vertices = model.vertices;
        std::vector<Element<T_index>>& elements = model.elements;

        if (!vtk_file_stream.is_open())
        {
                std::cerr << "Error: Could not open VTK file " << vtk_filename << std::endl;
                return;
        }

        std::string buffer;

        std::vector<T_index> offsets;
        std::vector<T_index> connectivity_raw;
        T_index n_cells_header = 0;

        while (vtk_file_stream >> buffer)
        {

                if (buffer == "POINTS")
                {
                        T_index n_points;
                        std::string type;
                        vtk_file_stream >> n_points >> type;

                        model.n_vertices = n_points;
                        vertices.resize(n_points);

                        for (T_index i = 0; i < n_points; ++i)
                        {
                                T_value x, y, z;
                                vtk_file_stream >> x >> y >> z;
                                vertices[i] = Vertex<T_index, T_value>(x, y, z, i);
                        }
                }

                else if (buffer == "CELLS")
                {
                        T_index size_dummy;
                        vtk_file_stream >> n_cells_header >> size_dummy;
                        n_cells_header --;
                        model.n_elements = n_cells_header;
                        elements.reserve(n_cells_header);
                }


                else if (buffer == "OFFSETS")
                {
                        std::string type;
                        vtk_file_stream >> type;
                        offsets.resize(n_cells_header + 1);

                        for (T_index i = 0; i <= n_cells_header; ++i)
                        {
                                vtk_file_stream >> offsets[i];
                        }
                }

                else if (buffer == "CONNECTIVITY")
                {
                        std::string type;
                        vtk_file_stream >> type; 


                        unsigned long int total_indices = offsets.back();
                        connectivity_raw.resize(total_indices);
                        for (T_index i = 0; i < total_indices; ++i)
                        {
                                vtk_file_stream >> connectivity_raw[i];
                        }


                        for (T_index i = 0; i < n_cells_header; ++i)
                        {
                                T_index start = offsets[i];
                                T_index end = offsets[i + 1];
                                T_index count = end - start;
                                if (count == 4)
                                {

                                        T_index v1 = static_cast<T_index>(connectivity_raw[start]);
                                        T_index v2 = static_cast<T_index>(connectivity_raw[start + 1]);
                                        T_index v3 = static_cast<T_index>(connectivity_raw[start + 2]);
                                        T_index v4 = static_cast<T_index>(connectivity_raw[start + 3]);

                                        elements.emplace_back(v1, v2, v3, v4);
                                }
                                else
                                {
                                        std::cerr << "Offset did not match a tetrahedra element: Cell " << i << " has " << count << " vertices " << std::endl;
                                        return;
                                }
                        }
                        connectivity_raw.clear();
                        connectivity_raw.shrink_to_fit();

                        offsets.clear();
                        offsets.shrink_to_fit();
                }
        }
        
        vtk_file_stream.close();
}




// TODO: adapt for tetrahedral elements - DONE
template<typename T_index, typename T_value>
void write_vtu(const std::string problem_name,
               Model<T_index, T_value> &model)
{
    std::ofstream fstream;
    std::string fname = problem_name + "_output.vtu";

    std::vector<Vertex<T_index, T_value>>& vertices = model.vertices;
    std::vector<Element<T_index>>& elements = model.elements;

    int nvertices = model.n_vertices;
    int nelements = model.n_elements; 

    fstream.open(fname);
    if (fstream.is_open())
    {
        fstream << "<?xml version=\"1.0\"?>" << std::endl;
        fstream << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\"";
        fstream << " byte_order=\"LittleEndian\" header_type=\"UInt64\">" << std::endl;
        fstream << "<UnstructuredGrid>" << std::endl;
        fstream << "<Piece NumberOfPoints=\"" << nvertices << "\"  ";
        fstream << "NumberOfCells=\"" << nelements << "\">" << std::endl;
        

        fstream << "<Points>" << std::endl;
        fstream << "<DataArray type=\"Float32\" Name=\"Points\"";
        fstream << " NumberOfComponents=\"3\" Format=\"ascii\">" << std::endl;

        for (size_t i = 0; i < nvertices; i++)
        {
            fstream << std::fixed << std::setprecision(7) << vertices[i].x
                    << " " << std::fixed << std::setprecision(7) << vertices[i].y
                    << " " << std::fixed << std::setprecision(7) << vertices[i].z
                    << std::endl;
        }

        fstream << "</DataArray>" << std::endl;
        fstream << "</Points>" << std::endl;


        fstream << "<Cells>" << std::endl;

        fstream << "<DataArray type=\"Int64\" ";
        fstream << "Name=\"connectivity\" format=\"ascii\">" << std::endl;

        for (size_t i = 0; i < nelements; i++)
        {

            fstream << elements[i].v1
                    << " " << elements[i].v2
                    << " " << elements[i].v3
                    << " " << elements[i].v4
                    << std::endl;
        }
        fstream << "</DataArray>" << std::endl;

 
        fstream << "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">" << std::endl;
        
        long long current_offset = 0;
        for (int i = 0; i < nelements; i++)
        {
            current_offset += 4; 
            fstream << current_offset << " ";
            if ((i + 1) % 20 == 0) fstream << std::endl;
        }
        fstream << std::endl;
        fstream << "</DataArray>" << std::endl;

        fstream << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << std::endl;

        for (size_t i = 0; i < nelements; i++)
        {
            fstream << "10" << " ";
            if ((i + 1) % 20 == 0) fstream << std::endl;
        }
        fstream << std::endl;
        fstream << "</DataArray>" << std::endl;
        fstream << "</Cells>" << std::endl;


        fstream << "<PointData>" << std::endl;

        fstream << "<DataArray type=\"Float32\" Name=\"Potential\" ";
        fstream << "NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
        for (int i = 0; i < nvertices; i++)
        {
            fstream << vertices[i].potential << std::endl;
        }
        fstream << "</DataArray>" << std::endl;


        fstream << "<DataArray type=\"Float32\" Name=\"density\" ";
        fstream << "NumberOfComponents=\"1\" format=\"ascii\">" << std::endl;
        for (int i = 0; i < nvertices; i++)
        {
            fstream << vertices[i].density << std::endl;
        }
        fstream << "</DataArray>" << std::endl;

        fstream << "</PointData>" << std::endl;

        // --- Footer ---
        fstream << "</Piece>" << std::endl;
        fstream << "</UnstructuredGrid>" << std::endl;
        fstream << "</VTKFile>";

        fstream.close();
    }
    else
    {
        std::cerr << "Error in opening result file" << std::endl;
    }
}


template<typename T_index, typename T_value>
void WriteVector(const std::vector<T_value> &b, const std::string model_name, std::string vector_file_desc) {
    std::ofstream fstream;
    std::string fname = model_name + "_" + vector_file_desc + ".txt";

    fstream.open(fname);

    fstream << b.size() << std::endl;
    if (fstream.is_open()) {
        fstream << std::scientific << std::setprecision(16);
        for (size_t i = 0; i < b.size(); ++i) {
            fstream << b[i] << std::endl;
        }
        fstream.close();
    } else {
        std::cerr << "Error in opening RHS file" << std::endl;
    }
}

template<typename T_index, typename T_value>
std::vector<T_value> ReadVector(const std::string model_name, const std::string vector_file_desc) {
    std::ifstream fstream;
    std::string fname = model_name + "_" + vector_file_desc + ".txt";

    std::string filepath = "data/matrix/" + fname;
    fstream.open(filepath);

    if (!fstream.is_open()) {
        std::cerr << "Error: Could not open RHS file " << fname << std::endl;
        return {};
    }

    size_t size;
    fstream >> size;
    std::vector<T_value> x(size);

    for (size_t i = 0; i < size; ++i) {
        fstream >> x[i];
    }

    fstream.close();
    return x;
}


template<typename T_index, typename T_value>
void WriteCSRMatrix(const CSR_matrix<T_index, T_value> &A, std::string model_name) {

    // csr format is 1-based indexing
    std::ofstream fstream;
    std::string fname = model_name + "_mtx.txt";

    fstream.open(fname);
    if (!fstream.is_open()) {
        std::cerr << "Error: Could not open file to write CSR matrix." << std::endl;
        return;
    }

    fstream << A.n_rows << " " << A.n_cols << " " << A.row_ptr[A.n_rows] << std::endl;


    fstream << std::scientific << std::setprecision(16);

    for (T_index i = 0; i < A.n_rows; ++i) {
        T_index start = A.row_ptr[i];
        T_index end = A.row_ptr[i + 1];

        for (T_index j = start; j < end; ++j) {
                
                fstream << std::setw(5) << (i + 1) << " " 
                            << std::setw(5) << (A.col_ind[j] + 1) << "    " 
                            << A.values[j] << std::endl;

        }
    }

    fstream.close();
}


template<typename T_index, typename T_value>
void readCOOMatrix(const std::string filename, CSR_matrix<T_index, T_value> &A) {
    std::string filepath = "data/matrix/" + filename;
    std::ifstream fstream(filepath);

    if (!fstream.is_open()) {
        std::cerr << "Error: Could not open COO matrix file " << filename << std::endl;
        return;
    }

    int n_rows, n_cols, n_nonzero;
    // Read the header: rows, columns, non-zeros
    if (!(fstream >> n_rows >> n_cols >> n_nonzero)) {
        std::cerr << "Error: Could not read matrix header." << std::endl;
        return;
    }

    // Initialize the CSR matrix using your existing constructor
    A = CSR_matrix<T_index, T_value>(n_rows, n_cols, n_nonzero);

    // Temporary arrays to hold the COO data
    std::vector<T_index> coo_row(n_nonzero);
    std::vector<T_index> coo_col(n_nonzero);
    std::vector<T_value> coo_val(n_nonzero);
    
    // Array to track how many non-zeros exist per row
    std::vector<T_index> nnz_per_row(n_rows, 0);

    for (int i = 0; i < n_nonzero; ++i) {
        T_index r, c;
        T_value v;
        fstream >> r >> c >> v;

        // Convert 1-based index from the file to 0-based index for C++
        r--;
        c--;

        coo_row[i] = r;
        coo_col[i] = c;
        coo_val[i] = v;

        nnz_per_row[r]++;
    }

    // 1. Compute row_ptr using a prefix sum
    A.row_ptr[0] = 0;
    for (int i = 0; i < n_rows; ++i) {
        A.row_ptr[i + 1] = A.row_ptr[i] + nnz_per_row[i];
    }

    // 2. Scatter the COO data into the CSR arrays
    // We make a copy of row_ptr to track insertion offsets dynamically 
    std::vector<T_index> current_row_offset = A.row_ptr;

    for (int i = 0; i < n_nonzero; ++i) {
        T_index r = coo_row[i];
        T_index dest = current_row_offset[r]++;
        A.col_ind[dest] = coo_col[i];
        A.values[dest] = coo_val[i];
    }

    fstream.close();
}

template void readCOOMatrix<int, float>(const std::string filename, CSR_matrix<int, float> &A);
template void readCOOMatrix<int, double>(const std::string filename, CSR_matrix<int, double> &A);


template void ReadVTK<int, float>(std::string model_name, Model<int, float> &model);
template void write_vtu<int, float>(const std::string model_name,
                                   Model<int, float> &model);
template void ReadVTK<int, double>(std::string model_name, Model<int, double> &model);
template void write_vtu<int, double>(const std::string model_name,
                                    Model<int, double> &model);

template void WriteCSRMatrix<int, float>(const CSR_matrix<int, float> &A, std::string model_name);
template void WriteCSRMatrix<int, double>(const CSR_matrix<int, double> &A, std::string model_name);

template void WriteVector<int, float>(const std::vector<float> &b, const std::string problem_name, std::string vector_file_desc);
template void WriteVector<int, double>(const std::vector<double> &b, const std::string problem_name, std::string vector_file_desc);

template std::vector<float> ReadVector<int, float>(const std::string model_name, const std::string vector_file_desc);
template std::vector<double> ReadVector<int, double>(const std::string model_name, const std::string vector_file_desc);

template void print_report<int, float>(std::string problem_name, Report<int, float>, std::string log_file);
template void print_report<int, double>(std::string problem_name, Report<int, double>, std::string log_file);