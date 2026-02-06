# FEM poisson 3d fem solver and matrix generation

This part of the repo includes a complete program to solve FEM problems for poisson problem.

Initially word started in another repository and moved here for cut out commit history. 

### Requirements

- c++ v17+
- CUDA toolkit
- OpenMP (not used but required during building of the project)
- GINKGO v1.9.0+

The program stricly takes input meshes in VTK unstructured grid v5.1 with first order elements. Some examples are found under `data/mesh`

### setup and usage

* Configuration and building

This project relies on Ginkgo, which should be installed already on the system ( check https://github.com/ginkgo-project/ginkgo for installation instructions )

For the first cmake call, specify Ginkgo installation directory. It is set with variable `Ginkgo_DIR` in the cmake files.

`cmake -S -DGinkgo_DIR=<Gingko root directory> . -B build`

`cmake --build build -j`

* Running 

the program resulted binary is called `poissfem`. It takes the following command line arguments

* first argument optional: `<problem name>`. The program expects to find a mesh file with the same problem name under `data/mesh` 

* `-w <write_option>`: whether to write the solver data to a files. Possible values are:
  - 0: No write - the matrix and RHS are written to files after the run is complete
  - 1: Yes write - the matrix and RHS are written to files

* `-m <pre-assembled matrix mode>`: Note: This feature with option `1` is still not implemented and will exit. The mode is for whether to assemble the matrix from scratch or fetch it from the matrices directory. Possible values are:
  - 0: no prefetch - built the matrix from scratch
  - 1: matrix file exist - skip building and prefetch

complete_command_example: 

On top of the `poissfem` binary, a Convergence benchmark script is provided `Convergence.sh` which performs scaling tests


- The prints one log line that contains the run information. The lines are appended for benchmark tests to have coherent benchmark log reports.

The output format is the following:
(t_trans for transfer time is redundant and not measuring actual transfer time, can be ignored)

`problem_name,  n_vertices,  num_non_zeros,   max_e_length,    l2_res_norm, energy(Joules), t_assem(us), t_trans(us),   t_solve(us)`

### Structure and file roles:

`src/main.cpp`: entry point, to load problem and execute the complete flow
`src/IO.cpp`: includes all I/O kernels. Reader and writer functions for data, as well as output log formatter.
`src/Solver.cpp`: incloses a Ginkgo solver runner instance
`src/ComputeModel.cpp`: all compuet kernels (FEM and geometry)
`include/model.hpp`: structs for storing model data. 







