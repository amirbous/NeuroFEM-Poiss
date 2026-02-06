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

* Funcionality

* Configuration and building

This project relies on Ginkgo, which should be installed already on the system ( check https://github.com/ginkgo-project/ginkgo for installation instructions )

For the first cmake call, specify Ginkgo installation directory. It is set with variable `Ginkgo_DIR` in the cmake files.

`cmake -S -DGinkgo_DIR=<Gingko root directory> . -B build`

`cmake --build build -j`

* Running 

the program resulted binary is called `poissfem`. It takes the following command line arguments

* first argument optional: `<problem name>`. The program expects to find a mesh file with the same problem name under `data/mesh` 

* `-w <write_option>`: whether to write the solver data to a files.
  - 0: No write - the matrix and RHS are written to files after the run is complete
  - 1: Yes write - the matrix and RHS are written to files

* `-m <pre-assembled matrix mode>`: whether to assemble the matrix from scratch or fetch it from the matrices directory
  - 0: no prefetch - built the matrix from scratch
  - 1: matrix file exist - skip building and prefetch

On top of the `poissfem` binary, a Convergence benchmark script is provided `Convergence.sh` which performs scaling tests

- The prints one log line that contains the run information. The lines are appended for benchmark tests to have coherent benchmark log reports.

The output format is the following:









