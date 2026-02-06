# NeuroFEM-Poiss

This repository is for the practical course, Future computing lab at TUM.

The project covers a FEM problem matrix generation program in C++, and matrix solution in Ginkgo.

The problem matrices are then used for the Neuromorphic hardware study on the Spinnaker2

### Repo structure 

#### Each of the subdirectories includes its own readme. Different functionalities are split over the directories therefore each subdirectory has its own readme explaining the functionality and the setup of the projects part. 

#### Please refer to directories GPUSolver and NeuromorphicSolver for more details. 

* `data`: includes simulation data. /mesh/: vtk mesh data for generating fem matrix, /matrix: matrices and rhs written into files
* `GPUSolver`: C++ oracle to generate and solve Finite element meshes given 3d unstructred tet
* `NeuromorphicSolver`: all files relevant to any processing 
* `logs_results` subdirectory to save results in text formats.
* `Analyse..._results.ipynb`: notebooks to visualize the results. 

### current state and TODO checks

Realized so far:

- [x] Framework to generate first order FEM for poisson problem with tetrahedra (3D) discretization
- [x] sparse matrices GPU solver using Ginkgo
- [x] pipeline for off solving on spinnaker2

Next steps 

- [ ] Numerical discussion and guards





