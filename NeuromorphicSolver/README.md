# Neuromorphic study

This part includes the neuromorphic study.

The chip complete pipeline as well as scripts for other areas

### Requirements

- python3 v3.10+
- py-spinnaker2 library: only relevant for solving on the chip but will be introduced more in detail later.


### py-spinnaker2 setup

In our work, we are provide the backend for the spinnaker2 chip library
and a seperate model neurofem_2048.


For the setup, we place the neurofem_2048 model files next to the other models. Compile it and it to the py-spinnaker model build list so it is recognized. 

A step by step guilde:


* building neurofem_2048 
1. Given the py-spinnaker2 folder, with the complete `libs/` correctly placed under `src/spinnaker2/`, we place `neurofem_2048` under `py-spinnaker2/src/spinnaker2/libs/chip/app-pe/s2app`. then go to `py-spinnaker2/src/spinnaker2/libs/chip/appe-pe`, make clean and run `make ROUTING_SETTINGS="C2C_ROUTING` to build neurofem_2048 with the correct flags.

After that, go to `setup.py` in py-spinnaker2 and add `neurofem_2048` in the models list to setup.

From there the rest of the installation is the same like in the moodle.



### Relevant files and usage

All data is loaded from the root data/ folder. Only problem name is required, and the scripts retrieve the data from the upper directory. 

* `CSR2SNN_pipeline`: main file that executes the pipeline and solve on the Spinnaker2 chip.
* `numerical analysis`: python3 script that takes a problem name as first single command line argument and runs a numerical analysis on the matrix, in order to estimate whether the chip will converge or not. 
* `neurosolve_theoretical`: Put together from sandilabs script. The script runs an instance the theoretical neurofem model for the problem matrix. The problem name is specified as the first command line arguments. 
* `generate_custom_matrices.py`: generates 2d and 3d poisson stencil matrices in the data format and writes them to data in the parents folder.
* `cuda_scipy.py` solves a system matrix on the GPU, using conjugate gradient cupy solver. Requires cuda toolkit and  pynvml (installation via pip)


the program resulted binary is called `poissfem`. It takes the following command line arguments

### Running CSR_pipeline.py

Like other scripts in the repository 






