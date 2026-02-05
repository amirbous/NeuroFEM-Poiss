# Neuromorphic study

This part includes the neuromorphic study.

The chip complete pipeline as well as scripts for other areas

### Requirements

- python3 v3.10+
- py-spinnaker2 library: only relevant for solving on the chip but will be introduced more in detail later.


### py-spinnaker2 setup

In our work, we are provide the backend for the spinnaker2 chip library
and a seperate model neurofem_2048.


For the setup, we place the neurofem_2048 in the models of the chip. Compile it and it to the py-spinnaker model build list so it is recognized.


* building neurofem_2048 
1. Given the py-spinnaker2 folder, with the complete `libs/` correctly placed under `src/spinnaker2/`, we place `neurofem_2048` under `libs/chip/app-pe/s2app`



### Relevant files and usage





the program resulted binary is called `poissfem`. It takes the following command line arguments

Running `poissfem`

* first argument: `<problem name>`. The program expects to find a mesh file with the same problem name under `data/mesh` 

* `-w <write_option>`: whether to write the solver data to a files.
  - 0: No write - the matrix and RHS are written to files after the run is complete
  - 1: Yes write - the matrix and RHS are written to files

On top of the `poissfem` binary, a Convergence benchmark script is provided `Convergence.sh` which performs scaling tests



* Output

- The prints one log line that contains the run information. The lines are appended for benchmark tests to have coherent benchmark log reports.

The output format is the following:









