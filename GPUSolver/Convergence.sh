#!/usr/bin/env bash

Base_problem_name=$1
number_of_problems=$2
write_data=$3
last_problem=$((number_of_problems - 1))


echo ${last_problem}
echo "   problem_name,  n_vertices,  num_non_zeros,   max_e_length,   
             l2_res_norm, energy(J), assemble_time(ms), ransfer_time(ms), solve_time(ms)"


    
for i in $(seq 0 ${last_problem}); do
for repeat in $(seq 1 5); do
    complete_name=${Base_problem_name}${i}
    ./poissfem ${complete_name} ${write_data}

done
    
done
