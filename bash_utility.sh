#!/bin/bash

# print number of directories in ./results/ece_experiment/

num_dirs=$(ls -l ./results/ece_experiment/ | grep ^d | wc -l)
echo $num_dirs