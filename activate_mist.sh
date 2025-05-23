#!/bin/bash

# Philippe Joly 2025-05-21

# This script is designed to load all the necessary modules and activate the albatros conda environment on the MIST machine

# Run as: source activate_mist.sh

module load MistEnv/2021a anaconda3/2021.05
ARG1="${1:-albatros}"
source activate $ARG1
module load cuda/11.7.1
module load gcc/11.4.0
