#!/bin/bash

# Philippe Joly 2025-05-21
# This script is to setup a standard Python virtual environment on the Niagara machine compatible with the albatros_analysis repository

show_help() {
        cat << EOF
Usage: 
	source niagara_build.sh [OPTIONS] <environment name>

Options:
        -h, --help      Shows help

Description:
        This script sets up a standard Python virtual environment on the Niagara machine comparible with the albatros_analysis repository.
        The environment is setup in the current directory.
        A environment log will be generated in ./env/niagara_<name>_environment.txt outlining every packages, modules loaded, and system properties.
EOF
}


if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        show_help 
else

	echo "Creating Python virtual environment..."
	module load NiaEnv/2019b
	ARG1="${1:-pythonEnv}"

	python3 -m venv $ARG1
	# At the creation of the script, it roots to Python 3.6.8

	cd $ARG1
	source ./bin/activate
	echo "$ARG1 Python virtual environment successfully created"
	echo "Installing packages..."

	python -m pip install pip==21.3.1
	python -m pip install numpy==1.19.5 astropy==4.1 pandas==1.1.5 notebook==6.4.10 skyfield==1.53 matplotlib==3.3.4 scipy==1.5.4
	if [ ! -d "../env" ]; then
	  mkdir "../env"
	fi
	LOG_FILE="../env/niagara_${ARG1}_environment.txt"
	echo "Environment: $ARG1" > $LOG_FILE
	echo "Created on: $(date)" >> $LOG_FILE
	echo "" >> $LOG_FILE

	# Log pip packages
	echo "=== Pip Packages ===" >> $LOG_FILE
	pip list >> $LOG_FILE

	# Log module versions
	echo "" >> $LOG_FILE
	echo "=== Loaded Modules ===" >> $LOG_FILE
	module list 2>> $LOG_FILE

	# Log system information
	echo "" >> $LOG_FILE
	echo "=== System Information ===" >> $LOG_FILE
	echo "Python Version: $(python --version)" >> $LOG_FILE

	echo "Environment details saved to ./env/niagara_${ARG1}_environment.txt"

	deactivate 
	cd ..
fi
