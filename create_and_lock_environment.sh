#!/bin/bash

# Read input file from argument 1, default to primary_deps.yml
input_file="primary_deps.yml"
output_file="environment.yml"
if [ "$#" -gt 0 ]; then
    input_file=$1
    echo "Using input file: $input_file"
fi
if [ "$#" -gt 1 ]; then
    output_file=$2
    echo "Using output file: $output_file"
fi

os_name=$(uname)
if [[ ! $os_name == *"Linux"* ]]; then
    echo "ERROR: cannot run environment locking in non-linux environment. Windows users can do this using WSL - https://docs.microsoft.com/en-us/windows/wsl/install"
    exit 1
else
    echo "Starting environment locking..."
fi

# get environment name from primary dependencies YAML file
name_line="$(cat $input_file | grep 'name:')"
IFS=':' read -ra name_arr <<< "$name_line"
env_name="${name_arr[1]}"

echo "Building Conda environment: $env_name"

# clear old conda envs, create new one
export CONDA_ALWAYS_YES="true"
conda activate base
conda env remove --name $env_name
conda env create --file $input_file

# export new environment to environment.yml
echo "Exporting environment $env_name to environment.tmp1"
conda env export -n $env_name | grep -v "prefix:" > environment.tmp1
unset CONDA_ALWAYS_YES

echo "Removing version hash from environment.tmp1"
# remove python version hash (technically not locked, so still potential for problems here if python secondary deps change)
while IFS='' read -r line; do
    if [[ $line == *"- python="* ]]; then

        IFS='=' read -ra python_arr <<< "$line"
        unset python_arr[-1]
        echo "${python_arr[0]}"="${python_arr[1]}"
    elif [[ ! $line == "#"* ]]; then
        echo "${line}"
    fi
done < environment.tmp1 > environment.tmp2
echo "Creating final environment.yml with warning line"
echo "# WARNING - DO NOT EDIT THIS FILE MANUALLY" > $output_file
echo "# To update, please modify '$input_file' and then run the locking script 'create_and_lock_environment.sh'">> $output_file
cat environment.tmp2 >> $output_file
rm environment.tmp1 environment.tmp2
