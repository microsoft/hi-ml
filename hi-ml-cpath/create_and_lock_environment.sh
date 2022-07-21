#!/bin/bash

os_name=$(uname)
if [[ ! $os_name == *"Linux"* ]]; then
    echo "ERROR: cannot run environment locking in non-linux environment. Windows users can do this using WSL - https://docs.microsoft.com/en-us/windows/wsl/install"
    exit 1
else
    echo "Starting environment locking..."
fi

# get environment name from primary dependencies YAML file
name_line="$(cat primary_deps.yml | grep 'name:')"
IFS=':' read -ra name_arr <<< "$name_line"
env_name="${name_arr[1]}"

echo "Building Conda environment: $env_name"

# clear old conda envs, create new one
export CONDA_ALWAYS_YES="true"
conda env remove --name $env_name
conda env create --file primary_deps.yml

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
echo "# WARNING - DO NOT EDIT THIS FILE MANUALLY" > environment.yml
echo "# To update, please modify `primary_deps.yml` and then run the locking script `create_and_lock_environment.sh`">> environment.yml
cat environment.tmp2 >> environment.yml
rm environment.tmp1 environment.tmp2
