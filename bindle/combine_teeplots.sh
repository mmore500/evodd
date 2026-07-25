#!/bin/bash

################################################################################
echo
echo "running combine_teeplots.sh"
echo "---------------------------------------------"
################################################################################

# fail on error
set -e

################################################################################
echo
echo "other initialization"
echo "--------------------"
################################################################################

# adapted from https://stackoverflow.com/a/24114056
script_dir="$(dirname -- "$BASH_SOURCE")"
echo "script_dir ${script_dir}"

################################################################################
echo
echo "combine each teeplots subdirectory into one pdf"
echo "-------------------------------------------------"
################################################################################

python3 "${script_dir}/combine_teeplots.py" "${script_dir}/teeplots"

################################################################################
echo
echo "recurse to subdirectories"
echo "-------------------------"
################################################################################

shopt -s nullglob

for script in "${script_dir}/"*/combine_teeplots.sh; do
  "${script}"
done

shopt -u nullglob
