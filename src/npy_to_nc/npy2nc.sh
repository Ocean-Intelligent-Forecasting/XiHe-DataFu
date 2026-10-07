#!/bin/bash
#Get the directory containing the current script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
folder_name=$(basename "$script_dir")

##########Validate input arguments########################

if [ $# -ne 3 ]; then
    echo "Error: three arguments are required: input path, output path, and target date"
    echo "Usage: bash $0 param1 param2 param3"
    exit 1
fi
##########0.Set parameters###############################


#Input data path, i.e. inference output path
inference_output_path=$1

#Output data path
nc_path=$2
targetdate=$3

#################################################
echo '========== Merge assimilated data into NC files =========='

#Get the current project directory
project_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#echo ${BASH_SOURCE[0]}
#echo $project_path
#echo $targetdate

#Data concatenation
python ${project_path}/src/npy2nc_concat_xarray.py --input_npy_path $inference_output_path --nc_path $nc_path --targetdate $targetdate
echo 'done!'

