#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi
input_file=$1

export PYTHONUNBUFFERED=1

num_rows=$(wc -l < "$input_file")
for (( i=1; i<=$num_rows; i++ )); do
    echo "Processing row $i"
    plot_ci_rates "$input_file" -i $i
done