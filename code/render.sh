#!/bin/bash
clear

repertoire="datascene"

paths=$(find "$repertoire" -type f -name "*scene.json")

for path in $paths

do

  dir_name=$(basename $(dirname "$path"))

  output_file_noise = "Output/${dir_name}/noise"
  output_file_truth = "Output/${dir_name}/truth"

  mkdir -p "Output/${dir_name}/noise"
  mkdir -p "Output/${dir_name}/truth"
  python render.py -d "$path" -t tungsten/build/tungsten -n 100 -s 1 -o "Output/${dir_name}/noise"
  python render.py -d "$path" -t tungsten/build/tungsten -n 1 -s 128 -o "Output/${dir_name}/truth"

done
