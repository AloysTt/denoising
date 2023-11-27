#!/bin/bash
clear

repertoire="datascene"

paths=$(find "$repertoire" -type f -name "*scene.json")

for path in $paths

do

  dir_name=$(basename $(dirname "$path"))

  mkdir -p "Output/${dir_name}/noise/target"
  mkdir -p "Output/${dir_name}/truth/target"

  python render.py -d "$path" -t tungsten/build/tungsten -n 10 -s 1 -o "Output/${dir_name}/noise"
  python render.py -d "$path" -t tungsten/build/tungsten -n 1 -s 128 -o "Output/${dir_name}/truth"

done
