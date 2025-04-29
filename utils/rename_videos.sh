#!/bin/bash

# Define the destination directory
dest_dir="../Figures and videos"

for i in {1..18}; do
    # Format the source file name
    src="polygon_points${i}.mp4"
    # Format the destination file name
    dest="${dest_dir}/"
    
    # Move and rename the file
    mv "$src" "$dest"
done
