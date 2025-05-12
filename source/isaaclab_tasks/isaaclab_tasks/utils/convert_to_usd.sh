#!/bin/bash

# Directory containing OBJ files
OBJ_DIR="/tmp/sq_obj"

# Create the OBJ directory if it doesn't exist
rm -rf $OBJ_DIR
python source/isaaclab_tasks/isaaclab_tasks/utils/create_sq_mesh.py $OBJ_DIR

# Output directory for USD files
USD_DIR="source/isaaclab_assets/data/Props/Superquadrics"
mkdir -p "$USD_DIR"

# Loop through each OBJ file and convert to USD
for obj_file in "$OBJ_DIR"/*.obj; do
    filename=$(basename "$obj_file" .obj)
    usd_file="$USD_DIR/${filename}.usd"

    echo "Converting $obj_file to $usd_file"
    DOCKER_ISAACLAB_PATH=. ./isaaclab.sh -p scripts/tools/convert_mesh.py \
        "$obj_file" \
        "$usd_file" \
        --collision-approximation convexDecomposition \
        --mass 1.0 --headless
done
