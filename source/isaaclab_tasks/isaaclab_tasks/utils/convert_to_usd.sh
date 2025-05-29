#!/bin/bash

set -e

# Directory containing generated OBJ files
OBJ_DIR="/tmp/sq_obj"

# Clean up any old mesh outputs
rm -rf "$OBJ_DIR"

# Generate superquadric OBJ meshes
python source/isaaclab_tasks/isaaclab_tasks/utils/create_sq_mesh.py "$OBJ_DIR"

# Target base directory for USD outputs
USD_BASE_DIR="source/isaaclab_assets/data/Props/Superquadrics"

rm -rf "$USD_BASE_DIR"
# Loop through each OBJ file and convert to USD
for obj_file in "$OBJ_DIR"/*.obj; do
    obj_name=$(basename "$obj_file" .obj)
    out_dir="$USD_BASE_DIR/$obj_name"
    usd_file="$out_dir/object.usd"
    usda_file="$out_dir/object.usda"
    abs_geom_ref="${out_dir}/Props/instanceable_meshes.usd"

    mkdir -p "$out_dir"

    echo "Converting $obj_file → $usd_file"

    # Use --flatten to avoid broken references
    DOCKER_ISAACLAB_PATH=. ./isaaclab.sh -p scripts/tools/convert_mesh.py \
        "$obj_file" \
        "$usd_file" \
        --collision-approximation convexDecomposition \
        --mass 1.0 \
        --headless
        # --make-instanceable \

    # echo "Converting $usd_file → $usda_file"
    # usdcat ${usd_file} --out ${usda_file}

    # echo "Referring to $abs_geom_ref in $usda_file"
    # sed -i "s|@./Props/instanceable_meshes.usd@|@${abs_geom_ref}@|g" "${usda_file}"
done

echo "All USDs flattened and saved under: $USD_BASE_DIR"
