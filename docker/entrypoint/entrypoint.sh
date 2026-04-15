#!/bin/bash

WS="/home/$(whoami)/ros2_ws"
DATASET_DIR="/home/$(whoami)/dataset"

VIO_FILE="${DATASET_DIR}/VIO_stripped/VIO_stripped_0.mcap"
STL_FILE="${DATASET_DIR}/meshes/omni_mesh.stl"

HF_DOWNLOADER="/home/$(whoami)/download_datasets.py"

is_built() {
    [[ -d "${WS}/install" ]]
}

build_ws() {
    echo "[entrypoint] Building ROS 2 workspace..."
    cd "${WS}"
    source /opt/ros/jazzy/setup.bash \
        && colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release \
        --parallel-workers $(nproc) && cd
}

source_ws() {
    local setup="${WS}/install/local_setup.bash"
    if [[ -f "$setup" ]]; then
        echo "[entrypoint] Sourcing workspace."
        # shellcheck disable=SC1090
        source "$setup"
    fi
}

download_missing_data() {
    echo "[entrypoint] Checking dataset completeness..."

    local need_vio=false
    local need_stl=false

    # check VIO dataset
    if [[ ! -f "${VIO_FILE}" ]]; then
        echo "[entrypoint] Missing VIO_stripped dataset."
        need_vio=true
    else
        echo "[entrypoint] Found VIO_stripped dataset."
    fi

    # check STL mesh
    if [[ ! -f "${STL_FILE}" ]]; then
        echo "[entrypoint] Missing omni_mesh.stl"
        need_stl=true
    else
        echo "[entrypoint] Found omni_mesh.stl mesh."
    fi

    echo "[entrypoint] Download decisions → VIO=$need_vio STL=$need_stl"

    # Perform downloads
    if [[ "$need_vio" = true && "$need_stl" = true ]]; then
        echo "[entrypoint] Downloading ALL assets (bag + mesh)..."
        /opt/venv/bin/python "$HF_DOWNLOADER" --download-all
    elif [[ "$need_vio" = true ]]; then
        echo "[entrypoint] Downloading VIO_stripped only..."
        /opt/venv/bin/python "$HF_DOWNLOADER" --download-vio
    elif [[ "$need_stl" = true ]]; then
        echo "[entrypoint] Downloading omni_mesh.stl only..."
        /opt/venv/bin/python "$HF_DOWNLOADER" --download-stl
    else
        echo "[entrypoint] All assets already present."
    fi
}

echo "[entrypoint] Starting entrypoint as user: $(whoami)"

# Build workspace if needed
if ! is_built; then
    build_ws
else
    echo "[entrypoint] Workspace already built. Skipping."
fi

# Source workspace
source_ws

# Ensure dataset folder completeness before running ROS code
download_missing_data

cd /home/$(whoami)/output

# ── CycloneDDS tuning ────────────────────────────────────────────────
# Activates the XML config that uses the large kernel socket buffers
# set by usb_dds_setup/install_ipfrag.sh (rmem_max = 2 GiB).
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/$(whoami)/.ros/cyclone_dds.xml

echo "[entrypoint] Executing command: $*"
exec "$@"
