#!/bin/bash
# ============================================================
#  Setup script for RAFT-Stereo inside Docker container
#  Run this once inside the container before launching the pipeline.
# ============================================================
set -eo pipefail

WS="${HOME}/ros2_ws/src"
RAFT_DIR="${WS}/RAFT-Stereo"
MODELS_DIR="${RAFT_DIR}/models"
VENV_PIP="/opt/venv/bin/pip"
VENV_PYTHON="/opt/venv/bin/python"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

echo "============================================"
echo " RAFT-Stereo Setup"
echo "============================================"

# ── 1. Clone RAFT-Stereo if not present ─────────────────────
if [ ! -d "${RAFT_DIR}" ]; then
    echo "[1/4] Cloning RAFT-Stereo..."
    cd "${WS}"
    git clone https://github.com/princeton-vl/RAFT-Stereo.git
else
    echo "[1/4] RAFT-Stereo already cloned."
fi

# ── 2. Download pretrained models if not present ────────────
if [ ! -f "${MODELS_DIR}/raftstereo-middlebury.pth" ]; then
    echo "[2/4] Downloading pretrained models..."
    mkdir -p "${MODELS_DIR}"
    cd "${MODELS_DIR}"
    wget -q --show-progress "https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip"
    unzip -o models.zip
    rm -f models.zip
    echo "  Available models:"
    ls -la "${MODELS_DIR}"/*.pth 2>/dev/null || echo "  (none found)"
else
    echo "[2/4] Pretrained models already present."
fi

# ── 3. Install Python dependencies ─────────────────────────
echo "[3/4] Installing Python dependencies..."

# Check the interpreter used by ROS runtime dependency injection, not system Python.
if "${VENV_PYTHON}" -c "import torch; assert torch.cuda.is_available(); print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda} already installed')" 2>/dev/null; then
    echo "  Skipping torch/torchvision (CUDA-enabled PyTorch already present)."
else
    echo "  Installing CUDA-enabled PyTorch from ${TORCH_INDEX_URL}."
    sudo ${VENV_PIP} install --quiet --no-warn-script-location --force-reinstall \
        --index-url "${TORCH_INDEX_URL}" \
        torch torchvision
fi

# /opt/venv is root-owned from Docker build → use sudo
if command -v ${VENV_PIP} &>/dev/null; then
    sudo ${VENV_PIP} install --quiet --no-warn-script-location \
        "numpy<2" scipy opt-einsum imageio scikit-image
else
    pip3 install --user --quiet --no-warn-script-location \
        "numpy<2" scipy opt-einsum imageio scikit-image
fi

# ── 4. Build the neural_depth ROS 2 package ────────────────
echo "[4/4] Building neural_depth package..."
cd "${HOME}/ros2_ws"
source /opt/ros/jazzy/setup.bash
colcon build --packages-select neural_depth --symlink-install
source install/local_setup.bash

echo ""
echo "============================================"
echo " Setup complete!"
echo " Run the pipeline with:"
echo "   ros2 launch tsdf_saver saver.launch.py"
echo "============================================"
