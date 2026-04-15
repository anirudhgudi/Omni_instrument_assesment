#!/bin/bash
set -e

VENV_PIP="${VENV_PIP:-/opt/venv/bin/pip}"

echo "==========================================="
echo "   Setting up HITNET ONNX Depth Model      "
echo "==========================================="

echo "[1/4] Installing dependencies..."
if command -v "${VENV_PIP}" >/dev/null 2>&1; then
    sudo "${VENV_PIP}" install --quiet --no-warn-script-location \
        "numpy<2" "opencv-python-headless<4.13" onnxruntime-gpu
else
    pip install --quiet --no-warn-script-location \
        "numpy<2" "opencv-python-headless<4.13" onnxruntime-gpu
fi

# Go to neural_depth models directory
cd "$(dirname "$0")"
mkdir -p models/hitnet
cd models/hitnet

echo "[2/4] Downloading HITNET weights from PINTO0309 Model Zoo..."
if [ ! -f "model_float32.onnx" ]; then
    # Download the main resources tarball from PINTO
    wget -q --show-progress "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/142_HITNET/resources.tar.gz" -O resources.tar.gz

    tar -zxvf resources.tar.gz "eth3d/saved_model_720x1280/model_float32.onnx"

    # Move and cleanup
    mv eth3d/saved_model_720x1280/model_float32.onnx ./
    rm -rf eth3d/
    rm resources.tar.gz
    echo "[4/4] Model successfully downloaded and extracted."
else
    echo "ONNX model already exists! Skipping download."
fi

echo ""
echo "HITNET ONNX setup complete!"
echo "Model path: $(pwd)/model_float32.onnx"
