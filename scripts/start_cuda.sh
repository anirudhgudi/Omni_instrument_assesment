#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

ORG="omniinstrument"
TAG="latest"

# Override this if needed, e.g.:
#   CUDA_BASE_IMAGE=nvcr.io/nvidia/tensorrt:25.11-py3 bash scripts/start_cuda.sh
CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-nvcr.io/nvidia/tensorrt:25.11-py3}"

BASE_IMAGE="${ORG}/gpu-base:${TAG}"
ROS_IMAGE="${ORG}/ros:${TAG}"
CYCLONE_IMAGE="${ORG}/cyclone:${TAG}"
PYTHON_IMAGE="${ORG}/python:${TAG}"

USERNAME="$(whoami)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
DOCKER_DIR="./docker"

# Default container to run interactively
RUN_CONTAINER="omnicomputervision"
RUN_IMAGE="${PYTHON_IMAGE}"

# ------------------------------------------------------------------------------
# DOCKERFILE PATHS
# ------------------------------------------------------------------------------
declare -A DOCKER_BUILDS=(
  ["${BASE_IMAGE}"]="${DOCKER_DIR}/Dockerfile.base"
  ["${ROS_IMAGE}"]="${DOCKER_DIR}/Dockerfile.ros"
  ["${CYCLONE_IMAGE}"]="${DOCKER_DIR}/Dockerfile.cyclone"
  ["${PYTHON_IMAGE}"]="${DOCKER_DIR}/Dockerfile.python"
)

# ------------------------------------------------------------------------------
# PARENT IMAGE MAPPING
# ------------------------------------------------------------------------------

# Build dependencies must form a DAG:
# base   → TensorRT
# ros    → base
# cyclone   → ros
# python  → cyclone

declare -A PARENTS=(
  ["${BASE_IMAGE}"]="${CUDA_BASE_IMAGE}"
  ["${ROS_IMAGE}"]="${BASE_IMAGE}"
  ["${CYCLONE_IMAGE}"]="${ROS_IMAGE}"
  ["${PYTHON_IMAGE}"]="${CYCLONE_IMAGE}"
)

# Build order: BASE → ROS → CYCLONE → PYTHON
BUILD_SEQUENCE=("${BASE_IMAGE}" "${ROS_IMAGE}" "${CYCLONE_IMAGE}" "${PYTHON_IMAGE}")

# ------------------------------------------------------------------------------
# Helper: build one image
# ------------------------------------------------------------------------------
build_image() {
  local image_tag="$1"
  local dockerfile="$2"
  local base_from="${PARENTS[$image_tag]}"

  echo "Building '${image_tag}' using parent '${base_from}'..."

  DOCKER_BUILDKIT=1 docker build \
    --build-arg BASE_FROM="${base_from}" \
    --build-arg USERNAME="${USERNAME}" \
    --build-arg USER_UID="${HOST_UID}" \
    --build-arg USER_GID="${HOST_GID}" \
    -t "${image_tag}" \
    -f "${dockerfile}" .
}

# ------------------------------------------------------------------------------
# Helper: verify Docker GPU support before long image builds
# ------------------------------------------------------------------------------
preflight_gpu() {
  local test_image="${GPU_TEST_IMAGE:-nvidia/cuda:12.2.2-base-ubuntu22.04}"

  echo "Checking Docker GPU access with '${test_image}'..."
  if ! docker run --rm --gpus all "${test_image}" nvidia-smi >/dev/null 2>&1; then
    cat <<'EOF'
ERROR: Docker cannot access the NVIDIA GPU.

Install/configure NVIDIA Container Toolkit on the host, then retry:
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker

Validation command:
  docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
EOF
    exit 1
  fi
}

# ------------------------------------------------------------------------------
# If container already running → attach
# ------------------------------------------------------------------------------
if docker ps --format '{{.Names}}' | grep -q "^${RUN_CONTAINER}$"; then
  echo "Container '${RUN_CONTAINER}' already running. Attaching..."
  exec docker exec -it "${RUN_CONTAINER}" bash
fi

preflight_gpu

# ------------------------------------------------------------------------------
# Build all images in order
# ------------------------------------------------------------------------------
for image in "${BUILD_SEQUENCE[@]}"; do
  build_image "${image}" "${DOCKER_BUILDS[$image]}"
done

# ------------------------------------------------------------------------------
# Enable GUI (X11)
# ------------------------------------------------------------------------------
echo "Enabling X11 permissions..."
xhost +local:docker >/dev/null 2>&1
xhost +SI:localuser:"${USERNAME}" >/dev/null 2>&1

# ------------------------------------------------------------------------------
# Launch the primary container (GAUSS)
# ------------------------------------------------------------------------------
echo "Launching '${RUN_CONTAINER}' using image '${RUN_IMAGE}'..."

# --runtime=nvidia is not required on modern Docker; --gpus all is sufficient.
docker run -it --rm \
  --gpus all \
  --privileged \
  --net=host \
  --pid=host \
  --ipc=host \
  --name "${RUN_CONTAINER}" \
  -e DISPLAY="${DISPLAY}" \
  --cap-add=SYS_PTRACE \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --security-opt seccomp=unconfined \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -v /tmp/runtime-root:/tmp/runtime-root \
  -v "$HOME/.Xauthority":/root/.Xauthority \
  -v $(pwd)/ros2_ws:/home/${USERNAME}/ros2_ws \
  --user "${HOST_UID}:${HOST_GID}" \
  --workdir /home/${USERNAME}/output \
  --entrypoint /usr/local/bin/scripts/entrypoint.sh \
  -v $(pwd)/dataset:/home/${USERNAME}/dataset \
  -v $(pwd)/src/compute_metrics.py:/home/${USERNAME}/compute_metrics.py \
  -v $(pwd)/output:/home/${USERNAME}/output \
  -v $(pwd)/src/download_datasets.py:/home/${USERNAME}/download_datasets.py \
  -v "$HOME/.ros":/home/${USERNAME}/.ros \
  "${RUN_IMAGE}" /bin/bash
