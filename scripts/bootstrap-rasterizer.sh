#!/usr/bin/env bash
# Minimal setup to build custom_rasterizer only (no flash-attention or Gradio).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-arch)
            GPU_ARCHS="$2"
            shift 2
            ;;
        --force-hipify)
            FORCE_HIPIFY_ARG="--force-hipify"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--gpu-arch GFX] [--force-hipify]"
            echo "  Creates venv, installs ROCm PyTorch, clones Hunyuan3D-2, builds custom_rasterizer."
            exit 0
            ;;
        *)
            die "Unknown option: $1 (try --help)"
            ;;
    esac
done

require_cmd python3
require_cmd git
check_python_version
check_rocm
warn_path_spaces

export_rocm_build_env
ensure_venv

TORCH_VER="${TORCH_VERSION:-2.7.1}"
TV_VER="${TORCHVISION_VERSION:-0.22.1}"
TA_VER="${TORCHAUDIO_VERSION:-2.7.1}"
INDEX="${PYTORCH_ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.3}"

log "Installing PyTorch ${TORCH_VER} (ROCm)..."
pip install --upgrade pip wheel setuptools
pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}" "torchaudio==${TA_VER}" \
    --index-url "${INDEX}"

log "Installing rasterizer build dependencies..."
pip install ninja==1.11.1.4 pybind11==2.13.6

HUNYUAN_DIR="$(hunyuan3d_dir)"
REPO_URL="${HUNYUAN3D_REPO:-https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git}"

if [[ ! -d "${HUNYUAN_DIR}/.git" ]]; then
    log "Cloning Hunyuan3D-2..."
    mkdir -p "$(dirname "${HUNYUAN_DIR}")"
    git clone "${REPO_URL}" "${HUNYUAN_DIR}"
else
    log "Using existing Hunyuan3D-2 at ${HUNYUAN_DIR}"
fi

"${SCRIPT_DIR}/build-custom-rasterizer.sh" ${FORCE_HIPIFY_ARG:-}

log "Rasterizer bootstrap complete."
