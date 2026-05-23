#!/usr/bin/env bash
# Complete Hunyuan3D-2 app install (deps, hy3dgen, extensions) after bootstrap-rasterizer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

SKIP_FLASH_ATTENTION=0
SKIP_MODEL_DOWNLOAD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            GRADIO_PORT="$2"
            shift 2
            ;;
        --skip-flash-attention)
            SKIP_FLASH_ATTENTION=1
            shift
            ;;
        --skip-model-download)
            SKIP_MODEL_DOWNLOAD=1
            shift
            ;;
        --no-multiview)
            DOWNLOAD_MULTIVIEW_MODELS=0
            shift
            ;;
        --gpu-arch)
            GPU_ARCHS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--port PORT] [--gpu-arch GFX] [--skip-flash-attention] [--skip-model-download] [--no-multiview]"
            echo "  Completes Hunyuan install for Gradio (use after bootstrap-rasterizer.sh)."
            echo "  Downloads public HF models without login unless --skip-model-download."
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

install_rocm_pytorch
install_python_deps
if [[ "${SKIP_MODEL_DOWNLOAD}" -eq 0 ]]; then
    download_hunyuan_models
else
    log "Skipping Hugging Face model download (--skip-model-download)"
fi
install_hy3dgen
ensure_custom_rasterizer
install_differentiable_renderer

if [[ "${SKIP_FLASH_ATTENTION}" -eq 0 ]]; then
    "${SCRIPT_DIR}/build-flash-attention.sh"
else
    log "Skipping flash-attention (--skip-flash-attention)"
fi

install_gradio_app
write_gradio_port

log ""
log "App installation complete."
log "  Run single-view:  ./scripts/run.sh"
log "  Run multiview:    ./scripts/run-multiview.sh"
log "  Port (default):   ${GRADIO_PORT:-8080} (override: GRADIO_PORT=9000 ./scripts/run.sh)"
