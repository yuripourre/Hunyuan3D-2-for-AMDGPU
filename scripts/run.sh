#!/usr/bin/env bash
# Launch Hunyuan3D-2 Gradio (single-view + text-to-3D).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ENABLE_FLASHVDM=1
GRADIO_EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-flashvdm)
            ENABLE_FLASHVDM=0
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--no-flashvdm] [gradio_app.py options...]"
            echo "  --no-flashvdm   Omit --enable_flashvdm (if flash-attention was not installed)"
            echo "  Other args are passed to gradio_app.py (e.g. --disable_tex)"
            exit 0
            ;;
        *)
            GRADIO_EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

activate_venv
export_rocm_build_env
require_gradio_ready

PORT="${GRADIO_PORT:-8080}"
if [[ -f "${REPO_ROOT}/config/port" ]]; then
    PORT="$(tr -d '[:space:]' < "${REPO_ROOT}/config/port")"
fi

HUNYUAN_DIR="$(hunyuan3d_dir)"
cd "${HUNYUAN_DIR}"

GRADIO_CMD=(
    python gradio_app.py
    --model_path tencent/Hunyuan3D-2
    --subfolder hunyuan3d-dit-v2-0-turbo
    --texgen_model_path tencent/Hunyuan3D-2
    --low_vram_mode
    --enable_t23d
    --port "${PORT}"
)
if [[ "${ENABLE_FLASHVDM}" -eq 1 ]]; then
    GRADIO_CMD+=(--enable_flashvdm)
fi
GRADIO_CMD+=("${GRADIO_EXTRA_ARGS[@]}")

log "Starting Gradio at http://127.0.0.1:${PORT}"
exec "${GRADIO_CMD[@]}"
