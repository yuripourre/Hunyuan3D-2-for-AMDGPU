#!/usr/bin/env bash
# Download Hunyuan3D-2 Hugging Face weights without login (public repos only).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

DOWNLOAD_MULTIVIEW_MODELS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-multiview)
            DOWNLOAD_MULTIVIEW_MODELS=0
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--no-multiview]"
            echo "  Downloads public Tencent Hunyuan3D-2 weights (no Hugging Face login)."
            exit 0
            ;;
        *)
            die "Unknown option: $1 (try --help)"
            ;;
    esac
done

require_cmd python3
check_python_version
ensure_venv
export_hf_anonymous_env

python -c "import huggingface_hub" 2>/dev/null \
    || die "huggingface_hub not installed. Run: ./scripts/install-app.sh"

download_hunyuan_models
