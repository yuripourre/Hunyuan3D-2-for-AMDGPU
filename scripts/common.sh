#!/usr/bin/env bash
# Shared helpers for Hunyuan3D-2 ROCm install and run scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${REPO_ROOT}/config/defaults.env"
VENV_DIR="${REPO_ROOT}/.venv"

if [[ -f "${CONFIG_FILE}" ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG_FILE}"
fi

log() { echo "[hunyuan3d] $*"; }
die() { echo "[hunyuan3d] ERROR: $*" >&2; exit 1; }

require_cmd() {
    local cmd="$1"
    command -v "${cmd}" >/dev/null 2>&1 || die "Required command not found: ${cmd}"
}

check_python_version() {
    local min_major min_minor
    min_major="${PYTHON_MIN%%.*}"
    min_minor="${PYTHON_MIN#*.}"
    min_minor="${min_minor%%.*}"

    local ver major minor
    ver="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
    major="${ver%%.*}"
    minor="${ver#*.}"

    if [[ "${major}" -lt "${min_major}" ]] \
        || { [[ "${major}" -eq "${min_major}" ]] && [[ "${minor}" -lt "${min_minor}" ]]; }; then
        die "Python ${ver} found; need >= ${PYTHON_MIN}"
    fi
    log "Using Python ${ver}"
}

check_rocm() {
    if command -v rocminfo >/dev/null 2>&1; then
        log "ROCm detected (rocminfo available)"
        return 0
    fi
    if [[ -d /opt/rocm ]]; then
        log "ROCm detected (/opt/rocm)"
        return 0
    fi
    die "ROCm not found. Install ROCm and ensure rocminfo or /opt/rocm exists."
}

warn_path_spaces() {
    case "${REPO_ROOT}" in
        *" "*)
            log "WARNING: Install path contains spaces; ROCm builds may fail."
            log "  See https://github.com/ROCm/ROCm/issues/4329"
            ;;
    esac
}

activate_venv() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        die "Virtualenv not found at ${VENV_DIR}. Run: ./scripts/bootstrap-rasterizer.sh or install.sh"
    fi
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
}

ensure_venv() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        log "Creating virtualenv at ${VENV_DIR}"
        python3 -m venv "${VENV_DIR}"
    fi
    activate_venv
}

hunyuan3d_dir() {
    echo "${REPO_ROOT}/${HUNYUAN3D_DIR:-vendor/Hunyuan3D-2}"
}

# Rank gfx arch strings: prefer discrete (gfx12 > gfx11 > other gfx*).
_gfx_arch_rank() {
    local arch="$1"
    case "${arch}" in
        gfx12*) echo 300 ;;
        gfx11*) echo 200 ;;
        gfx10*) echo 100 ;;
        gfx*)   echo 50 ;;
        *)      echo 0 ;;
    esac
}

detect_gpu_arch() {
    local best="" best_rank=0 rank arch
    if command -v rocminfo >/dev/null 2>&1; then
        while IFS= read -r arch; do
            [[ "${arch}" =~ ^gfx ]] || continue
            rank="$(_gfx_arch_rank "${arch}")"
            if [[ "${rank}" -gt "${best_rank}" ]]; then
                best_rank="${rank}"
                best="${arch}"
            fi
        done < <(rocminfo 2>/dev/null | awk '/^  Name:/ {print $2}' | grep '^gfx' || true)
    fi
    if [[ -n "${best}" ]]; then
        echo "${best}"
        return 0
    fi
    die "Could not detect GPU arch from rocminfo. Set GPU_ARCHS manually (e.g. gfx1201)."
}

detect_primary_hip_device() {
    local target_arch="${1:-}"
    python - <<'PY' "${target_arch}"
import sys
target = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
try:
    import torch
except ImportError:
    sys.exit(2)
if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
    sys.exit(2)
best_idx = 0
best_score = -1
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    arch = getattr(props, "gcnArchName", "") or ""
    score = props.multi_processor_count
    if target and arch == target:
        score += 10000
    if score > best_score:
        best_score = score
        best_idx = i
print(best_idx)
PY
}

require_build_tools() {
    require_cmd hipcc
    require_cmd g++
    require_cmd ninja
}

export_torch_lib_path() {
    local torch_lib
    torch_lib="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)" || return 0
    if [[ -d "${torch_lib}" ]]; then
        export LD_LIBRARY_PATH="${torch_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
}

export_hf_anonymous_env() {
    # Public Tencent models: do not use cached HF login tokens (avoids spurious auth prompts).
    export HF_HUB_DISABLE_IMPLICIT_TOKEN="${HF_HUB_DISABLE_IMPLICIT_TOKEN:-1}"
}

export_rocm_build_env() {
    export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}"
    export_hf_anonymous_env
    export_torch_lib_path

    local arch="${GPU_ARCHS:-}"
    if [[ -z "${arch}" || "${arch}" == "auto" ]]; then
        arch="$(detect_gpu_arch)"
        GPU_ARCHS="${arch}"
    fi
    export GPU_ARCHS

    if [[ -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
        local hip_dev
        if hip_dev="$(detect_primary_hip_device "${GPU_ARCHS}" 2>/dev/null)"; then
            export HIP_VISIBLE_DEVICES="${hip_dev}"
        else
            export HIP_VISIBLE_DEVICES=0
        fi
    fi

    log "ROCm build env: GPU_ARCHS=${GPU_ARCHS} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
}

has_rocm_pytorch() {
    python -c "import torch; exit(0 if getattr(torch.version, 'hip', None) else 1)" 2>/dev/null
}

install_rocm_pytorch() {
    if has_rocm_pytorch; then
        log "ROCm PyTorch already installed in venv"
        return 0
    fi
    local torch_ver tv_ver ta_ver index
    torch_ver="${TORCH_VERSION:-2.7.1}"
    tv_ver="${TORCHVISION_VERSION:-0.22.1}"
    ta_ver="${TORCHAUDIO_VERSION:-2.7.1}"
    index="${PYTORCH_ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.3}"
    log "Installing PyTorch ${torch_ver} (ROCm)..."
    pip install --upgrade pip wheel setuptools
    pip install "torch==${torch_ver}" "torchvision==${tv_ver}" "torchaudio==${ta_ver}" \
        --index-url "${index}"
}

install_python_deps() {
    log "Installing Python dependencies..."
    pip install -r "${REPO_ROOT}/requirements.txt"
}

clone_hunyuan3d_if_needed() {
    local hunyuan_dir repo_url
    hunyuan_dir="$(hunyuan3d_dir)"
    repo_url="${HUNYUAN3D_REPO:-https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git}"
    if [[ ! -d "${hunyuan_dir}/.git" ]]; then
        log "Cloning Hunyuan3D-2..."
        mkdir -p "$(dirname "${hunyuan_dir}")"
        git clone "${repo_url}" "${hunyuan_dir}"
    else
        log "Using existing Hunyuan3D-2 at ${hunyuan_dir}"
    fi
}

install_hy3dgen() {
    clone_hunyuan3d_if_needed
    log "Installing hy3dgen (editable)..."
    pip install -e "$(hunyuan3d_dir)"
}

install_differentiable_renderer() {
    local diff_dir
    diff_dir="$(hunyuan3d_dir)/hy3dgen/texgen/differentiable_renderer"
    [[ -d "${diff_dir}" ]] || die "differentiable_renderer not found at ${diff_dir}"
    log "Building differentiable_renderer..."
    pip install -e "${diff_dir}" --no-build-isolation
}

install_gradio_app() {
    local hunyuan_dir gradio_src gradio_dst
    hunyuan_dir="$(hunyuan3d_dir)"
    gradio_src="${REPO_ROOT}/gradio_app.py"
    gradio_dst="${hunyuan_dir}/gradio_app.py"
    if [[ -f "${gradio_src}" ]]; then
        log "Installing gradio_app.py into Hunyuan3D-2..."
        cp -f "${gradio_src}" "${gradio_dst}"
    else
        die "gradio_app.py not found at ${gradio_src}"
    fi
}

write_gradio_port() {
    mkdir -p "${REPO_ROOT}/config"
    echo "${GRADIO_PORT:-8080}" > "${REPO_ROOT}/config/port"
}

download_hunyuan_models() {
    local with_multiview="${DOWNLOAD_MULTIVIEW_MODELS:-1}"
    export_hf_anonymous_env
    log "Downloading Hugging Face models (no login required for public Tencent repos)..."
    DOWNLOAD_MULTIVIEW_MODELS="${with_multiview}" python - <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

# Anonymous access to public model repos (no HF token).
TOKEN = False
MULTIVIEW = os.environ.get("DOWNLOAD_MULTIVIEW_MODELS", "1") == "1"

HUNYUAN3D_2 = "tencent/Hunyuan3D-2"
SUBFOLDERS_2 = [
    "hunyuan3d-dit-v2-0-turbo",
    "hunyuan3d-vae-v2-0-turbo",
    "hunyuan3d-delight-v2-0",
    "hunyuan3d-paint-v2-0-turbo",
]

def download_subfolders(repo_id: str, subfolders: list[str]) -> None:
    for sub in subfolders:
        print(f"Downloading {repo_id} ({sub}) ...", flush=True)
        # Use ** so nested paths (unet/, vae/, etc.) are included; "sub/*" is one level only.
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{sub}/**"],
            token=TOKEN,
        )

download_subfolders(HUNYUAN3D_2, SUBFOLDERS_2)

if MULTIVIEW:
    print("Downloading tencent/Hunyuan3D-2mv (hunyuan3d-dit-v2-mv-turbo) ...", flush=True)
    snapshot_download(
        repo_id="tencent/Hunyuan3D-2mv",
        allow_patterns=["hunyuan3d-dit-v2-mv-turbo/**"],
        token=TOKEN,
    )

print("Downloading Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled (text-to-3D) ...", flush=True)
snapshot_download(
    repo_id="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
    token=TOKEN,
)

print("Model download complete.", flush=True)
PY
}

ensure_custom_rasterizer() {
    if python -c "import custom_rasterizer_kernel; assert hasattr(custom_rasterizer_kernel, 'rasterize_image')" 2>/dev/null; then
        log "custom_rasterizer already installed"
        return 0
    fi
    log "custom_rasterizer not found; building..."
    "${SCRIPT_DIR}/build-custom-rasterizer.sh"
}

require_gradio_ready() {
    local hunyuan_dir
    hunyuan_dir="$(hunyuan3d_dir)"
    [[ -d "${VENV_DIR}" ]] || die "Virtualenv missing. Run: ./scripts/install-app.sh or ./scripts/install.sh"
    python -c "import hy3dgen" 2>/dev/null \
        || die "hy3dgen not installed. Run: ./scripts/install-app.sh or ./scripts/install.sh"
    python -c "import gradio" 2>/dev/null \
        || die "gradio not installed. Run: ./scripts/install-app.sh or ./scripts/install.sh"
    pip show custom_rasterizer >/dev/null 2>&1 \
        || die "custom_rasterizer not built. Run: ./scripts/build-custom-rasterizer.sh"
    local kernel_hip
    kernel_hip="$(hunyuan3d_dir)/hy3dgen/texgen/custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer_hip.cpp"
    [[ -f "${kernel_hip}" ]] \
        || die "custom_rasterizer HIP sources missing. Run: ./scripts/build-custom-rasterizer.sh"
    [[ -f "${hunyuan_dir}/gradio_app.py" ]] \
        || die "gradio_app.py missing in ${hunyuan_dir}. Run: ./scripts/install-app.sh or ./scripts/install.sh"
    [[ -d "${hunyuan_dir}/assets" ]] \
        || die "assets/ missing in ${hunyuan_dir}. Re-clone or run install-app.sh"
    log "Gradio environment OK"
}
