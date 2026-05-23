#!/usr/bin/env bash
# Hipify and build custom_rasterizer for ROCm from Hunyuan3D-2 sources.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

FORCE_HIPIFY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-arch)
            GPU_ARCHS="$2"
            shift 2
            ;;
        --force-hipify)
            FORCE_HIPIFY=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--gpu-arch GFX] [--force-hipify]"
            echo "  --gpu-arch GFX   Target AMDGPU arch (default: auto-detect)"
            echo "  --force-hipify   Re-run hipify even if *_hip.* files exist"
            exit 0
            ;;
        *)
            die "Unknown option: $1 (try --help)"
            ;;
    esac
done

activate_venv
require_build_tools
export_rocm_build_env

HUNYUAN_DIR="$(hunyuan3d_dir)"
RASTERIZER_DIR="${HUNYUAN_DIR}/hy3dgen/texgen/custom_rasterizer"
PATCH_FILE="${REPO_ROOT}/patches/custom_rasterizer-setup-rocm.patch"

[[ -d "${RASTERIZER_DIR}" ]] || die "custom_rasterizer not found. Run ./scripts/bootstrap-rasterizer.sh or install.sh first."

python -c "import torch; assert getattr(torch.version, 'hip', None), 'PyTorch must be ROCm/HIP build'" \
    || die "Install ROCm PyTorch first: ./scripts/bootstrap-rasterizer.sh or install.sh"

log "Building custom_rasterizer at ${RASTERIZER_DIR}"
cd "${RASTERIZER_DIR}"

KERNEL_DIR="lib/custom_rasterizer_kernel"

apply_rocm_setup_patch() {
    if grep -q 'rasterizer_hip.cpp' setup.py 2>/dev/null; then
        return 0
    fi
    if command -v patch >/dev/null 2>&1; then
        log "Applying ROCm setup.py patch..."
        patch -p1 -N < "${PATCH_FILE}" || die "Failed to apply ${PATCH_FILE}"
        return 0
    fi
    log "patch(1) not found; applying ROCm setup.py changes inline..."
    python - <<'PY'
from pathlib import Path

setup = Path("setup.py")
text = setup.read_text()
replacements = [
    ("lib/custom_rasterizer_kernel/rasterizer.cpp", "lib/custom_rasterizer_kernel/rasterizer_hip.cpp"),
    ("lib/custom_rasterizer_kernel/grid_neighbor.cpp", "lib/custom_rasterizer_kernel/grid_neighbor_hip.cpp"),
    ("lib/custom_rasterizer_kernel/rasterizer_gpu.cu", "lib/custom_rasterizer_kernel/rasterizer_gpu.hip"),
]
for old, new in replacements:
    if old not in text:
        raise SystemExit(f"setup.py missing expected source: {old}")
    text = text.replace(old, new, 1)
setup.write_text(text)
if "rasterizer_hip.cpp" not in text:
    raise SystemExit("setup.py patch did not apply")
PY
}

hipify_sources() {
    log "Hipifying CUDA sources..."
    python - <<'PY'
import os
from torch.utils.hipify import hipify_python

root = os.path.abspath(".")
sources = [
    os.path.join(root, "lib/custom_rasterizer_kernel/rasterizer.cpp"),
    os.path.join(root, "lib/custom_rasterizer_kernel/grid_neighbor.cpp"),
    os.path.join(root, "lib/custom_rasterizer_kernel/rasterizer_gpu.cu"),
]
hipify_python.hipify(
    project_directory=root,
    output_directory=root,
    extra_files=sources,
    hipify_extra_files_only=True,
    is_pytorch_extension=True,
)
PY
}

if [[ "${FORCE_HIPIFY}" -eq 1 ]]; then
    log "Removing existing hipified sources (--force-hipify)..."
    rm -f "${KERNEL_DIR}"/*_hip.* "${KERNEL_DIR}"/*.hip
fi

if [[ ! -f "${KERNEL_DIR}/rasterizer_hip.cpp" ]] \
    || [[ ! -f "${KERNEL_DIR}/grid_neighbor_hip.cpp" ]] \
    || [[ ! -f "${KERNEL_DIR}/rasterizer_gpu.hip" ]]; then
    hipify_sources
fi

for f in "${KERNEL_DIR}/rasterizer_hip.cpp" \
         "${KERNEL_DIR}/grid_neighbor_hip.cpp" \
         "${KERNEL_DIR}/rasterizer_gpu.hip"; do
    [[ -f "${f}" ]] || die "Hipify failed: missing ${f}"
done

apply_rocm_setup_patch

log "Installing custom_rasterizer (editable)..."
pip install -e . --no-build-isolation

log "GPU smoke test: custom_rasterizer_kernel.rasterize_image"
python -c "
import torch
import custom_rasterizer_kernel

assert getattr(torch.version, 'hip', None), 'PyTorch must be ROCm/HIP build'
assert torch.cuda.is_available(), 'CUDA/HIP device not available'

device = torch.device('cuda', int('${HIP_VISIBLE_DEVICES:-0}'.split(',')[0]))
props = torch.cuda.get_device_properties(device)
print(f'Smoke test on device {device.index}: {props.gcnArchName}')

pos = torch.tensor(
    [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
    device=device,
)
tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
clamp = torch.zeros(0, device=device)
custom_rasterizer_kernel.rasterize_image(
    pos[0], tri, clamp, 8, 8, 1e-6, 0,
)
print('custom_rasterizer OK')
"

log "custom_rasterizer build complete."
