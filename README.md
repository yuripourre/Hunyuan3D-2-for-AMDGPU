[Español-Spanish](README-ES.md)

# Hunyuan3D-2 for AMDGPU on Linux

Scripts to install and run [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) on Linux with an AMD GPU and ROCm. Texture generation requires building `custom_rasterizer` for HIP; this repo builds it from source during install (no prebuilt wheels).

Tested with ROCm 6.3–6.4 on RX 7900 XTX (`gfx1100`) and RDNA4-class GPUs (`gfx1201`). `GPU_ARCHS` defaults to `auto` (detected from `rocminfo`).

## Prerequisites

1. **ROCm** — [Install ROCm on Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). Verify with `rocminfo` or `/opt/rocm`.

2. **System packages** (Fedora example; adjust for your distro):
   ```bash
   sudo dnf install git python3 python3-pip patch gcc-c++ ninja-build rocm-dev
   ```

3. **Python** — 3.10 or newer (`python3 --version`).

4. **Path** — Do not install or run from a directory path that contains spaces ([ROCm #4329](https://github.com/ROCm/ROCm/issues/4329)).

5. **Versions** — PyTorch 2.8+ is not supported here. Defaults use torch 2.7.1 from the ROCm 6.3 wheel index (see `config/defaults.env`).

## Quick install

```bash
git clone https://github.com/dgarcia1985/Hunyuan3d-2-for-AMDGPU-linux.git
cd Hunyuan3d-2-for-AMDGPU-linux
./scripts/install.sh
```

Options:

```bash
./scripts/install.sh --port 8080
./scripts/install.sh --gpu-arch gfx1030    # e.g. RX 6000 series
./scripts/install.sh --skip-flash-attention  # skip long flash-attention build
```

Override defaults via environment variables:

```bash
GPU_ARCHS=gfx1030 GRADIO_PORT=9000 ./scripts/install.sh
```

## Run Gradio

### Full install (fresh)

```bash
./scripts/install.sh
./scripts/run.sh              # http://127.0.0.1:8080
./scripts/run-multiview.sh    # multiview mode
```

### After rasterizer bootstrap

If you already ran `./scripts/bootstrap-rasterizer.sh`, complete the app stack then launch:

```bash
./scripts/install-app.sh
# faster (skips long flash-attention build; use --no-flashvdm when running):
# ./scripts/install-app.sh --skip-flash-attention

./scripts/run.sh
```

`install.sh` and `install-app.sh` **prefetch public Hugging Face weights without login** (same as the original installer: models are public Tencent repos). No `huggingface-cli login` is required.

To download or refresh models only:

```bash
./scripts/download-models.sh
./scripts/download-models.sh --no-multiview   # skip multiview checkpoint (~smaller download)
```

If Gradio warns about missing `diffusion_pytorch_model.safetensors` under `hunyuan3d-paint-v2-0-turbo/vae`, re-run `download-models.sh` (older installs used a shallow pattern and skipped nested weight files). The paint VAE may use `.bin` weights; diffusers will load those automatically.

Skip prefetch during install (models download on first Gradio run instead):

```bash
./scripts/install-app.sh --skip-model-download
```

Options:

```bash
GRADIO_PORT=9000 ./scripts/run.sh
./scripts/run.sh --no-flashvdm          # if install-app skipped flash-attention
./scripts/run.sh --disable_tex        # shape only, no texture pipeline
```

## Rasterizer-only build (fast)

To build only `custom_rasterizer` (venv + ROCm PyTorch + clone upstream, no flash-attention or Gradio):

```bash
./scripts/bootstrap-rasterizer.sh
```

Options: `--gpu-arch gfx1201`, `--force-hipify` (re-hipify after upstream updates).

## Repository layout

```
config/defaults.env          # pinned versions, GPU arch, ports
scripts/
  install.sh                 # full installer (venv + everything)
  install-app.sh             # complete app after bootstrap-rasterizer
  download-models.sh         # prefetch HF weights (no login)
  bootstrap-rasterizer.sh    # minimal venv + rasterizer only
  build-custom-rasterizer.sh # hipify + build texture rasterizer
  build-flash-attention.sh   # ROCm flash-attention
  run.sh / run-multiview.sh  # launch Gradio (with preflight checks)
patches/                     # setup.py patch for HIP sources
gradio_app.py                # copied into vendor/Hunyuan3D-2 on install
vendor/Hunyuan3D-2/          # cloned upstream (gitignored)
.venv/                       # Python virtualenv (gitignored)
```

## Building custom_rasterizer manually

If install fails at the rasterizer step, ensure the venv has ROCm PyTorch, then:

```bash
./scripts/build-custom-rasterizer.sh
```

Override arch or force a clean hipify:

```bash
./scripts/build-custom-rasterizer.sh --gpu-arch gfx1201 --force-hipify
```

Build scripts auto-detect `GPU_ARCHS` and set `HIP_VISIBLE_DEVICES` to the primary dGPU (useful when an iGPU is also present).

What the script does:

1. Runs PyTorch hipify on `rasterizer.cpp`, `grid_neighbor.cpp`, `rasterizer_gpu.cu` under `vendor/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer/`
2. Applies `patches/custom_rasterizer-setup-rocm.patch` so `setup.py` compiles `*_hip.*` sources
3. `pip install -e . --no-build-isolation` and runs a minimal GPU `rasterize_image()` smoke test

Some users report the extension build succeeds on Arch but fails on Ubuntu with the same Python/ROCm; if that happens, try building on Arch (or a container) and reusing the same venv layout, or adjust compiler/ROCm dev packages on your distro.

## Known issues

- **Spaces in path** — Model generation can fail if the project path contains spaces (ROCm limitation).
- **Torch 2.8** — Not working with this setup; use 2.7.x as in `config/defaults.env`.
- **Dual GPU (iGPU + dGPU)** — Build scripts set `HIP_VISIBLE_DEVICES` to the detected dGPU. Texture generation at runtime may still need the iGPU disabled in BIOS on some systems.
- **VRAM** — Texture pipeline is heavy; `--low_vram_mode` is enabled in the run scripts by default.

## Configuration

Edit [`config/defaults.env`](config/defaults.env) or export variables before `install.sh`, `install-app.sh`, or `run.sh`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GPU_ARCHS` | `auto` | ROCm offload arch (`auto` uses `rocminfo`; e.g. `gfx1201`, `gfx1100`) |
| `TORCH_VERSION` | `2.7.1` | PyTorch version |
| `PYTORCH_ROCM_INDEX` | rocm6.3 index URL | pip index for ROCm wheels |
| `GRADIO_PORT` | `8080` | Web UI port |
