# Manga Upscaler — Agent Instructions

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.14. Required deps: `torch`, `opencv_python`, `numpy`, `typer`, `rich`, `safetensors`, `einops`, `requests`.

## Architecture

```
manga_upscale.py        # CLI entry (argparse) — orchestrates extract/upscale/download
backend/upscale.py      # Core upscaler engine (typer) — loads models, runs inference
backend/utils/          # Model architectures (RRDB/ESRGAN, SRVGG, SPSR, FDAT, DAT)
backend/models/         # Model files (.pth or .safetensors) — required at runtime
manga_pipeline.py       # User-specific automation (kobodl → extract → upscale → Komga)
```

- `manga_upscale.py` delegates actual upscaling to `backend/upscale.py` via `subprocess.run`.
- `manga_pipeline.py` has hardcoded paths (`LOCAL_WORK_DIR`, `NAS_KOMGA_DIR`, `KOBODL_PATH`) — treat as reference, not portable code.
- Model scale inferred from filename prefix: `4x_*` → 4, `2x_*` → 2, fallback → 2.

## CLI Commands

```bash
# Extract .zip/.cbz archives (deletes archive after extraction)
python manga_upscale.py extract --input /path/to/archives [--overwrite]

# Upscale images
python manga_upscale.py upscale --color /path/to/images --output /path/out [--model-color NAME]
python manga_upscale.py upscale --bw /path/to/images --output /path/out [--model-bw NAME]

# Download models
python manga_upscale.py download best|bw|color
```

Models resolve by alias prefix match against `backend/models/` or as a direct file path.

## Backend Engine

```bash
python backend/upscale.py <model_path> -i input/ -o output/ [-se] [-c] [-fp16] [-s tile|mirror|replicate|alpha_pad]
```

- `-se` — skip existing output files
- `-c` — force CPU (very slow)
- `-fp16` — float16 inference (default True)
- `-s` — seamless tile mode (tile|mirror|replicate|alpha_pad)
- Model chaining: `model1+model2` or `model1&model2@50` (interpolation)
- Supports both `.pth` and `.safetensors` model files

## Known Constraints

- **DAT2 models unsupported** — backend exits with error. Only DAT (non-2) works.
- **No GUI in this repo** — `pyqt_app.py` referenced in README does not exist here.
- **No lint/format/typecheck/CI/Makefile/task runner** — direct script execution only.
- **Tests require real model files** in `backend/models/` — won't pass with dummy data.
- **Images auto-padded** to multiples of scale factor before upscaling.
- **Auto-split upscaling** recursively splits images to avoid CUDA OOM.

## Running Tests

```bash
cd backend && python -m pytest tests/
cd backend && python -m unittest tests/test_fdat_architecture.py
```

Tests must be run from `backend/` directory. They require real `.pth`/`.safetensors` files in `backend/models/`.
