# Talk-to-Edit (Extended Project)

This repository contains the full Talk-to-Edit pipeline plus two web interfaces:
- the classic web UI (`web_app.py`), and
- an isolated futuristic multi-agent UI (`futuristic_agentic_ui/`).

## What this project does
- Edits face attributes in StyleGAN latent space.
- Supports direct attribute editing (no dialog).
- Supports dialog-based editing from natural language requests.
- Supports inversion from a real image (optional).
- Includes an agentic multi-cycle sweep UI that evaluates multiple intensities and selects the best result.

Supported attributes:
- `Bangs`
- `Eyeglasses`
- `No_Beard`
- `Smiling`
- `Young`

## Project structure
- `editing_wo_dialog.py`: CLI attribute editing pipeline.
- `editing_with_dialog.py`: CLI dialog-driven editing pipeline.
- `web_app.py`: classic Flask UI (manual attribute controls + dialog mode).
- `futuristic_agentic_ui/app.py`: isolated futuristic agentic UI.
- `train.py`: training for field-function models.
- `editing_quantitative.py`, `quantitative_results.py`: evaluation scripts.
- `configs/`: train/edit configuration files.
- `download/`: pretrained models, latent data, and sample real images.
- `results/`: generated outputs and logs.

## Setup
From project root:

```powershell
conda env create -f environment.yml
conda activate talk_edit
```

If you use pip/venv instead of conda, install equivalent dependencies from `environment.yml`.

## Run: Direct attribute editing (CLI)

```powershell
python editing_wo_dialog.py --opt configs/editing/editing_wo_dialog.yml --attr Smiling --target_val 4 --latent_index 38
```

Output is saved under:
- `results/editing_wo_dialog/visualization/`

Optional real-image inversion edit:

```powershell
python editing_wo_dialog.py --opt configs/editing/editing_wo_dialog.yml --attr Eyeglasses --target_val 3 --input_image download/real_images/annehathaway.png
```

## Run: Dialog editing (CLI)

```powershell
python editing_with_dialog.py --opt configs/editing/editing_with_dialog.yml --latent_index 38 --request "make the person look younger"
```

Output is saved under:
- `results/dialog_editing/visualization/`

## Run: Classic web UI

```powershell
python web_app.py
```

Open:
- `http://127.0.0.1:7860`

Features:
- dataset face preview by latent index
- direct attribute editing
- dialog-based editing

## Run: Futuristic agentic UI

```powershell
python futuristic_agentic_ui/app.py
```

Open:
- `http://127.0.0.1:7861`

Features:
- prompt-driven editing
- knowledge-base-guided attribute + intensity inference
- multi-agent / multi-cycle sweep and candidate scoring
- live event timeline of decisions

Agentic outputs are saved under:
- `results/agentic_runs/`

## Training field models
Use one of the configs in `configs/train/`.

Example:

```powershell
python train.py --opt configs/train/field_1024_eyeglasses.yml
```

## Quantitative evaluation

```powershell
python quantitative_results.py --attribute Eyeglasses --work_dir results/eval --image_dir results/editing_wo_dialog/visualization --image_num 100
```

## Notes
- The repo already expects pretrained checkpoints inside `download/pretrained_models/`.
- Default configs target latent indices from the teaser latent set (`0-99`).
- For GPU changes, update `gpu_ids` and device settings in the YAML configs.
