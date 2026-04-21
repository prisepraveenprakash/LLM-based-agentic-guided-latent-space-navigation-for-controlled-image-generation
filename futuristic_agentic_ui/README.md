# Futuristic Agentic UI

This module is an isolated Flask-based interface for prompt-driven, agentic face editing.
It lives in `futuristic_agentic_ui/` and reuses the core editing pipeline from the main project.

## What it does
- Accepts natural-language user prompts (for example: "make the person look younger with a subtle smile").
- Uses `knowledge_base.json` to infer target attribute(s) and desired intensity.
- Runs agentic sweep cycles over candidate intensities.
- Scores each candidate and selects the best output.
- Streams LLM + agent events to a live frontend timeline.

Supported attributes:
- `Bangs`
- `Eyeglasses`
- `No_Beard`
- `Smiling`
- `Young`

## Module structure
- `app.py`: Flask app, async job lifecycle, status/event APIs.
- `agentic_engine.py`: planning, candidate scheduling, scoring, and edit execution.
- `knowledge_base.json`: attribute keywords, intensity language cues, and agent policy.
- `templates/index.html`: UI shell.
- `static/app.js`, `static/styles.css`: frontend behavior and styling.

## Prerequisites
From project root, install dependencies:

```powershell
conda env create -f environment.yml
conda activate talk_edit
```

Required assets are expected in:
- `download/pretrained_models/`
- `download/editing_data/teaser_latent_code.npz.npy`

## Run
From project root:

```powershell
python futuristic_agentic_ui/app.py
```

Open:
- `http://127.0.0.1:7861`

## API overview
- `POST /api/agentic/start`: start an agentic run.
- `GET /api/agentic/status/<job_id>`: poll run status, events, progress, and results.
- `POST /api/agentic/preview/start`: start dataset-face preview generation.
- `GET /api/agentic/preview/status/<preview_id>`: poll preview job status.
- `GET /results/<path:filename>`: serve generated result images.
- `GET /health`: health endpoint.

## Outputs
Generated images are saved under:
- `results/agentic_runs/`

This includes:
- dataset preview faces (`dataset_preview_latent_<idx>.png`)
- per-cycle sweep candidates
- selected best outputs from each run

## Notes
- This UI is isolated from the classic `web_app.py` flow.
- The module executes edits by calling `editing_wo_dialog.py` internally.
- Default latent index range is `0-99` (teaser latent dataset).
- You can tune behavior via `knowledge_base.json` (attribute mapping, intensity language, cycle policy, max active agents).
