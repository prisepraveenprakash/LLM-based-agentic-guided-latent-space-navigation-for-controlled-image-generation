# Futuristic Agentic UI

This module is fully isolated from the existing web UI and lives in `futuristic_agentic_ui/`.

## What it does
- Accepts user text instructions.
- Uses a built-in knowledge base (`knowledge_base.json`) to infer the best attribute and desired intensity.
- Runs multi-cycle agentic sweeps across intensities.
- Scores all sweeped images and selects the best candidate.
- Streams every LLM + Agent decision/update to the frontend timeline.

## Run
From project root:

```powershell
python futuristic_agentic_ui/app.py
```

Open:
- `http://127.0.0.1:7861`

## Notes
- It reuses your existing `editing_wo_dialog.py` pipeline.
- Output images for each sweep candidate are saved in:
  - `results/agentic_runs/`
