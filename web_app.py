import glob
import os
import subprocess
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results" / "editing_wo_dialog" / "visualization"
DEFAULT_CONFIG = ROOT_DIR / "configs" / "editing" / "editing_wo_dialog.yml"
RESULTS_DIALOG_DIR = ROOT_DIR / "results" / "dialog_editing" / "visualization"
DIALOG_CONFIG = ROOT_DIR / "configs" / "editing" / "editing_with_dialog.yml"
PREVIEW_DIR = ROOT_DIR / "results" / "editing_wo_dialog" / "dataset_preview"

ATTRIBUTES = [
    {"label": "Bangs", "value": "Bangs"},
    {"label": "Eyeglasses", "value": "Eyeglasses"},
    {"label": "Beard", "value": "No_Beard"},
    {"label": "Smiling", "value": "Smiling"},
    {"label": "Age (Young)", "value": "Young"},
]
INTENSITIES = list(range(0, 6))
FACE_INDICES = list(range(0, 100))

app = Flask(__name__, template_folder="webui/templates", static_folder="webui/static")


def _latest_image(results_dir, exclude_start=False):
    image_candidates = glob.glob(str(results_dir / "*.png"))
    if not image_candidates:
        return None
    if exclude_start:
        image_candidates = [
            p for p in image_candidates if Path(p).name != "start_image.png"
        ]
    if not image_candidates:
        return None
    image_candidates.sort(key=os.path.getmtime, reverse=True)
    latest = Path(image_candidates[0])
    return f"/results/{results_dir.relative_to(ROOT_DIR / 'results').as_posix()}/{latest.name}"


def _start_image(results_dir):
    start_image_path = results_dir / "start_image.png"
    if not start_image_path.exists():
        return None
    return f"/results/{results_dir.relative_to(ROOT_DIR / 'results').as_posix()}/start_image.png"


@app.get("/")
def index():
    return render_template(
        "index.html",
        attributes=ATTRIBUTES,
        intensities=INTENSITIES,
        face_indices=FACE_INDICES,
        config_path=str(DEFAULT_CONFIG).replace("\\", "/"),
        dialog_config_path=str(DIALOG_CONFIG).replace("\\", "/"),
    )


@app.post("/api/edit")
def run_edit():
    payload = request.get_json(silent=True) if request.is_json else request.form
    payload = payload or {}
    attribute = payload.get("attribute", "Bangs")
    target_val = str(payload.get("target_val", 3))
    latent_index = str(payload.get("latent_index", "38"))

    allowed_attributes = {item["value"] for item in ATTRIBUTES}
    allowed_intensities = {str(v) for v in INTENSITIES}
    allowed_face_indices = {str(v) for v in FACE_INDICES}

    if attribute not in allowed_attributes:
        return jsonify({"ok": False, "error": "Invalid attribute value."}), 400
    if target_val not in allowed_intensities:
        return jsonify({"ok": False, "error": "Invalid intensity value."}), 400
    if latent_index not in allowed_face_indices:
        return jsonify({"ok": False, "error": "Invalid dataset face index."}), 400

    cmd = [
        sys.executable,
        "editing_wo_dialog.py",
        "--opt",
        str(DEFAULT_CONFIG),
        "--attr",
        attribute,
        "--target_val",
        target_val,
        "--latent_index",
        latent_index,
    ]

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Edit timed out after 10 minutes."}), 504

    if completed.returncode != 0:
        error_output = (completed.stderr or completed.stdout or "").strip()
        return jsonify(
            {
                "ok": False,
                "error": "Editing command failed.",
                "details": error_output[:2500],
            }
        ), 500

    latest_image = _latest_image(RESULTS_DIR, exclude_start=True)
    if latest_image is None:
        return jsonify({"ok": True, "message": "Edit completed, no image found yet."})

    return jsonify(
        {
            "ok": True,
            "image_url": latest_image,
            "input_face_url": _start_image(RESULTS_DIR),
        }
    )


@app.post("/api/edit-dialog")
def run_dialog_edit():
    payload = request.get_json(silent=True) if request.is_json else request.form
    payload = payload or {}
    dialog_text = str(payload.get("dialog_text", "")).strip()
    latent_index = str(payload.get("latent_index", "38"))

    if not dialog_text:
        return jsonify({"ok": False, "error": "Dialog request text is required."}), 400

    allowed_face_indices = {str(v) for v in FACE_INDICES}
    if latent_index not in allowed_face_indices:
        return jsonify({"ok": False, "error": "Invalid dataset face index."}), 400

    cmd = [
        sys.executable,
        "editing_with_dialog.py",
        "--opt",
        str(DIALOG_CONFIG),
        "--latent_index",
        latent_index,
        "--request",
        dialog_text,
        "--request",
        "That's all, thank you.",
        "--auto_end_on_exhausted",
        "--max_rounds",
        "6",
    ]

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Dialog edit timed out after 15 minutes."}), 504

    if completed.returncode != 0:
        error_output = (completed.stderr or completed.stdout or "").strip()
        return jsonify(
            {
                "ok": False,
                "error": "Dialog editing command failed.",
                "details": error_output[:2500],
            }
        ), 500

    latest_image = _latest_image(RESULTS_DIALOG_DIR, exclude_start=True)
    if latest_image is None:
        return jsonify(
            {
                "ok": True,
                "message": "Dialog edit completed, no edited image found yet.",
                "input_face_url": _start_image(RESULTS_DIALOG_DIR),
            }
        )

    return jsonify(
        {
            "ok": True,
            "image_url": latest_image,
            "input_face_url": _start_image(RESULTS_DIALOG_DIR),
        }
    )


@app.post("/api/preview-face")
def preview_face():
    payload = request.get_json(silent=True) or {}
    latent_index = str(payload.get("latent_index", "38"))
    allowed_face_indices = {str(v) for v in FACE_INDICES}
    if latent_index not in allowed_face_indices:
        return jsonify({"ok": False, "error": "Invalid dataset face index."}), 400

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    preview_file = PREVIEW_DIR / f"face_{int(latent_index):03d}.png"

    if not preview_file.exists():
        cmd = [
            sys.executable,
            "editing_wo_dialog.py",
            "--opt",
            str(DEFAULT_CONFIG),
            "--attr",
            "Bangs",
            "--target_val",
            "0",
            "--latent_index",
            latent_index,
            "--preview_only",
            "--preview_output",
            str(preview_file),
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return jsonify({"ok": False, "error": "Preview generation timed out."}), 504

        if completed.returncode != 0:
            error_output = (completed.stderr or completed.stdout or "").strip()
            return jsonify(
                {
                    "ok": False,
                    "error": "Failed to generate dataset face preview.",
                    "details": error_output[:2500],
                }
            ), 500

    return jsonify(
        {
            "ok": True,
            "input_face_url": f"/results/editing_wo_dialog/dataset_preview/{preview_file.name}",
            "latent_index": int(latent_index),
            "generated_at": int(time.time()),
        }
    )


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(ROOT_DIR / "results", filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
