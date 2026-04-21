import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request, send_from_directory

from agentic_engine import prepare_dataset_preview, run_agentic_cycles


ROOT_DIR = Path(__file__).resolve().parent.parent

app = Flask(__name__, template_folder="templates", static_folder="static")

JOBS = {}
JOBS_LOCK = threading.Lock()
PREVIEW_JOBS = {}
PREVIEW_LOCK = threading.Lock()
PREVIEW_AVG_SECONDS = 12.0


def _new_job(
    prompt: str,
    latent_index: Optional[int],
):
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "prompt": prompt,
            "latent_index": latent_index,
            "input_source": "dataset",
            "selected_image_url": None,
            "status": "queued",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "events": [],
            "sweep_results": [],
            "progress": {
                "completed_steps": 0,
                "total_steps": 1,
                "percent": 0.0,
                "eta_seconds": None,
                "message": "Queued",
            },
            "result": None,
            "error": None,
        }
    return job_id


def _push_event(job_id: str, actor: str, message: str):
    now = int(time.time())
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["events"].append(
            {
                "t": now,
                "actor": actor,
                "message": message,
            }
        )
        job["updated_at"] = now


def _set_status(job_id: str, status: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["status"] = status
        job["updated_at"] = int(time.time())


def _set_result(job_id: str, payload):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["result"] = payload
        job["updated_at"] = int(time.time())


def _set_error(job_id: str, message: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["error"] = message
        job["updated_at"] = int(time.time())


def _set_selected_image(job_id: str, image_url: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["selected_image_url"] = image_url
        job["updated_at"] = int(time.time())


def _add_candidate(job_id: str, item):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job["sweep_results"].append(item)
        job["updated_at"] = int(time.time())


def _set_progress(job_id: str, progress):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        existing = job.get("progress", {})
        existing.update(progress or {})
        job["progress"] = existing
        job["updated_at"] = int(time.time())


def _worker(job_id: str, prompt: str, latent_index: Optional[int]):
    _set_status(job_id, "running")
    _push_event(job_id, "system", "Job started. Initializing LLM planner and agentic loop.")

    def emit(actor, message):
        _push_event(job_id, actor, message)

    try:
        result = run_agentic_cycles(
            prompt=prompt,
            latent_index=latent_index,
            run_id=job_id,
            emit=emit,
            on_preview=lambda image_url: _set_selected_image(job_id, image_url),
            on_candidate=lambda item: _add_candidate(job_id, item),
            on_progress=lambda prog: _set_progress(job_id, prog),
        )
        _set_result(job_id, result)
        _set_status(job_id, "completed")
        _push_event(job_id, "system", "Job completed successfully.")
    except Exception as exc:
        _set_status(job_id, "failed")
        _set_error(job_id, str(exc))
        _push_event(job_id, "system", f"Job failed: {exc}")


def _new_preview_job(latent_index: int) -> str:
    preview_id = uuid.uuid4().hex
    with PREVIEW_LOCK:
        PREVIEW_JOBS[preview_id] = {
            "id": preview_id,
            "latent_index": latent_index,
            "status": "queued",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "eta_seconds": int(round(PREVIEW_AVG_SECONDS)),
            "image_url": None,
            "error": None,
        }
    return preview_id


def _set_preview_state(preview_id: str, **fields):
    with PREVIEW_LOCK:
        job = PREVIEW_JOBS.get(preview_id)
        if not job:
            return
        for k, v in fields.items():
            job[k] = v
        job["updated_at"] = int(time.time())


def _worker_preview(preview_id: str, latent_index: int):
    global PREVIEW_AVG_SECONDS
    _set_preview_state(preview_id, status="running", eta_seconds=int(round(PREVIEW_AVG_SECONDS)))
    t0 = time.time()
    try:
        ok, detail, image_url, was_cached = prepare_dataset_preview(latent_index)
        elapsed = max(0.0, time.time() - t0)
        if not was_cached:
            PREVIEW_AVG_SECONDS = 0.75 * PREVIEW_AVG_SECONDS + 0.25 * elapsed
        if not ok or not image_url:
            _set_preview_state(preview_id, status="failed", error=detail, eta_seconds=0)
            return
        _set_preview_state(
            preview_id,
            status="completed",
            image_url=image_url,
            eta_seconds=0,
        )
    except Exception as exc:
        _set_preview_state(preview_id, status="failed", error=str(exc), eta_seconds=0)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/agentic/start")
def start_agentic():
    payload = request.get_json(silent=True) if request.is_json else request.form
    payload = payload or {}

    prompt = str(payload.get("prompt", "")).strip()
    latent_index = None

    if not prompt:
        return jsonify({"ok": False, "error": "Prompt is required."}), 400

    try:
        latent_index = int(payload.get("latent_index", 38))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "latent_index must be an integer."}), 400
    if latent_index < 0 or latent_index > 99:
        return jsonify({"ok": False, "error": "latent_index must be between 0 and 99."}), 400

    job_id = _new_job(prompt=prompt, latent_index=latent_index)

    t = threading.Thread(
        target=_worker,
        args=(job_id, prompt, latent_index),
        daemon=True,
    )
    t.start()

    return jsonify({"ok": True, "job_id": job_id})


@app.post("/api/agentic/preview/start")
def start_preview():
    payload = request.get_json(silent=True) if request.is_json else request.form
    payload = payload or {}
    try:
        latent_index = int(payload.get("latent_index", 38))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "latent_index must be an integer."}), 400
    if latent_index < 0 or latent_index > 99:
        return jsonify({"ok": False, "error": "latent_index must be between 0 and 99."}), 400

    preview_id = _new_preview_job(latent_index)
    t = threading.Thread(target=_worker_preview, args=(preview_id, latent_index), daemon=True)
    t.start()
    return jsonify({"ok": True, "preview_id": preview_id})


@app.get("/api/agentic/preview/status/<preview_id>")
def preview_status(preview_id: str):
    with PREVIEW_LOCK:
        job = PREVIEW_JOBS.get(preview_id)
        if not job:
            return jsonify({"ok": False, "error": "Preview job not found."}), 404
        return jsonify({"ok": True, "preview": dict(job)})


@app.get("/api/agentic/status/<job_id>")
def status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "Job not found."}), 404

        return jsonify(
            {
                "ok": True,
                "job": {
                    "id": job["id"],
                    "status": job["status"],
                    "prompt": job["prompt"],
                    "latent_index": job["latent_index"],
                    "input_source": job["input_source"],
                    "selected_image_url": job["selected_image_url"],
                    "created_at": job["created_at"],
                    "updated_at": job["updated_at"],
                    "events": list(job["events"]),
                    "sweep_results": list(job["sweep_results"]),
                    "progress": dict(job.get("progress", {})),
                    "result": job["result"],
                    "error": job["error"],
                },
            }
        )


@app.get("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(ROOT_DIR / "results", filename)


@app.get("/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7861, debug=False)
