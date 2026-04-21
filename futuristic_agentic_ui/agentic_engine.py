import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "editing" / "editing_wo_dialog.yml"
VISUALIZATION_DIR = PROJECT_ROOT / "results" / "editing_wo_dialog" / "visualization"
AGENT_RUNS_DIR = PROJECT_ROOT / "results" / "agentic_runs"
KB_PATH = Path(__file__).resolve().parent / "knowledge_base.json"


@dataclass
class Plan:
    attribute: str
    attribute_label: str
    desired_intensity: int
    confidence: float
    matched_statement: str


def _load_kb() -> Dict:
    with KB_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().strip().split())


def _count_phrase_hits(text: str, phrases: List[str]) -> int:
    return sum(1 for p in (phrases or []) if p and p in text)


def _extract_level_hint(text: str) -> Optional[int]:
    markers = ("intensity", "level", "score", "strength")
    tokens = text.replace(":", " ").replace(",", " ").split()
    for idx, tok in enumerate(tokens):
        if tok in markers and idx + 1 < len(tokens):
            nxt = tokens[idx + 1]
            if nxt.isdigit():
                val = int(nxt)
                if 0 <= val <= 5:
                    return val
    return None


def _intensity_language_votes(text: str, intensity_lang: Dict[str, List[str]]) -> Tuple[Dict[int, float], int]:
    level_votes = {i: 0.0 for i in range(6)}
    level_map = {
        0: intensity_lang.get("ultra_low", []),
        1: intensity_lang.get("very_low", []),
        2: intensity_lang.get("low", []),
        3: intensity_lang.get("medium", []),
        4: intensity_lang.get("high", []),
        5: intensity_lang.get("very_high", []),
    }
    total_hits = 0
    for level, cues in level_map.items():
        hits = _count_phrase_hits(text, cues)
        if hits > 0:
            level_votes[level] += hits * 1.9
            total_hits += hits
    return level_votes, total_hits


def _infer_attr_intensity(
    *,
    text: str,
    attr: Dict,
    intensity_lang: Dict,
) -> Tuple[int, float]:
    desired = int(attr["default_intensity"])
    tokens = text.replace(";", ".").replace(",", ".").split(".")
    attr_segments = [
        seg.strip()
        for seg in tokens
        if seg.strip() and any(k in seg for k in attr.get("keywords", []))
    ]

    global_votes, global_hits = _intensity_language_votes(text, intensity_lang)
    local_text = " ".join(attr_segments) if attr_segments else ""
    local_votes, local_hits = _intensity_language_votes(local_text, intensity_lang)
    for level in range(6):
        global_votes[level] += local_votes[level] * 1.35
    total_intensity_hits = global_hits + local_hits

    explicit_level = _extract_level_hint(local_text or text)
    if explicit_level is not None:
        global_votes[explicit_level] += 4.5
        total_intensity_hits += 2

    pos_hits = _count_phrase_hits(text, attr.get("positive_keywords", []))
    neg_hits = _count_phrase_hits(text, attr.get("negative_keywords", []))
    if pos_hits > 0:
        global_votes[min(5, desired + 1)] += 1.5 * pos_hits
        global_votes[min(5, desired + 2)] += 0.9 * pos_hits
        total_intensity_hits += pos_hits
    if neg_hits > 0:
        global_votes[max(0, desired - 1)] += 1.5 * neg_hits
        global_votes[max(0, desired - 2)] += 0.9 * neg_hits
        total_intensity_hits += neg_hits

    voted_level = max(global_votes, key=lambda k: global_votes[k])
    if global_votes[voted_level] > 0:
        desired = voted_level
    desired = max(0, min(5, int(desired)))

    intensity_conf = min(
        0.995,
        0.68
        + min(0.24, total_intensity_hits * 0.08)
        + (0.05 if local_hits > 0 else 0.0)
        + (0.08 if explicit_level is not None else 0.0),
    )
    return desired, intensity_conf


def build_plans(prompt: str) -> Tuple[List[Plan], Dict]:
    kb = _load_kb()
    text = _normalize(prompt)
    max_active_agents = int(kb.get("max_active_agents", len(kb.get("attributes", [])) or 1))
    max_active_agents = max(1, min(max_active_agents, len(kb.get("attributes", [])) or 1))
    scored = []
    for attr in kb["attributes"]:
        keyword_hits = _count_phrase_hits(text, attr["keywords"])
        score = 0.0
        score += 2.6 * keyword_hits
        score += 1.2 * _count_phrase_hits(text, attr["positive_keywords"])
        score += 0.9 * _count_phrase_hits(text, attr["negative_keywords"])

        if keyword_hits > 0:
            score += 0.6  # bonus for explicit attribute intent

        scored.append((score, keyword_hits, attr))

    scored.sort(key=lambda x: x[0], reverse=True)
    explicit_selected = [(s, a) for (s, kh, a) in scored if s > 0 and kh > 0]
    selected: List[Tuple[float, Dict]] = explicit_selected
    if not selected:
        selected = [(s, a) for (s, _, a) in scored if s > 0]
    if not selected:
        selected = [(0.8, kb["attributes"][0])]
    selected = selected[:max_active_agents]

    intensity_lang = kb["intensity_language"]
    plans: List[Plan] = []
    for idx, (attr_score, attr) in enumerate(selected):
        second_best_score = selected[idx + 1][0] if idx + 1 < len(selected) else 0.5
        desired, intensity_conf = _infer_attr_intensity(
            text=text,
            attr=attr,
            intensity_lang=intensity_lang,
        )
        attr_gap = max(0.0, attr_score - max(0.0, second_best_score))
        attr_conf = min(0.99, 0.58 + min(0.24, attr_score / 28.0) + min(0.12, attr_gap / 10.0))
        confidence = min(0.995, 0.12 + 0.30 * attr_conf + 0.58 * intensity_conf)
        plans.append(
            Plan(
                attribute=attr["id"],
                attribute_label=attr["label"],
                desired_intensity=desired,
                confidence=round(confidence, 3),
                matched_statement=(
                    "Statement A + Statement B: attribute and intensity inferred from weighted language cues."
                ),
            )
        )
    return plans, kb


def _latest_visual_after(ts: float) -> Optional[Path]:
    if not VISUALIZATION_DIR.exists():
        return None

    candidates = []
    for p in VISUALIZATION_DIR.glob("*.png"):
        if p.name == "start_image.png":
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if mtime >= ts:
            candidates.append((mtime, p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _score_candidate(intensity: int, desired: int, cycle_idx: int) -> float:
    dist = abs(intensity - desired)
    semantic = max(0.0, 1.0 - (dist / 5.0))
    cycle_bonus = 0.08 if cycle_idx > 0 else 0.0

    # A tiny deterministic tie-break helps keep outputs stable across equal scores.
    tie_break = ((intensity * 17 + cycle_idx * 7) % 9) * 0.001
    score = semantic + cycle_bonus + tie_break
    return round(score, 4)


def _robust_candidate_scores(
    *,
    intensity: int,
    desired: int,
    cycle_idx: int,
    total_cycles: int,
    plan_confidence: float,
) -> Dict[str, float]:
    distance = abs(intensity - desired)
    intensity_score = max(0.0, 1.0 - (distance / 5.0))
    refinement_bonus = 0.08 * (cycle_idx / max(1, total_cycles - 1))
    confidence_score = (
        0.45 * float(plan_confidence)
        + 0.45 * intensity_score
        + 0.10 * (0.92 + refinement_bonus)
    )
    confidence_score = max(0.0, min(1.0, confidence_score))

    tie_break = ((intensity * 19 + cycle_idx * 11) % 13) * 0.0007
    final_score = (0.7 * intensity_score) + (0.3 * confidence_score) + tie_break

    return {
        "intensity_score": round(intensity_score, 4),
        "confidence_score": round(confidence_score, 4),
        "final_score": round(final_score, 4),
    }


def _copy_for_run(src: Path, run_id: str, cycle_idx: int, intensity: int, attribute: str) -> Path:
    AGENT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    safe_attr = attribute.replace(" ", "_")
    target = AGENT_RUNS_DIR / (
        f"{run_id}_{safe_attr}_cycle{cycle_idx + 1}_intensity{intensity}.png"
    )
    target.write_bytes(src.read_bytes())
    return target


def _generate_dataset_preview(
    latent_index: int,
    output_stem: str,
    timeout: int = 900,
    use_cache: bool = True,
) -> Tuple[bool, str, Optional[Path], bool]:
    AGENT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = AGENT_RUNS_DIR / f"{output_stem}.png"
    if use_cache and preview_path.exists():
        return True, "ok", preview_path, True
    cmd = [
        sys.executable,
        "editing_wo_dialog.py",
        "--opt",
        str(DEFAULT_CONFIG),
        "--attr",
        "Smiling",
        "--target_val",
        "3",
        "--latent_index",
        str(latent_index),
        "--preview_only",
        "--preview_output",
        str(preview_path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"Preview timed out after {timeout} seconds.", None, False

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        return False, details[:2500], None, False

    if not preview_path.exists():
        return False, "Preview generation completed but output file was not found.", None, False

    return True, "ok", preview_path, False


def prepare_dataset_preview(latent_index: int) -> Tuple[bool, str, Optional[str], bool]:
    output_stem = f"dataset_preview_latent_{latent_index}"
    ok, detail, preview_path, was_cached = _generate_dataset_preview(
        latent_index=latent_index,
        output_stem=output_stem,
        use_cache=True,
    )
    if not ok or preview_path is None:
        return False, detail, None, was_cached
    return True, "ok", f"/results/agentic_runs/{preview_path.name}", was_cached


def _dataset_preview_url(latent_index: int) -> str:
    return f"/results/agentic_runs/dataset_preview_latent_{latent_index}.png"


def _dataset_preview_path(latent_index: int) -> Path:
    return AGENT_RUNS_DIR / f"dataset_preview_latent_{latent_index}.png"


def _run_single_edit(
    attribute: str,
    intensity: int,
    latent_index: Optional[int],
    timeout: Optional[int] = None,
) -> Tuple[bool, str, Optional[Path]]:
    ts = time.time()
    cmd = [
        sys.executable,
        "editing_wo_dialog.py",
        "--opt",
        str(DEFAULT_CONFIG),
        "--attr",
        attribute,
        "--target_val",
        str(intensity),
    ]
    if latent_index is not None:
        cmd.extend(["--latent_index", str(latent_index)])
    else:
        return False, "No valid input source was provided.", None

    try:
        effective_timeout = None
        if timeout is not None and int(timeout) > 0:
            effective_timeout = int(timeout)
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "Timed out while waiting for edit completion.", None

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        return False, details[:2500], None

    out_path = _latest_visual_after(ts)
    if out_path is None:
        return False, "Edit finished but no new visualization image was found.", None

    return True, "ok", out_path


def _candidate_schedule_for_plan(
    *,
    plan: Plan,
    cycles: int,
    first_cycle: List[int],
    neighbor_window: int,
    multi_agent_mode: bool,
) -> List[List[int]]:
    if not multi_agent_mode:
        return [list(first_cycle)] + [[] for _ in range(1, cycles)]

    # In multi-agent mode, keep sweeps focused so the framework is practical in realtime.
    center = int(plan.desired_intensity)
    focused = sorted({max(0, min(5, center + d)) for d in (-1, 0, 1)})
    return [focused] + [[] for _ in range(1, cycles)]


def run_agentic_cycles(
    *,
    prompt: str,
    latent_index: Optional[int],
    run_id: str,
    emit,
    on_preview=None,
    on_candidate=None,
    on_progress=None,
) -> Dict:
    plans, kb = build_plans(prompt)
    policy = kb["agent_policy"]
    per_edit_timeout = None

    for plan in plans:
        emit(
            "llm",
            (
                f"Agent assigned for {plan.attribute_label}. "
                f"Desired intensity: {plan.desired_intensity}, confidence: {plan.confidence}."
            ),
        )
    emit("llm", "Launching agentic attribute sweep.")

    first_cycle = [int(x) for x in policy.get("first_cycle", [0, 1, 2, 3, 4, 5])]
    neighbor_window = int(policy.get("neighbor_window", 1))
    cycles = int(policy.get("cycles", 2))
    multi_agent_mode = len(plans) > 1
    if multi_agent_mode:
        emit(
            "system",
            "Multi-agent mode active: using focused intensity sweeps per attribute for faster completion.",
        )
        cycles = 1
    per_agent_steps = 3 + max(0, (cycles - 1)) * ((2 * neighbor_window) + 1) if multi_agent_mode else (
        len(first_cycle) + max(0, (cycles - 1)) * ((2 * neighbor_window) + 1)
    )
    total_steps = per_agent_steps * len(plans)
    completed_steps = 0
    step_durations = []

    if on_progress:
        on_progress(
            {
                "completed_steps": 0,
                "total_steps": total_steps,
                "percent": 0.0,
                "eta_seconds": None,
                "message": "Generating selected dataset image preview...",
            }
        )

    preview_path = None
    preview_url = None
    if latent_index is not None:
        ok_preview, preview_detail, preview_url, _ = prepare_dataset_preview(latent_index)
        candidate_preview_path = _dataset_preview_path(latent_index)
        if ok_preview and preview_url is not None and candidate_preview_path.exists():
            preview_path = candidate_preview_path
            emit("system", f"Selected dataset face preview ready for latent index {latent_index}.")
            if on_preview:
                on_preview(preview_url)
        else:
            emit("system", f"Preview generation failed. Continuing without preview: {preview_detail}")
            # If preview generation failed this run but a cached file exists, still use it for fallback display.
            if candidate_preview_path.exists():
                preview_path = candidate_preview_path
                preview_url = _dataset_preview_url(latent_index)

    all_sweep_results: List[Dict] = []
    agent_outputs: List[Dict] = []

    for agent_idx, plan in enumerate(plans, start=1):
        emit(
            "agent",
            (
                f"[{plan.attribute_label} Agent {agent_idx}/{len(plans)}] "
                f"starting sweep."
            ),
        )
        best = {
            "score": -math.inf,
            "path": None,
            "intensity": None,
            "cycle": None,
        }
        no_visualization_failure_count = 0
        agent_sweeps: List[Dict] = []
        agent_status = "completed"
        agent_error = None
        schedule_hint = _candidate_schedule_for_plan(
            plan=plan,
            cycles=cycles,
            first_cycle=first_cycle,
            neighbor_window=neighbor_window,
            multi_agent_mode=multi_agent_mode,
        )

        for cycle_idx in range(cycles):
            if cycle_idx == 0:
                candidates = schedule_hint[0] if schedule_hint else first_cycle
            else:
                center = (
                    best["intensity"]
                    if best["intensity"] is not None
                    else plan.desired_intensity
                )
                if multi_agent_mode:
                    candidates = sorted({max(0, min(5, center + d)) for d in (-1, 0, 1)})
                else:
                    candidates = sorted(
                        {
                            max(0, min(5, center + delta))
                            for delta in range(-neighbor_window, neighbor_window + 1)
                        }
                    )

            emit(
                "agent",
                (
                    f"[{plan.attribute_label} Agent] Cycle {cycle_idx + 1}/{cycles} "
                    f"intensities: {candidates}"
                ),
            )

            stop_current_agent = False
            for idx, intensity in enumerate(candidates, start=1):
                emit(
                    "agent",
                    (
                        f"[{plan.attribute_label} Agent] running {idx}/{len(candidates)} "
                        f"at intensity {intensity}."
                    ),
                )
                if on_progress:
                    remaining_with_current = max(0, total_steps - completed_steps)
                    if step_durations:
                        avg_step = sum(step_durations) / max(1, len(step_durations))
                        eta_before = int(round(avg_step * remaining_with_current))
                    else:
                        eta_before = None
                    on_progress(
                        {
                            "completed_steps": completed_steps,
                            "total_steps": total_steps,
                            "percent": round((completed_steps / max(1, total_steps)) * 100.0, 1),
                            "eta_seconds": eta_before,
                            "message": (
                                f"Agent {agent_idx}/{len(plans)} ({plan.attribute_label}) is "
                                f"processing cycle {cycle_idx + 1}/{cycles}, "
                                f"candidate {idx}/{len(candidates)}..."
                            ),
                        }
                    )

                t0 = time.time()
                ok, detail, image_path = _run_single_edit(
                    attribute=plan.attribute,
                    intensity=intensity,
                    latent_index=latent_index,
                    timeout=per_edit_timeout,
                )
                duration = time.time() - t0
                completed_steps += 1
                step_durations.append(duration)
                avg_step = sum(step_durations) / max(1, len(step_durations))
                remaining = max(0, total_steps - completed_steps)
                eta_seconds = int(round(avg_step * remaining))
                if on_progress:
                    on_progress(
                        {
                            "completed_steps": completed_steps,
                            "total_steps": total_steps,
                            "percent": round((completed_steps / max(1, total_steps)) * 100.0, 1),
                            "eta_seconds": eta_seconds,
                            "message": (
                                f"Agent {agent_idx}/{len(plans)} ({plan.attribute_label}), "
                                f"cycle {cycle_idx + 1}/{cycles}, "
                                f"candidate {idx}/{len(candidates)} completed."
                            ),
                        }
                    )

                if not ok:
                    emit(
                        "system",
                        f"[{plan.attribute_label} Agent] failed at intensity {intensity}: {detail}",
                    )
                    detail_lower = (detail or "").lower()
                    is_no_visual = "no new visualization image was found" in detail_lower
                    is_timeout = "timed out after" in detail_lower
                    if is_no_visual or is_timeout:
                        no_visualization_failure_count += 1
                        fallback_url = None
                        if preview_path is not None:
                            fallback_url = f"/results/agentic_runs/{preview_path.name}"
                        elif latent_index is not None:
                            cached_candidate = _dataset_preview_path(latent_index)
                            if cached_candidate.exists():
                                preview_path = cached_candidate
                                fallback_url = _dataset_preview_url(latent_index)

                        if fallback_url is not None:
                            result_item = {
                                "agent_id": f"agent_{agent_idx}_{plan.attribute}",
                                "attribute": plan.attribute,
                                "attribute_label": plan.attribute_label,
                                "cycle": cycle_idx + 1,
                                "intensity": intensity,
                                "score": 0.0,
                                "intensity_score": 0.0,
                                "confidence_score": 0.0,
                                "image_url": fallback_url,
                                "unchanged": True,
                                "failure_reason": (
                                    "timeout" if is_timeout else "no_new_visualization"
                                ),
                            }
                            agent_sweeps.append(result_item)
                            all_sweep_results.append(result_item)
                            if on_candidate:
                                on_candidate(result_item)

                        if no_visualization_failure_count >= 3:
                            agent_status = "failed"
                            agent_error = (
                                "Selected attribute is unchangeable, try a different attribute."
                            )
                            emit("LLM", f" {agent_error}")
                            stop_current_agent = True
                            break
                    continue

                local_copy = _copy_for_run(
                    src=image_path,
                    run_id=run_id,
                    cycle_idx=cycle_idx,
                    intensity=intensity,
                    attribute=plan.attribute,
                )

                robust_scores = _robust_candidate_scores(
                    intensity=intensity,
                    desired=plan.desired_intensity,
                    cycle_idx=cycle_idx,
                    total_cycles=cycles,
                    plan_confidence=plan.confidence,
                )

                result_item = {
                    "agent_id": f"agent_{agent_idx}_{plan.attribute}",
                    "attribute": plan.attribute,
                    "attribute_label": plan.attribute_label,
                    "cycle": cycle_idx + 1,
                    "intensity": intensity,
                    "score": robust_scores["final_score"],
                    "intensity_score": robust_scores["intensity_score"],
                    "confidence_score": robust_scores["confidence_score"],
                    "image_url": f"/results/agentic_runs/{local_copy.name}",
                    "unchanged": False,
                }
                agent_sweeps.append(result_item)
                all_sweep_results.append(result_item)
                if on_candidate:
                    on_candidate(result_item)

                emit(
                    "llm",
                    (
                        f"[{plan.attribute_label} Agent] scored intensity={intensity}: "
                        f"final={result_item['score']}, "
                        f"intensity_score={result_item['intensity_score']}, "
                        f"confidence_score={result_item['confidence_score']}."
                    ),
                )

                if result_item["score"] > best["score"]:
                    best.update(
                        {
                            "score": result_item["score"],
                            "path": local_copy,
                            "intensity": intensity,
                            "cycle": cycle_idx + 1,
                        }
                    )

            if stop_current_agent:
                break

        best_payload = None
        if best["path"] is not None:
            best_payload = {
                "cycle": best["cycle"],
                "intensity": best["intensity"],
                "score": round(float(best["score"]), 4),
                "image_url": f"/results/agentic_runs/{Path(best['path']).name}",
            }
            emit(
                "llm",
                (
                    f"[{plan.attribute_label} Agent] final selection: intensity {best['intensity']}, "
                    f"cycle {best['cycle']}, score {round(best['score'], 4)}."
                ),
            )
        elif agent_status != "failed":
            agent_status = "failed"
            agent_error = "No successful sweep candidate was produced."

        agent_outputs.append(
            {
                "agent_id": f"agent_{agent_idx}_{plan.attribute}",
                "status": agent_status,
                "error": agent_error,
                "plan": {
                    "attribute": plan.attribute,
                    "attribute_label": plan.attribute_label,
                    "desired_intensity": plan.desired_intensity,
                    "confidence": plan.confidence,
                },
                "best": best_payload,
                "sweep_results": agent_sweeps,
            }
        )

    successful = [a for a in agent_outputs if a["status"] == "completed" and a["best"]]
    if not successful:
        unique_errors: List[str] = []
        seen_errors = set()
        for agent in agent_outputs:
            raw = (agent.get("error") or "unknown error").strip()
            key = raw.lower()
            if key in seen_errors:
                continue
            seen_errors.add(key)
            unique_errors.append(raw)
        all_errors = "; ".join(unique_errors)
        raise RuntimeError(all_errors or "No successful sweep candidate was produced.")

    overall = max(successful, key=lambda a: a["best"]["score"])
    overall_best = dict(overall["best"])
    overall_best["attribute"] = overall["plan"]["attribute"]
    overall_best["attribute_label"] = overall["plan"]["attribute_label"]

    emit(
        "llm",
        (
            f"Multi-agent run complete. Overall best: {overall_best['attribute_label']} "
            f"intensity {overall_best['intensity']} (score {overall_best['score']})."
        ),
    )
    if on_progress:
        on_progress(
            {
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "percent": 100.0,
                "eta_seconds": 0,
                "message": "Run complete.",
            }
        )

    return {
        "multi_agent": True,
        "plan": successful[0]["plan"],
        "plans": [a["plan"] for a in agent_outputs],
        "best": overall_best,
        "agents": agent_outputs,
        "sweep_results": all_sweep_results,
    }
