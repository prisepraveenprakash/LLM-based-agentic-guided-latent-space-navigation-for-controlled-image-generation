const form = document.getElementById("run-form");
const runBtn = document.getElementById("run-btn");
const promptInput = document.getElementById("prompt");
const jobState = document.getElementById("job-state");
const timeline = document.getElementById("timeline");
const finalImage = document.getElementById("final-image");
const finalMeta = document.getElementById("final-meta");
const finalEmpty = document.getElementById("final-empty");
const finalResults = document.getElementById("final-results");
const candidatesWrap = document.getElementById("candidates");
const latentInput = document.getElementById("latent-index");
const selectedImage = document.getElementById("selected-image");
const selectedEmpty = document.getElementById("selected-empty");
const selectedLoader = document.getElementById("selected-loader");
const selectedLoaderText = document.getElementById("selected-loader-text");
const selectedEta = document.getElementById("selected-eta");
const runProgress = document.getElementById("run-progress");
const progressMeta = document.getElementById("progress-meta");
const demoButtons = Array.from(document.querySelectorAll(".demo-btn"));

let activeJobId = null;
let pollTimer = null;
let renderedEvents = 0;

let activePreviewId = null;
let previewPollTimer = null;
let previewDebounceTimer = null;
let previewStartAtMs = 0;
let previewEtaBaseSec = null;
const renderedCandidateKeys = new Set();

function clearFinalResults() {
  if (finalResults) {
    finalResults.innerHTML = "";
  }
}

function clearCandidates() {
  renderedCandidateKeys.clear();
  if (candidatesWrap) {
    candidatesWrap.innerHTML = "";
  }
}

function setState(status, text) {
  jobState.className = `job-state ${status}`;
  jobState.textContent = text;
}

function actorLabel(actor) {
  return String(actor || "system").toLowerCase();
}

function formatTime(unixTs) {
  const d = new Date((unixTs || 0) * 1000);
  return d.toLocaleTimeString();
}

function formatEta(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) {
    return "--";
  }
  const s = Math.max(0, Number(seconds));
  if (s < 60) {
    return `${Math.round(s)}s`;
  }
  const mins = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return `${mins}m ${rem}s`;
}

function setSelectedLoader(visible, message = "Loading selected dataset image...", eta = null) {
  selectedLoader.classList.toggle("hidden", !visible);
  selectedLoaderText.textContent = message;
  selectedEta.textContent = `ETA: ${formatEta(eta)}`;
}

function renderEvent(evt) {
  const card = document.createElement("div");
  card.className = "log";
  const actor = actorLabel(evt.actor);
  card.innerHTML = `
    <div class="head">
      <span class="actor-${actor}">${actor.toUpperCase()}</span>
      <span>${formatTime(evt.t)}</span>
    </div>
    <div>${evt.message}</div>
  `;
  timeline.appendChild(card);
  timeline.scrollTop = timeline.scrollHeight;
}

function renderCandidates(list) {
  if (!Array.isArray(list) || list.length === 0) {
    if (renderedCandidateKeys.size === 0) {
      candidatesWrap.innerHTML = '<div class="placeholder">No candidates yet.</div>';
    }
    return;
  }

  const placeholder = candidatesWrap.querySelector(".placeholder");
  if (placeholder) {
    placeholder.remove();
  }

  const sorted = [...list].sort((a, b) => {
    const scoreDiff = (Number(b.score) || 0) - (Number(a.score) || 0);
    if (scoreDiff !== 0) {
      return scoreDiff;
    }
    return (Number(a.cycle) || 0) - (Number(b.cycle) || 0);
  });
  sorted.forEach((item) => {
    const key = [
      item.agent_id || "-",
      item.attribute || "-",
      item.cycle || "-",
      item.intensity || "-",
      item.image_url || "-",
    ].join("|");

    if (renderedCandidateKeys.has(key)) {
      return;
    }
    renderedCandidateKeys.add(key);

    const card = document.createElement("article");
    card.className = "candidate";
    card.innerHTML = `
      <img src="${item.image_url}" alt="cycle ${item.cycle}, intensity ${item.intensity}">
      <div class="txt"><strong>${item.attribute_label || item.attribute || "Agent"}</strong></div>
      <div class="txt">Cycle ${item.cycle}</div>
      <div class="txt">Intensity ${item.intensity}</div>
      <div class="txt">Final score ${item.score}</div>
      <div class="txt">Intensity score ${item.intensity_score ?? "-"}</div>
      <div class="txt">Confidence score ${item.confidence_score ?? "-"}</div>
    `;
    candidatesWrap.prepend(card);
  });
}

function renderFinal(result) {
  if (!result) {
    return;
  }

  const agents = Array.isArray(result.agents) ? result.agents : [];
  if (agents.length === 0 && result.best) {
    clearFinalResults();
    finalImage.src = `${result.best.image_url}?t=${Date.now()}`;
    finalImage.style.display = "block";
    finalEmpty.style.display = "none";
    const plan = result.plan || {};
    finalMeta.textContent = [
      `Attribute: ${plan.attribute_label || plan.attribute || "-"}`,
      `Planner confidence: ${plan.confidence ?? "-"}`,
      `Desired intensity: ${plan.desired_intensity ?? "-"}`,
      `Selected intensity: ${result.best.intensity}`,
      `Final score: ${result.best.score}`,
      `Selected in cycle: ${result.best.cycle}`,
    ].join(" | ");
    return;
  }

  if (!result.best && agents.length === 0) {
    return;
  }

  finalImage.style.display = "none";
  finalEmpty.style.display = "none";
  clearFinalResults();

  agents.forEach((agent) => {
    if (!(agent.status === "completed" && agent.best)) {
      return;
    }
    if (!finalResults) {
      return;
    }
    const p = agent.plan || {};
    const card = document.createElement("article");
    card.className = "final-result-card";
    card.innerHTML = `
      <img src="${agent.best.image_url}?t=${Date.now()}" alt="${p.attribute_label || p.attribute || "attribute"} result">
      <div class="txt"><strong>${p.attribute_label || p.attribute || "-"}</strong></div>
      <div class="txt">Intensity ${agent.best.intensity}</div>
      <div class="txt">Score ${agent.best.score}</div>
    `;
    finalResults.appendChild(card);
  });

  const blocks = [];
  if (result.best) {
    blocks.push(
      `Overall best: ${result.best.attribute_label || result.best.attribute || "-"} ` +
      `(intensity ${result.best.intensity}, score ${result.best.score})`
    );
  }
  agents.forEach((agent) => {
    const p = agent.plan || {};
    if (agent.status === "completed" && agent.best) {
      blocks.push(
        `${p.attribute_label || p.attribute || "-"}: desired ${p.desired_intensity}, ` +
        `selected ${agent.best.intensity}, score ${agent.best.score}, ` +
        `confidence ${p.confidence}`
      );
    } else {
      blocks.push(
        `${p.attribute_label || p.attribute || "-"}: failed - ${agent.error || "unknown error"}`
      );
    }
  });
  finalMeta.textContent = blocks.join(" | ");
}

function renderFailure(job) {
  const err = job?.error || "Unknown error";
  finalImage.style.display = "none";
  clearFinalResults();
  finalEmpty.style.display = "block";
  finalEmpty.textContent = "Run failed.";
  finalMeta.textContent = `Report: ${err}`;
}

function renderSelectedImage(imageUrl) {
  if (!imageUrl) {
    return;
  }
  selectedImage.src = `${imageUrl}?t=${Date.now()}`;
  selectedImage.style.display = "block";
  selectedEmpty.style.display = "none";
}

function renderProgress(progress) {
  const p = progress || {};
  const percent = Number(p.percent || 0);
  runProgress.value = Math.max(0, Math.min(100, percent));
  const done = p.completed_steps ?? 0;
  const total = p.total_steps ?? "-";
  progressMeta.textContent = `${p.message || "Running..."} (${done}/${total})`;
}

function getLatentIndex() {
  const val = Number(latentInput.value);
  if (!Number.isInteger(val) || val < 0 || val > 99) {
    return null;
  }
  return val;
}

demoButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const sample = btn.getAttribute("data-prompt") || "";
    if (!promptInput) {
      return;
    }
    promptInput.value = sample;
    promptInput.focus();
    setState("idle", "Demo command loaded. You can edit it or launch directly.");
  });
});

function previewEtaLiveValue(serverEtaSeconds) {
  if (serverEtaSeconds === null || serverEtaSeconds === undefined) {
    return null;
  }
  if (!previewStartAtMs) {
    return Number(serverEtaSeconds);
  }
  const elapsed = (Date.now() - previewStartAtMs) / 1000;
  return Math.max(0, Number(previewEtaBaseSec ?? serverEtaSeconds) - elapsed);
}

async function pollPreviewStatus() {
  if (!activePreviewId) {
    return;
  }
  const res = await fetch(`/api/agentic/preview/status/${activePreviewId}`);
  const data = await res.json();
  if (!res.ok || !data.ok) {
    throw new Error(data.error || "Failed to fetch preview status.");
  }

  const preview = data.preview;
  if (preview.status === "running" || preview.status === "queued") {
    if (previewEtaBaseSec === null && preview.eta_seconds !== null && preview.eta_seconds !== undefined) {
      previewEtaBaseSec = Number(preview.eta_seconds);
    }
    const etaLive = previewEtaLiveValue(preview.eta_seconds);
    setSelectedLoader(true, "Loading selected dataset image...", etaLive);
    return;
  }

  if (preview.status === "completed") {
    setSelectedLoader(false);
    renderSelectedImage(preview.image_url);
    clearInterval(previewPollTimer);
    previewPollTimer = null;
    activePreviewId = null;
    return;
  }

  if (preview.status === "failed") {
    setSelectedLoader(false);
    selectedEmpty.textContent = `Preview failed: ${preview.error || "Unknown error"}`;
    selectedEmpty.style.display = "block";
    clearInterval(previewPollTimer);
    previewPollTimer = null;
    activePreviewId = null;
  }
}

async function startPreviewForSelectedIndex() {
  const latentIndex = getLatentIndex();
  if (latentIndex === null) {
    selectedEmpty.textContent = "Dataset index must be an integer between 0 and 99.";
    selectedEmpty.style.display = "block";
    setSelectedLoader(false);
    return;
  }

  if (previewPollTimer) {
    clearInterval(previewPollTimer);
    previewPollTimer = null;
  }

  setSelectedLoader(true, "Loading selected dataset image...", null);
  selectedImage.style.display = "none";
  selectedEmpty.style.display = "none";

  const payload = new FormData();
  payload.append("latent_index", String(latentIndex));
  const res = await fetch("/api/agentic/preview/start", { method: "POST", body: payload });
  const data = await res.json();
  if (!res.ok || !data.ok) {
    setSelectedLoader(false);
    selectedEmpty.textContent = data.error || "Could not start image preview.";
    selectedEmpty.style.display = "block";
    return;
  }

  activePreviewId = data.preview_id;
  previewStartAtMs = Date.now();
  previewEtaBaseSec = null;

  await pollPreviewStatus();
  previewPollTimer = setInterval(async () => {
    try {
      await pollPreviewStatus();
    } catch (err) {
      setSelectedLoader(false);
      selectedEmpty.textContent = err.message;
      selectedEmpty.style.display = "block";
      clearInterval(previewPollTimer);
      previewPollTimer = null;
      activePreviewId = null;
    }
  }, 700);
}

async function pollStatus() {
  if (!activeJobId) {
    return;
  }
  const res = await fetch(`/api/agentic/status/${activeJobId}`);
  const data = await res.json();

  if (!res.ok || !data.ok) {
    throw new Error(data.error || "Failed to fetch job status.");
  }

  const job = data.job;
  const events = job.events || [];

  for (let i = renderedEvents; i < events.length; i += 1) {
    renderEvent(events[i]);
  }
  renderedEvents = events.length;

  renderSelectedImage(job.selected_image_url);
  renderProgress(job.progress || {});
  renderCandidates(job.sweep_results || []);

  if (job.status === "running" || job.status === "queued") {
    setState("running", `Running job ${job.id.slice(0, 8)}...`);
    return;
  }

  if (job.status === "completed") {
    setState("completed", "Completed");
    renderFinal(job.result);
    clearInterval(pollTimer);
    runBtn.disabled = false;
    return;
  }

  if (job.status === "failed") {
    setState("failed", `Failed: ${job.error || "Unknown error"}`);
    renderFailure(job);
    clearInterval(pollTimer);
    runBtn.disabled = false;
  }
}

latentInput.addEventListener("input", () => {
  if (previewDebounceTimer) {
    clearTimeout(previewDebounceTimer);
  }
  previewDebounceTimer = setTimeout(() => {
    startPreviewForSelectedIndex();
  }, 500);
});

latentInput.addEventListener("change", () => {
  startPreviewForSelectedIndex();
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const prompt = document.getElementById("prompt").value.trim();
    const latentIndex = getLatentIndex();

    if (!prompt) {
      setState("failed", "Please enter an instruction.");
      return;
    }
    if (latentIndex === null) {
      setState("failed", "Dataset index must be between 0 and 99.");
      return;
    }

    timeline.innerHTML = "";
    clearCandidates();
    finalMeta.textContent = "";
    clearFinalResults();
    finalImage.style.display = "none";
    finalEmpty.style.display = "block";
    runProgress.value = 0;
    progressMeta.textContent = "Starting...";
    renderedEvents = 0;

    runBtn.disabled = true;
    setState("running", "Starting agentic job...");

    const payload = new FormData();
    payload.append("prompt", prompt);
    payload.append("latent_index", String(latentIndex));

    const res = await fetch("/api/agentic/start", {
      method: "POST",
      body: payload,
    });
    const data = await res.json();

    if (!res.ok || !data.ok) {
      runBtn.disabled = false;
      setState("failed", data.error || "Could not start job.");
      return;
    }

    activeJobId = data.job_id;
    clearInterval(pollTimer);
    await pollStatus();
    pollTimer = setInterval(async () => {
      try {
        await pollStatus();
      } catch (err) {
        setState("failed", err.message);
        clearInterval(pollTimer);
        runBtn.disabled = false;
      }
    }, 1500);
  } catch (err) {
    runBtn.disabled = false;
    setState("failed", err.message || "Launch failed.");
  }
});

startPreviewForSelectedIndex();
