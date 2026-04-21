const form = document.getElementById("edit-form");
const runBtn = document.getElementById("run-btn");
const statusEl = document.getElementById("status");
const img = document.getElementById("result-image");
const inputImg = document.getElementById("input-image");
const imageWrap = document.getElementById("image-wrap");
const inputPlaceholder = document.getElementById("input-placeholder");
const errorLog = document.getElementById("error-log");
const latentIndex = document.getElementById("latent_index");
const progressFill = document.getElementById("progress-fill");
const modeSelect = document.getElementById("edit_mode");
const attributeControls = document.getElementById("attribute-controls");
const dialogControls = document.getElementById("dialog-controls");
const dialogText = document.getElementById("dialog_text");

let progressTimer = null;
let progressValue = 0;

function setStatus(kind, message) {
  statusEl.className = `status ${kind}`;
  statusEl.textContent = message;
}

function showError(message) {
  errorLog.style.display = "block";
  errorLog.textContent = message || "Unknown error";
}

function clearError() {
  errorLog.style.display = "none";
  errorLog.textContent = "";
}

function setLoading(isLoading) {
  runBtn.disabled = isLoading;
  latentIndex.disabled = isLoading;
  modeSelect.disabled = isLoading;
  form.attribute.disabled = isLoading;
  form.target_val.disabled = isLoading;
  if (dialogText) {
    dialogText.disabled = isLoading;
  }
  const btnText = runBtn.querySelector(".btn-text");
  const isDialogMode = modeSelect.value === "with_dialog";
  btnText.textContent = isLoading ? "Processing..." : (isDialogMode ? "Run Dialog Edit" : "Run Edit");
}

function startProgress() {
  clearInterval(progressTimer);
  progressValue = 6;
  progressFill.style.width = `${progressValue}%`;
  progressTimer = setInterval(() => {
    if (progressValue < 92) {
      progressValue += progressValue < 50 ? 4 : 1.4;
      progressFill.style.width = `${Math.min(progressValue, 92)}%`;
    }
  }, 350);
}

function completeProgress() {
  clearInterval(progressTimer);
  progressFill.style.width = "100%";
}

function resetProgress() {
  clearInterval(progressTimer);
  setTimeout(() => {
    progressFill.style.width = "0%";
  }, 450);
}

async function loadDatasetFacePreview() {
  try {
    const res = await fetch("/api/preview-face", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ latent_index: latentIndex.value }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || "Failed to load dataset face preview.");
    }
    const cacheBust = `?t=${data.generated_at || Date.now()}`;
    inputImg.src = data.input_face_url + cacheBust;
    inputImg.style.display = "block";
    inputPlaceholder.textContent = `Dataset face ${data.latent_index}`;
  } catch (err) {
    inputImg.style.display = "none";
    inputPlaceholder.textContent = "Failed to load selected dataset face.";
    showError(err.message);
  }
}

latentIndex.addEventListener("change", async () => {
  clearError();
  setStatus("running", "Loading selected dataset face preview...");
  await loadDatasetFacePreview();
  setStatus("idle", "Ready");
});

loadDatasetFacePreview();

function updateModeUI() {
  const mode = modeSelect.value;
  const dialogMode = mode === "with_dialog";
  attributeControls.style.display = dialogMode ? "none" : "grid";
  dialogControls.style.display = dialogMode ? "grid" : "none";
  runBtn.querySelector(".btn-text").textContent = dialogMode ? "Run Dialog Edit" : "Run Edit";
}

modeSelect.addEventListener("change", () => {
  clearError();
  updateModeUI();
  setStatus("idle", "Ready");
});

updateModeUI();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  const isDialogMode = modeSelect.value === "with_dialog";
  setStatus("running", isDialogMode
    ? "Running dialog edit. This can take some time..."
    : "Running edit command. This can take some time...");
  setLoading(true);
  startProgress();

  const payload = new FormData();
  payload.append("latent_index", latentIndex.value);
  let endpoint = "/api/edit";

  if (isDialogMode) {
    const prompt = (dialogText.value || "").trim();
    if (!prompt) {
      setStatus("error", "Dialog request is required.");
      showError("Please type a dialog request before running dialog mode.");
      setLoading(false);
      resetProgress();
      return;
    }
    endpoint = "/api/edit-dialog";
    payload.append("dialog_text", prompt);
  } else {
    payload.append("attribute", form.attribute.value);
    payload.append("target_val", String(Number(form.target_val.value)));
  }

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      body: payload,
    });

    const data = await res.json();
    if (!res.ok || !data.ok) {
      const details = data.details ? `\n\n${data.details}` : "";
      throw new Error(`${data.error || "Failed to run edit."}${details}`);
    }

    if (data.image_url) {
      const cacheBust = `?t=${Date.now()}`;
      img.src = data.image_url + cacheBust;
      img.style.display = "block";
      const placeholder = imageWrap.querySelector(".placeholder");
      if (placeholder) {
        placeholder.remove();
      }
      setStatus("ok", "Edit complete.");
    } else {
      setStatus("ok", data.message || "Edit completed.");
    }
    if (data.input_face_url) {
      const cacheBust = `?t=${Date.now()}`;
      inputImg.src = data.input_face_url + cacheBust;
      inputImg.style.display = "block";
      inputPlaceholder.textContent = `Model input/start face (dataset ${latentIndex.value})`;
    }
    completeProgress();
  } catch (err) {
    setStatus("error", "Edit failed.");
    showError(err.message);
    completeProgress();
  } finally {
    setLoading(false);
    resetProgress();
  }
});
