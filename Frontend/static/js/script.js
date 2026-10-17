const form = document.getElementById("upload-form");
const input = document.getElementById("image-input");
const dropzone = document.getElementById("dropzone");
const preview = document.getElementById("preview");
const dropText = document.getElementById("drop-text");
const analyseBtn = document.getElementById("analyse-btn");
const resetBtn = document.getElementById("reset-btn");

const emptyBox = document.getElementById("empty");
const loadingBox = document.getElementById("loading");
const errorBox = document.getElementById("error");
const resultBox = document.getElementById("result");
const verdict = document.getElementById("verdict");
const confidence = document.getElementById("confidence");
const scoresList = document.getElementById("scores");
const demoNote = document.getElementById("demo-note");

function show(which) {
  emptyBox.hidden = which !== "empty";
  loadingBox.hidden = which !== "loading";
  errorBox.hidden = which !== "error";
  resultBox.hidden = which !== "result";
}

function showPreview(file) {
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  dropText.hidden = true;
  analyseBtn.disabled = false;
}

input.addEventListener("change", () => {
  if (input.files[0]) showPreview(input.files[0]);
});

["dragenter", "dragover"].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach(evt =>
  dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove("drag"); }));

dropzone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (file) {
    input.files = e.dataTransfer.files;
    showPreview(file);
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  preview.hidden = true;
  preview.src = "";
  dropText.hidden = false;
  analyseBtn.disabled = true;
  show("empty");
});

form.addEventListener("submit", async e => {
  e.preventDefault();
  if (!input.files[0]) return;

  show("loading");
  analyseBtn.disabled = true;

  try {
    const res = await fetch("/predict", { method: "POST", body: new FormData(form) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Something went wrong.");
    renderResult(data);
  } catch (err) {
    errorBox.textContent = err.message;
    show("error");
  } finally {
    analyseBtn.disabled = false;
  }
});

function renderResult(data) {
  verdict.textContent = data.label;
  verdict.className = "verdict " + data.label;
  confidence.textContent = `Confidence: ${data.confidence}%`;
  demoNote.hidden = !data.demo;

  scoresList.innerHTML = "";
  Object.entries(data.scores)
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, pct]) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${name}</span>
        <div class="track"><div class="bar" style="width:${pct}%"></div></div>
        <span class="pct">${pct}%</span>`;
      scoresList.appendChild(li);
    });

  show("result");
}