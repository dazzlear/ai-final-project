/**
 * app.js
 * Entry point — wires up DOM events and calls the real Flask backend.
 * Depends on: config.js, state.js, navigation.js, inference.js, renderer.js
 */

/* ---- Preset checkbox style helpers ---- */
window.togglePresetStyle = function (idx, isChecked) {
  const text = document.getElementById(`text-preset-${idx}`);
  const glow = document.getElementById(`glow-preset-${idx}`);
  if (isChecked) {
    text.className =
      "font-bold tracking-wide text-purple-300 transition-colors";
    glow.className =
      "absolute inset-0 bg-purple-500 blur-md opacity-20 rounded-md -z-10 transition-opacity";
  } else {
    text.className = "font-bold tracking-wide text-slate-200 transition-colors";
    glow.className =
      "absolute inset-0 bg-purple-500 blur-md opacity-0 rounded-md -z-10 transition-opacity";
  }
};

/* ---- Error helpers ---- */
function showError(msg) {
  DOM.errorMessage.textContent = msg;
  DOM.errorContainer.classList.remove("hidden");
}

function hideError() {
  DOM.errorContainer.classList.add("hidden");
}

function parseDecimalInput(value) {
  const normalized = String(value ?? "")
    .trim()
    .replace(",", ".");
  if (!normalized) return NaN;
  const number = Number(normalized);
  return Number.isFinite(number) ? number : NaN;
}

function formatThresholdInput() {
  const threshold = parseDecimalInput(DOM.thresholdValue.value);
  if (Number.isFinite(threshold)) {
    DOM.thresholdValue.value = threshold.toFixed(1);
  }
}

function getSelectedMaxLength() {
  const selected = Number.parseInt(
    DOM.maxTokens?.value ?? DEFAULT_MAX_LENGTH,
    10,
  );
  return Number.isFinite(selected) ? selected : DEFAULT_MAX_LENGTH;
}

function populateTokenLengthOptions() {
  if (!DOM.maxTokens || !Array.isArray(TOKEN_LENGTH_OPTIONS)) return;
  DOM.maxTokens.innerHTML = TOKEN_LENGTH_OPTIONS.map((length) => {
    const selected =
      Number(length) === Number(DEFAULT_MAX_LENGTH) ? "selected" : "";
    return `<option value="${length}" class="bg-slate-900" ${selected}>${length}</option>`;
  }).join("");
}

function applyThresholdToResult(result, threshold) {
  const minK = Number(result?.metrics?.min_k_score);
  if (!Number.isFinite(minK) || !Number.isFinite(threshold)) return result;

  const isMember = minK > threshold;
  result.prediction = {
    ...(result.prediction || {}),
    label: isMember ? "Likely member / seen" : "Likely non-member / unseen",
    tone: isMember ? "member" : "non-member",
    is_member: isMember,
    threshold,
    rule: "min_k_score > threshold",
    comparison: `${minK.toFixed(4)} > ${threshold.toFixed(1)}`,
  };
  result.runtime = {
    ...(result.runtime || {}),
    threshold,
  };
  return result;
}

function updateRenderedThresholdDecision() {
  const threshold = parseDecimalInput(DOM.thresholdValue.value);
  if (!Number.isFinite(threshold) || !STATE.resultsData.length) return;

  STATE.resultsData = STATE.resultsData.map((result) =>
    applyThresholdToResult(result, threshold),
  );
  renderResults();
}

/* ---- Core analysis handler ---- */
async function handleAnalyze() {
  hideError();

  // Collect selected models
  const checkboxes = document.querySelectorAll(
    'input[name="model_presets"]:checked',
  );
  let selectedModels = Array.from(checkboxes).map((cb) => cb.value);
  const customVal = DOM.customModels.value.trim();
  if (customVal)
    customVal.split(",").forEach((m) => {
      if (m.trim()) selectedModels.push(m.trim());
    });
  selectedModels = [...new Set(selectedModels)];

  if (selectedModels.length === 0)
    return showError(
      "Please select at least one model preset or enter a custom model.",
    );
  if (!DOM.inputText.value.trim())
    return showError("Please enter some text to analyze.");

  const threshold = parseDecimalInput(DOM.thresholdValue.value);
  if (!Number.isFinite(threshold))
    return showError("Please enter a valid threshold, for example -4.0.");
  formatThresholdInput();

  const maxLength = getSelectedMaxLength();

  // Loading state
  DOM.analyzeBtn.disabled = true;
  DOM.btnText.textContent = "Running Real Model...";
  DOM.loadingSpinner.classList.remove("hidden");
  DOM.resultsContainer.classList.remove("opacity-100");
  DOM.resultsContainer.classList.add("opacity-0");

  try {
    DOM.resultsContainer.classList.remove("hidden");

    const rawResults = await runBackendAnalysis(
      DOM.inputText.value.trim(),
      selectedModels,
      parseFloat(DOM.kPercent.value),
      threshold,
      maxLength,
    );

    const failedResults = rawResults.filter((r) => r.status === "error");
    const okResults = rawResults.filter((r) => r.status !== "error");

    if (failedResults.length > 0) {
      const failedNames = failedResults
        .map(
          (r) =>
            `${r.model_name || r.model || "model"}: ${r.error || "failed"}`,
        )
        .join(" | ");
      showError("Some models failed. " + failedNames);
    }

    if (okResults.length === 0) {
      throw new Error(
        "All selected models failed. Try Pythia-70M first, or run heavy models in Colab/GPU.",
      );
    }

    STATE.resultsData = okResults.map((result) =>
      applyThresholdToResult(result, threshold),
    );
    renderResults();

    setTimeout(() => {
      DOM.resultsContainer.classList.add("opacity-100");
      DOM.resultsContainer.classList.remove("opacity-0");
      DOM.resultsContainer.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }, 50);
  } catch (err) {
    showError("Backend error: " + err.message);
    DOM.resultsContainer.classList.add("hidden");
  } finally {
    DOM.analyzeBtn.disabled = false;
    DOM.btnText.textContent = "Analyze All Selected Models";
    DOM.loadingSpinner.classList.add("hidden");
  }
}

/* ---- Dashboard API helpers ---- */
let dashboardLoaded = false;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadJson(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Failed to load ${url}`);
  }
  return data;
}

function formatDashboardValue(value) {
  const raw = String(value ?? "").trim();
  if (!raw || raw.toUpperCase() === "N/A") return "N/A";
  const number = Number(raw);
  if (Number.isFinite(number)) return number.toFixed(3);
  return escapeHtml(raw);
}

function renderDashboardTable(payload, maxRows = null) {
  const columns = payload.columns || [];
  const rows = maxRows
    ? (payload.rows || []).slice(0, maxRows)
    : payload.rows || [];

  if (!columns.length || !rows.length) {
    return `<p class="text-slate-400 text-sm">No rows found in ${escapeHtml(payload.filename || "CSV file")}.</p>`;
  }

  return `
        <div class="overflow-x-auto rounded-xl border border-white/10 bg-black/10">
            <table class="w-full text-sm text-left">
                <thead class="bg-white/5 text-slate-300 uppercase text-[11px] tracking-widest">
                    <tr>
                        ${columns.map((col) => `<th class="px-4 py-3 whitespace-nowrap">${escapeHtml(col)}</th>`).join("")}
                    </tr>
                </thead>
                <tbody>
                    ${rows
                      .map((row) => {
                        const isMinK = String(row.method || "")
                          .toLowerCase()
                          .includes("min-k");
                        return `
                            <tr class="${isMinK ? "bg-purple-500/10 text-purple-100" : "text-slate-300"} border-t border-white/5">
                                ${columns
                                  .map(
                                    (col) => `
                                    <td class="px-4 py-3 whitespace-nowrap font-mono ${col === "method" ? "font-bold" : ""}">
                                        ${formatDashboardValue(row[col])}
                                    </td>
                                `,
                                  )
                                  .join("")}
                            </tr>
                        `;
                      })
                      .join("")}
                </tbody>
            </table>
        </div>
        ${
          maxRows && payload.rows && payload.rows.length > maxRows
            ? `<p class="text-xs text-slate-500 mt-3 font-mono">Showing first ${maxRows} of ${payload.rows.length} rows.</p>`
            : ""
        }
    `;
}

function renderDashboardCards(summaryPayload) {
  const cards = summaryPayload.cards || [];

  if (!cards.length) {
    return `<p class="text-slate-400 text-sm">No summary cards available yet.</p>`;
  }

  return `
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            ${cards
              .map(
                (card) => `
                <div class="rounded-2xl border border-white/10 bg-white/5 p-5 text-left">
                    <div class="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-2">${escapeHtml(card.title)}</div>
                    <div class="text-2xl font-extrabold text-white">${escapeHtml(card.value)}</div>
                    <div class="text-xs text-slate-400 mt-2 leading-relaxed">${escapeHtml(card.detail)}</div>
                </div>
            `,
              )
              .join("")}
        </div>
    `;
}

async function loadDashboard(force = false) {
  if (dashboardLoaded && !force) return;
  dashboardLoaded = true;

  if (DOM.dashboardTable1Slot) {
    DOM.dashboardTable1Slot.innerHTML = "Loading table1_results.csv...";
  }
  if (DOM.dashboardSummarySlot) {
    DOM.dashboardSummarySlot.innerHTML = "Loading evaluation_summary.csv...";
  }
  if (DOM.dashboardRocSlot) {
    DOM.dashboardRocSlot.innerHTML = "Loading roc_curve_min_k.png...";
  }

  try {
    const tablePayload = await loadJson("/api/dashboard/table1");
    DOM.dashboardTable1Slot.className = "relative z-10";
    DOM.dashboardTable1Slot.innerHTML = renderDashboardTable(tablePayload);
  } catch (error) {
    DOM.dashboardTable1Slot.innerHTML = `
            <p class="text-slate-400 text-sm leading-relaxed">
                No <code class="px-1.5 py-0.5 rounded text-slate-300 font-mono text-[0.85em] bg-white/5 border border-white/10">outputs/table1_results.csv</code> found yet.<br>
                ${escapeHtml(error.message)}
            </p>
        `;
  }

  try {
    const summaryPayload = await loadJson("/api/dashboard/summary");
    const evalPayload = await loadJson("/api/dashboard/evaluation");

    DOM.dashboardSummarySlot.className = "relative z-10 space-y-5";
    DOM.dashboardSummarySlot.innerHTML = `
            ${renderDashboardCards(summaryPayload)}
            <div>
                <h3 class="text-white font-bold mb-3 text-left">Evaluation summary preview</h3>
                ${renderDashboardTable(evalPayload, 12)}
            </div>
        `;
  } catch (error) {
    DOM.dashboardSummarySlot.innerHTML = `
            <p class="text-slate-400 text-sm leading-relaxed">
                No <code class="px-1.5 py-0.5 rounded text-slate-300 font-mono text-[0.85em] bg-white/5 border border-white/10">outputs/evaluation_summary.csv</code> found yet.<br>
                ${escapeHtml(error.message)}
            </p>
        `;
  }

  try {
    const rocPayload = await loadJson("/api/dashboard/roc-image");
    DOM.dashboardRocSlot.className =
      "relative z-10 border border-white/10 bg-black/10 rounded-xl p-4 flex items-center justify-center";
    DOM.dashboardRocSlot.innerHTML = `
            <img src="${rocPayload.url}?v=${Date.now()}" alt="ROC curve for Min-K% Prob" class="w-full rounded-xl border border-white/10 bg-white/5">
        `;
  } catch (error) {
    DOM.dashboardRocSlot.innerHTML = `
            <p class="text-slate-400 text-sm leading-relaxed">
                No <code class="px-1.5 py-0.5 rounded text-slate-300 font-mono text-[0.85em] bg-white/5 border border-white/10">figures/roc_curve_min_k.png</code> found yet.<br>
                ${escapeHtml(error.message)}
            </p>
        `;
  }
}

/* ---- Initialisation ---- */
function init() {
  populateTokenLengthOptions();

  // Navigation
  DOM.tabDemo.addEventListener("click", () => switchView("demo"));

  // Render preset checkboxes
  DOM.presetContainer.innerHTML = MODEL_PRESETS.map(
    (item, idx) => `
        <label class="flex items-center p-4 rounded-xl border border-white/5 bg-white/5 hover:bg-white/10 hover:border-white/20 cursor-pointer transition-all group backdrop-blur-sm" id="label-preset-${idx}">
            <div class="relative flex items-center justify-center mr-4">
                <input type="checkbox" name="model_presets" value="${item.id}" ${item.selected ? "checked" : ""}
                    onchange="togglePresetStyle(${idx}, this.checked)" class="peer sr-only">
                <div class="w-6 h-6 rounded-md border border-white/20 peer-checked:bg-purple-500 peer-checked:border-purple-400 transition-all flex items-center justify-center">
                    <svg class="w-4 h-4 text-white opacity-0 peer-checked:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="3">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/>
                    </svg>
                </div>
                <div class="absolute inset-0 bg-purple-500 blur-md ${item.selected ? "opacity-20" : "opacity-0"} rounded-md -z-10" id="glow-preset-${idx}"></div>
            </div>
            <div>
                <div class="font-bold tracking-wide ${item.selected ? "text-purple-300" : "text-slate-200"} transition-colors" id="text-preset-${idx}">${item.label}</div>
                <div class="text-xs text-slate-400 font-mono mt-1 opacity-80">${item.hint}</div>
            </div>
        </label>
    `,
  ).join("");

  // K% slider live update
  DOM.kPercent.addEventListener("input", (e) => {
    DOM.kValueDisplay.textContent = `${e.target.value}%`;
  });

  // Keep threshold formatting consistent as dot-decimal, e.g. -4.0.
  DOM.thresholdValue.addEventListener("input", (e) => {
    const cursorPosition = e.target.selectionStart;
    e.target.value = e.target.value.replace(",", ".");
    e.target.setSelectionRange(cursorPosition, cursorPosition);
    updateRenderedThresholdDecision();
  });
  DOM.thresholdValue.addEventListener("blur", () => {
    formatThresholdInput();
    updateRenderedThresholdDecision();
  });

  // Sample text loader
  DOM.sampleBtn.addEventListener("click", () => {
    DOM.inputText.value = SAMPLE_TEXT;
  });

  // Analyze button
  DOM.analyzeBtn.addEventListener("click", handleAnalyze);
}

init();
