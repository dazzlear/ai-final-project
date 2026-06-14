/**
 * app.js
 * Entry point — wires up DOM events and kicks off the application.
 * Depends on: config.js, state.js, navigation.js, inference.js, renderer.js
 */

/* ---- Preset checkbox style helpers ---- */
window.togglePresetStyle = function (idx, isChecked) {
    const text = document.getElementById(`text-preset-${idx}`);
    const glow = document.getElementById(`glow-preset-${idx}`);
    if (isChecked) {
        text.className = "font-bold tracking-wide text-purple-300 transition-colors";
        glow.className = "absolute inset-0 bg-purple-500 blur-md opacity-20 rounded-md -z-10 transition-opacity";
    } else {
        text.className = "font-bold tracking-wide text-slate-200 transition-colors";
        glow.className = "absolute inset-0 bg-purple-500 blur-md opacity-0 rounded-md -z-10 transition-opacity";
    }
};

/* ---- Error helpers ---- */
function showError(msg) {
    DOM.errorMessage.textContent = msg;
    DOM.errorContainer.classList.remove('hidden');
}

function hideError() {
    DOM.errorContainer.classList.add('hidden');
}

/* ---- Core analysis handler ---- */
async function handleAnalyze() {
    hideError();

    // Collect selected models
    const checkboxes     = document.querySelectorAll('input[name="model_presets"]:checked');
    let selectedModels   = Array.from(checkboxes).map(cb => cb.value);
    const customVal      = DOM.customModels.value.trim();
    if (customVal) customVal.split(',').forEach(m => { if (m.trim()) selectedModels.push(m.trim()); });
    selectedModels = [...new Set(selectedModels)];

    if (selectedModels.length === 0) return showError("Please select at least one model preset or enter a custom model.");
    if (!DOM.inputText.value.trim())  return showError("Please enter some text to analyze.");

    // Loading state
    DOM.analyzeBtn.disabled = true;
    DOM.btnText.textContent = "Processing Stream...";
    DOM.loadingSpinner.classList.remove('hidden');
    DOM.resultsContainer.classList.remove('opacity-100');
    DOM.resultsContainer.classList.add('opacity-0');

    await new Promise(r => setTimeout(r, 400));
    DOM.resultsContainer.classList.remove('hidden');

    try {
        await new Promise(r => setTimeout(r, 1200 + (selectedModels.length * 400)));

        STATE.resultsData = generateMockResults(
            DOM.inputText.value.trim(),
            selectedModels,
            parseFloat(DOM.kPercent.value),
            parseFloat(DOM.thresholdValue.value)
        );

        renderResults();

        setTimeout(() => {
            DOM.resultsContainer.classList.add('opacity-100');
            DOM.resultsContainer.classList.remove('opacity-0');
            DOM.resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 50);

    } catch (err) {
        showError("Simulation error: " + err.message);
        DOM.resultsContainer.classList.add('hidden');
    } finally {
        DOM.analyzeBtn.disabled = false;
        DOM.btnText.textContent = "Analyze All Selected Models";
        DOM.loadingSpinner.classList.add('hidden');
    }
}

/* ---- Initialisation ---- */
function init() {
    // Navigation
    DOM.tabDemo.addEventListener('click', () => switchView('demo'));
    DOM.tabDash.addEventListener('click', () => switchView('dash'));

    // Render preset checkboxes
    DOM.presetContainer.innerHTML = MODEL_PRESETS.map((item, idx) => `
        <label class="flex items-center p-4 rounded-xl border border-white/5 bg-white/5 hover:bg-white/10 hover:border-white/20 cursor-pointer transition-all group backdrop-blur-sm" id="label-preset-${idx}">
            <div class="relative flex items-center justify-center mr-4">
                <input type="checkbox" name="model_presets" value="${item.id}" ${item.selected ? 'checked' : ''}
                    onchange="togglePresetStyle(${idx}, this.checked)" class="peer sr-only">
                <div class="w-6 h-6 rounded-md border border-white/20 peer-checked:bg-purple-500 peer-checked:border-purple-400 transition-all flex items-center justify-center">
                    <svg class="w-4 h-4 text-white opacity-0 peer-checked:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="3">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7"/>
                    </svg>
                </div>
                <div class="absolute inset-0 bg-purple-500 blur-md ${item.selected ? 'opacity-20' : 'opacity-0'} rounded-md -z-10" id="glow-preset-${idx}"></div>
            </div>
            <div>
                <div class="font-bold tracking-wide ${item.selected ? 'text-purple-300' : 'text-slate-200'} transition-colors" id="text-preset-${idx}">${item.label}</div>
                <div class="text-xs text-slate-400 font-mono mt-1 opacity-80">${item.hint}</div>
            </div>
        </label>
    `).join('');

    // K% slider live update
    DOM.kPercent.addEventListener('input', e => {
        DOM.kValueDisplay.textContent = `${e.target.value}%`;
    });

    // Sample text loader
    DOM.sampleBtn.addEventListener('click', () => {
        DOM.inputText.value = SAMPLE_TEXT;
    });

    // Analyze button
    DOM.analyzeBtn.addEventListener('click', handleAnalyze);
}

init();
