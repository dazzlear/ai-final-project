/**
 * renderer.js
 * Builds and injects the results UI from STATE.resultsData:
 *   - Summary comparison table rows
 *   - Per-model tab buttons
 *   - Per-model token visualisation panels
 */

/** Build a tooltip HTML snippet for a single token. */
function buildTooltip(t) {
    return `
        <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-slate-800 text-slate-200 text-xs rounded-lg border border-slate-600 shadow-[0_10px_25px_-5px_rgba(0,0,0,0.5)] opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 whitespace-nowrap z-50 pointer-events-none font-sans flex flex-col items-center">
            <div class="font-bold text-white mb-0.5">Rank #${t.rank}</div>
            <div class="text-slate-400 font-mono text-[10px]">LogProb: ${t.logprob.toFixed(4)}</div>
            <div class="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-600"></div>
        </div>`;
}

/** Convert a token object to its pill HTML. */
function buildTokenPill(t) {
    const tooltip = buildTooltip(t);
    if (t.selected) {
        return `<span class="token-pill highlighted relative group inline-flex items-center gap-2 px-3 py-1.5 m-1 rounded-xl font-mono text-sm font-bold cursor-help">
            ${t.token} <span class="text-[10px] bg-blue-900/50 text-blue-200 px-1.5 py-0.5 rounded-md border border-blue-400/20">${t.logprob.toFixed(1)}</span>
            ${tooltip}
        </span>`;
    }
    const opacity = Math.max(0.3, 1 - (Math.abs(t.logprob) / 12));
    return `<span class="token-pill relative group inline-flex items-center gap-1 px-2 py-1 m-0.5 rounded-lg font-mono text-sm text-slate-200 hover:bg-white/10 cursor-help" style="opacity: ${opacity}">
        ${t.token}
        ${tooltip}
    </span>`;
}

/** Build the token-visualisation panel for one model. */
function buildTabPanel(res, index) {
    const tokensHTML = res.tokens.map(buildTokenPill).join('');
    const m          = res.metrics;

    const panel = document.createElement('div');
    panel.id        = `panel-${index}`;
    panel.className = `p-8 ${index === 0 ? 'block' : 'hidden'} animate-[fadeIn_0.5s_ease-out]`;

    panel.innerHTML = `
        <div class="mb-8 pb-6 border-b border-white/5">
            <!-- Title -->
            <div class="mb-5">
                <h3 class="text-sm font-bold text-slate-200 uppercase tracking-widest mb-2 flex items-center gap-2">
                    <svg class="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/>
                    </svg>
                    Token Visualization
                </h3>
                <p class="text-xs text-slate-400 font-medium leading-relaxed">
                    Glowing tokens represent the bottom K% probability subset. Hover over any token to see its exact rank.
                </p>
            </div>

            <!-- Metric Cards -->
            <div class="flex flex-row flex-wrap gap-4">
                <div class="glass-panel px-5 py-3 rounded-xl border-white/5 flex flex-col justify-center min-w-[140px]">
                    <div class="text-lg sm:text-xl font-mono font-bold text-white">
                        ${m.ppl.toFixed(1)} <span class="text-xs text-slate-500 font-normal">/ ${m.loss_score.toFixed(2)}</span>
                    </div>
                    <div class="text-[10px] text-purple-400 font-bold uppercase tracking-widest mt-1">PPL / Loss</div>
                </div>
                <div class="glass-panel px-5 py-3 rounded-xl border-white/5 flex flex-col justify-center min-w-[140px]">
                    <div class="text-lg sm:text-xl font-mono font-bold text-white">${m.min_k_score.toFixed(3)}</div>
                    <div class="text-[10px] text-blue-400 font-bold uppercase tracking-widest mt-1">Min-K Score</div>
                </div>
                <div class="glass-panel px-5 py-3 rounded-xl border-white/5 flex flex-col justify-center min-w-[140px]">
                    <div class="text-lg sm:text-xl font-mono font-bold text-white">${m.zlib_score.toFixed(4)}</div>
                    <div class="text-[10px] text-indigo-400 font-bold uppercase tracking-widest mt-1">Z-lib Score</div>
                </div>
            </div>
        </div>

        <!-- Token Stream -->
        <div class="leading-loose relative text-justify cursor-default">
            ${tokensHTML}
        </div>`;

    return panel;
}

/** Re-render the entire results section from STATE.resultsData. */
function renderResults() {
    DOM.summaryTableBody.innerHTML = '';
    DOM.modelTabs.innerHTML        = '';
    DOM.tabContents.innerHTML      = '';

    STATE.resultsData.forEach((res, index) => {
        const p          = res.prediction;
        const badgeStyle = p.tone === 'member'
            ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.15)]'
            : 'bg-amber-500/10  text-amber-400  border-amber-500/30  shadow-[0_0_15px_rgba(245,158,11,0.15)]';

        // -- Summary Table Row --
        const tr      = document.createElement('tr');
        tr.className  = "hover:bg-white/5 transition-colors cursor-pointer group";
        tr.onclick    = () => switchTab(index);
        tr.innerHTML  = `
            <td class="py-5 pl-2 pr-4 text-sm">
                <div class="flex items-center gap-3">
                    <div class="w-2 h-2 rounded-full ${index === 0 ? 'bg-indigo-400 shadow-[0_0_8px_#818cf8]' : 'bg-white/20'} transition-all duration-300 tab-indicator-${index}"></div>
                    <div>
                        <div class="font-bold text-white tracking-wide">${res.model_name.split('/').pop()}</div>
                        <div class="text-xs font-mono text-slate-500 mt-0.5 opacity-70">${res.model_name}</div>
                    </div>
                </div>
            </td>
            <td class="py-5 px-4">
                <span class="inline-flex border px-3 py-1.5 rounded-lg text-xs font-bold uppercase tracking-widest ${badgeStyle} backdrop-blur-sm">
                    ${p.label}
                </span>
            </td>
            <td class="py-5 px-4 font-mono text-sm ${p.tone === 'member' ? 'font-bold text-emerald-300' : 'text-slate-300'}">${res.metrics.min_k_score.toFixed(4)}</td>
            <td class="py-5 px-4 font-mono text-sm text-slate-400">${res.metrics.loss_score.toFixed(4)}</td>
            <td class="py-5 px-4 font-mono text-sm text-indigo-300">${res.metrics.zlib_score.toFixed(5)}</td>`;
        DOM.summaryTableBody.appendChild(tr);

        // -- Model Tab Button --
        const tabBtn      = document.createElement('button');
        tabBtn.className  = `px-8 py-5 text-sm font-bold tracking-widest uppercase border-b-2 whitespace-nowrap transition-all outline-none ${
            index === 0
                ? 'border-indigo-400 text-indigo-300 bg-white/5'
                : 'border-transparent text-slate-500 hover:text-slate-300 hover:bg-white/5'
        }`;
        tabBtn.textContent = res.model_name.split('/').pop();
        tabBtn.onclick     = () => switchTab(index);
        DOM.modelTabs.appendChild(tabBtn);

        // -- Token Visualisation Panel --
        DOM.tabContents.appendChild(buildTabPanel(res, index));
    });
}
