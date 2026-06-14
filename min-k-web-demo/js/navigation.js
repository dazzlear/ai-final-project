/**
 * navigation.js
 * Handles top-level view switching (Interactive Demo ↔ Results Dashboard)
 * and model-tab switching inside the results panel.
 */

const NAV_ACTIVE   = "px-6 py-2 rounded-xl text-sm font-bold tracking-wide transition-all bg-purple-500/20 text-purple-300 shadow-[0_0_10px_rgba(168,85,247,0.2)]";
const NAV_INACTIVE = "px-6 py-2 rounded-xl text-sm font-bold tracking-wide transition-all text-slate-400 hover:text-slate-200 hover:bg-white/5";

window.switchView = function (view) {
    if (view === 'demo') {
        DOM.tabDemo.className = NAV_ACTIVE;
        DOM.tabDash.className = NAV_INACTIVE;
        DOM.viewInteractive.classList.remove('hidden');
        DOM.viewDashboard.classList.add('hidden');
    } else {
        DOM.tabDash.className  = NAV_ACTIVE;
        DOM.tabDemo.className  = NAV_INACTIVE;
        DOM.viewDashboard.classList.remove('hidden');
        DOM.viewInteractive.classList.add('hidden');
    }
};

window.switchTab = function (activeIndex) {
    const tabs   = DOM.modelTabs.children;
    const panels = DOM.tabContents.children;

    for (let i = 0; i < tabs.length; i++) {
        if (i === activeIndex) {
            tabs[i].className = 'px-8 py-5 text-sm font-bold tracking-widest uppercase border-b-2 whitespace-nowrap transition-all outline-none border-indigo-400 text-indigo-300 bg-white/5';
            panels[i].classList.remove('hidden');
            panels[i].classList.add('block');
        } else {
            tabs[i].className = 'px-8 py-5 text-sm font-bold tracking-widest uppercase border-b-2 whitespace-nowrap transition-all outline-none border-transparent text-slate-500 hover:text-slate-300 hover:bg-white/5';
            panels[i].classList.remove('block');
            panels[i].classList.add('hidden');
        }
    }

    // Sync row indicator dots in the summary table
    document.querySelectorAll('[class*="tab-indicator-"]').forEach((el, i) => {
        el.className = i === activeIndex
            ? `w-2 h-2 rounded-full bg-indigo-400 shadow-[0_0_8px_#818cf8] transition-all duration-300 tab-indicator-${i}`
            : `w-2 h-2 rounded-full bg-white/20 transition-all duration-300 tab-indicator-${i}`;
    });
};
