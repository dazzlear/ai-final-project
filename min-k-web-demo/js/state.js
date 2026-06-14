/**
 * state.js
 * Centralised DOM references and mutable application state.
 * Must be loaded after the HTML body is parsed.
 */

const DOM = {
    // Sidebar
    presetContainer:  document.getElementById('presetContainer'),
    kPercent:         document.getElementById('kPercent'),
    kValueDisplay:    document.getElementById('kValueDisplay'),
    thresholdValue:   document.getElementById('thresholdValue'),
    customModels:     document.getElementById('customModels'),

    // Text input panel
    sampleBtn:        document.getElementById('sampleBtn'),
    inputText:        document.getElementById('inputText'),
    analyzeBtn:       document.getElementById('analyzeBtn'),
    btnText:          document.getElementById('btnText'),
    loadingSpinner:   document.getElementById('loadingSpinner'),
    errorContainer:   document.getElementById('errorContainer'),
    errorMessage:     document.getElementById('errorMessage'),

    // Results
    resultsContainer: document.getElementById('resultsContainer'),
    summaryTableBody: document.getElementById('summaryTableBody'),
    modelTabs:        document.getElementById('modelTabs'),
    tabContents:      document.getElementById('tabContents'),

    // Navigation
    tabDemo:          document.getElementById('tabDemo'),
    tabDash:          document.getElementById('tabDash'),
    viewInteractive:  document.getElementById('viewInteractive'),
    viewDashboard:    document.getElementById('viewDashboard'),
};

const STATE = { resultsData: [] };
