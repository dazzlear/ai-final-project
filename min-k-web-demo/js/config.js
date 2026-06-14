/**
 * config.js
 * Static configuration for the real backend version.
 */

const MODEL_PRESETS = [
    { id: "EleutherAI/pythia-410m",    label: "Pythia-410M", hint: "Smoke test model",        selected: true },
    { id: "EleutherAI/pythia-2.8b",    label: "Pythia-2.8B", hint: "Main model",              selected: false },
    { id: "EleutherAI/gpt-neo-1.3B",   label: "GPT-Neo-1.3B", hint: "Main model",             selected: false },
    { id: "facebook/opt-1.3b",         label: "OPT-1.3B",    hint: "Main model",              selected: false }
];

const SAMPLE_TEXT =
    "The 2016 Summer Olympics were held in Rio de Janeiro, Brazil, from 5 to 21 August 2016. " +
    "More than eleven thousand athletes from over two hundred national teams competed in different sporting events during the Games.";

const DEFAULT_MAX_LENGTH = 128;
