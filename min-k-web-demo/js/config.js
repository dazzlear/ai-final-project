/**
 * config.js
 * Static configuration: model presets and sample text.
 */

const MODEL_PRESETS = [
    { id: "sshleifer/tiny-gpt2",        label: "Tiny GPT-2",    hint: "Fast local baseline",       selected: true  },
    { id: "EleutherAI/pythia-160m",     label: "Pythia-160M",  hint: "Small size sweep",           selected: true  },
    { id: "EleutherAI/pythia-410m",     label: "Pythia-410M",  hint: "Medium size sweep",          selected: false },
    { id: "facebook/opt-125m",          label: "OPT-125M",     hint: "Alternative architecture",   selected: false }
];

const SAMPLE_TEXT =
    "The 2014 Winter Olympics, officially known as the XXII Olympic Winter Games, " +
    "were held in Sochi, Russia, with events taking place across venues near the " +
    "Black Sea and the Caucasus Mountains. It was the first time the Olympics were " +
    "held in a CIS state.";
