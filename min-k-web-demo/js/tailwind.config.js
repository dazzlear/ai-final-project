/**
 * tailwind.config.js
 * Min-K% Multi-Model Comparison — Tailwind configuration
 */
tailwind.config = {
    darkMode: 'class',
    theme: {
        extend: {
            fontFamily: {
                sans:  ['Inter', 'sans-serif'],
                serif: ['Lora', 'serif'],
                mono:  ['JetBrains Mono', 'monospace'],
            },
            colors: {
                glass: {
                    100:  'rgba(255, 255, 255, 0.03)',
                    200:  'rgba(255, 255, 255, 0.05)',
                    300:  'rgba(255, 255, 255, 0.08)',
                    dark: 'rgba(15, 23, 42, 0.6)',
                }
            },
            animation: {
                blob:        "blob 15s infinite alternate",
                'spin-slow': "spin 8s linear infinite",
            },
            keyframes: {
                blob: {
                    "0%":   { transform: "translate(0px, 0px) scale(1)" },
                    "33%":  { transform: "translate(40px, -60px) scale(1.1)" },
                    "66%":  { transform: "translate(-30px, 30px) scale(0.9)" },
                    "100%": { transform: "translate(0px, 0px) scale(1)" }
                }
            }
        }
    }
};
