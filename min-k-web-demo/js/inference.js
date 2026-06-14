/**
 * inference.js
 * Simulated frontend Min-K% inference engine.
 * Generates mock token-level data and metrics for each selected model.
 */

/**
 * Build a mock result set for the given text and model list.
 *
 * @param {string}   text       - Input passage
 * @param {string[]} models     - Array of model id strings
 * @param {number}   kPercent   - The K% ratio (1–100)
 * @param {number}   threshold  - Membership decision threshold
 * @returns {object[]} Array of per-model result objects
 */
function generateMockResults(text, models, kPercent, threshold) {
    const rawTokens    = text.match(/[\w]+|[.,!?;:()"]/g) || [];
    const visualTokens = rawTokens.map((t, i) =>
        i === 0 ? t : (t.match(/^[.,!?;:()"]$/) ? t : "Ġ" + t)
    );

    return models.map((modelName, modelIndex) => {
        const modelSeed = modelName.length + modelIndex;

        let tokens = visualTokens.map((vTok, index) => {
            const isCommon = vTok.length < 5;
            const baseLp   = isCommon ? -(Math.random() * 3) : -(Math.random() * 10 + 2);
            const finalLp  = baseLp - (Math.random() * (modelSeed % 4));
            return {
                index:    index + 1,
                token:    vTok.replace('Ġ', '·').replace('\n', '\\n'),
                token_id: Math.floor(Math.random() * 40000) + 1000,
                logprob:  finalLp,
                selected: false,
                rank:     null,
            };
        });

        // Rank tokens by log-probability (lowest = most surprising = selected)
        const sortedIndices = tokens.map((_, i) => i).sort((a, b) => tokens[a].logprob - tokens[b].logprob);
        const kCount        = Math.max(1, Math.ceil(tokens.length * (kPercent / 100)));

        sortedIndices.forEach((origIndex, sortedRank) => {
            tokens[origIndex].rank     = sortedRank + 1;
            tokens[origIndex].selected = sortedRank < kCount;
        });

        const bottomKIndices = sortedIndices.slice(0, kCount);
        const minKScore  = bottomKIndices.reduce((a, i) => a + tokens[i].logprob, 0) / kCount;
        const lossScore  = tokens.reduce((a, t) => a + t.logprob, 0) / tokens.length;

        // Simulate compression and perplexity scores
        const mockCompressedLen = Math.max(15, text.length * 0.45);
        const zlibScore = lossScore / mockCompressedLen;
        const ppl       = Math.exp(Math.abs(lossScore));

        const isMember = minKScore > threshold;

        return {
            model_name: modelName,
            metrics: { min_k_score: minKScore, loss_score: lossScore, zlib_score: zlibScore, ppl },
            prediction: {
                label: isMember ? "Likely member / seen" : "Likely non-member / unseen",
                tone:  isMember ? "member" : "non-member",
            },
            tokens,
        };
    });
}
