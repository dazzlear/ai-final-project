async function runBackendAnalysis(text, models, kPercent, threshold, maxLength = DEFAULT_MAX_LENGTH) {
    const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: text,
            models: models,
            k_percent: kPercent,
            threshold: threshold,
            max_length: maxLength
        })
    });

    let data = null;
    try {
        data = await response.json();
    } catch (err) {
        throw new Error('Backend did not return valid JSON. Check the Python terminal.');
    }

    if (!response.ok) {
        throw new Error(data.error || data.details || 'Backend request failed.');
    }

    return data.results || [];
}
