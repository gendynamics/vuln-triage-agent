# VulnTriage GUI

Single-file, no-build prototype console for the triage pipeline. Open `index.html` by double-clicking it (fonts load from Google Fonts and fall back to system fonts offline).

- **01 Report**: textarea prefilled with `samples/sqli_report.txt`.
- **02 AURA gate**: sliders for glint arc ratio, eye/mouth width, viseme match, close/range distance, and a skip toggle. The verdict is a line-for-line JS port of `src/aura.py` `fuse()`, with the same constants, weights, log-odds, and 0.85 threshold. A failed gate blocks triage, just like `agent.triage()`.
- **03 Model tier**: nano / super / ultra cards with the real `MODEL_ROUTING` ids.
- **04 Output**: CVSS gauge, severity, MITRE ATT&CK technique, summary, remediation checklist, assets, effort, priority, and raw JSON with a copy button. `_validate()` is ported to JS and shown as the schema status.

**Demo mode** (default) returns a clearly labelled demo result built from the report text. No model is called.
**Live mode** takes a Nebius key that stays in memory only and calls `https://api.tokenfactory.nebius.com/v1/chat/completions` with the same system prompt and user template, at temperature 0. Browsers may block this with CORS. When that happens, the UI says so and falls back to demo output, so use `python src/agent.py` for real runs.

Open `index.html#demo` to auto-run a demo triage on load.
