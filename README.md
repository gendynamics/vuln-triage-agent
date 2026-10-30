# VulnTriage

AI-powered vulnerability triage with biometric submission gating.

- **Triage agent** — live Nemotron call via Nebius Token Factory, CVSS scoring, MITRE ATT&CK mapping, structured remediation JSON
- **AURA gate** — corneal glint-arc geometry + lip-sync viseme fusion (weighted log-odds, distance-adaptive weights, target false-accept 1 in 100,000)
- **Geometry discovery engine** — Fisher-information proxy, manifold curvature, topological depth steer an agent toward the hardest unsolved problems
- **CI** — GitHub Actions runs the full suite on every push

## Quick start

```bash
export NEBIUS_API_KEY=v1.your_token_factory_key
pip install -r requirements.txt
python src/agent.py
```

## Tests

```bash
bash run_tests.sh
```
