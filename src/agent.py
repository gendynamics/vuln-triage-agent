#!/usr/bin/env python3
"""VulnTriage — CVSS + MITRE ATT&CK triage agent with AURA biometric gate.

Makes a live runtime call to Nebius Token Factory serving an NVIDIA Nemotron
model. Every submission passes through AURA liveness verification first.
"""

from __future__ import annotations

import json
import os
import sys

from openai import OpenAI

from aura import verify_submission

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

MODEL_ROUTING = {
    "nano":  "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "super": "nvidia/llama-3.1-nemotron-70b-instruct",
    "ultra": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
}

SYSTEM_PROMPT = (
    "You are a senior application security engineer. Given a vulnerability "
    "report, classify severity using CVSS, map it to the most relevant MITRE "
    "ATT&CK technique, and produce a remediation plan with concrete steps, "
    "affected assets, and estimated effort. Output strict JSON with keys "
    "severity, cvss_score, attack_technique, summary, remediation_steps, "
    "affected_assets, estimated_effort, and priority. No markdown, no preamble."
)

USER_TEMPLATE = "Vulnerability report:\n{report}\n\nAnalyze and respond with the JSON schema above only."

REQUIRED_KEYS = {
    "severity", "cvss_score", "attack_technique", "summary",
    "remediation_steps", "affected_assets", "estimated_effort", "priority",
}


def _client() -> OpenAI:
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        print("NEBIUS_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    if not key.startswith("v1."):
        print("Token Factory key must start with 'v1.'", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=key, base_url=NEBIUS_BASE_URL)


def _validate(resp: dict) -> dict:
    missing = REQUIRED_KEYS - set(resp.keys())
    if missing:
        raise ValueError(f"schema missing keys: {missing}")
    if resp["severity"] not in {"critical", "high", "medium", "low", "informational"}:
        raise ValueError(f"invalid severity: {resp['severity']}")
    if not (0.0 <= float(resp["cvss_score"]) <= 10.0):
        raise ValueError(f"cvss_score out of range: {resp['cvss_score']}")
    if not isinstance(resp["remediation_steps"], list) or not resp["remediation_steps"]:
        raise ValueError("remediation_steps must be a non-empty list")
    if resp["estimated_effort"] not in {"S", "M", "L"}:
        raise ValueError(f"invalid effort: {resp['estimated_effort']}")
    if not (1 <= int(resp["priority"]) <= 5):
        raise ValueError(f"priority out of bounds: {resp['priority']}")
    return resp


def scan_gate(aura_scan) -> dict | None:
    """Blocked response if a silent-scan result (ScanResult or dict) failed, else None."""
    if aura_scan is None:
        return None
    d = aura_scan.to_dict() if hasattr(aura_scan, "to_dict") else dict(aura_scan)
    passed = d.get("aura_passed", d.get("status") == "pass")
    if passed:
        return None
    return {"aura_gate": d, "triage": None, "error": "AURA liveness scan failed"}


def triage(report: str, tier: str = "ultra",
           aura: dict | None = None, aura_scan=None) -> dict:
    """Triage a vulnerability report. AURA gate runs first if readings given.
    `aura_scan` (an aura_scan.ScanResult or its dict) blocks triage when failed."""
    blocked = scan_gate(aura_scan)
    if blocked:
        return blocked
    if aura:
        verdict = verify_submission(**aura)
        if not verdict.passed:
            return {
                "aura_gate": verdict.to_dict(),
                "triage": None,
                "error": "AURA biometric verification failed",
            }

    client = _client()
    model = MODEL_ROUTING.get(tier, MODEL_ROUTING["ultra"])
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(report=report)},
        ],
        temperature=0.0,
    )
    raw = completion.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
    resp = _validate(json.loads(raw))
    if aura:
        resp["aura_gate"] = verdict.to_dict()
    if aura_scan is not None:
        resp["aura_scan"] = (aura_scan.to_dict() if hasattr(aura_scan, "to_dict") else dict(aura_scan))
    return resp


if __name__ == "__main__":
    sample = open(os.path.join(os.path.dirname(__file__), "..",
                               "samples", "sqli_report.txt")).read()
    # demo AURA readings: real eye, close range, mouth 120px, 92% viseme match
    aura_demo = {
        "glint_arc_ratio": 0.52,
        "eye_width_px": 120,
        "mouth_width_px": 120,
        "viseme_match": 0.92,
        "distance_hint": "close",
    }
    print(json.dumps(triage(sample, tier="nano", aura=aura_demo), indent=2))
