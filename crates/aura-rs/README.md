# aura-rs

Rust reference implementation of the AURA biometric liveness gate.

Same fusion math as the Python module — corneal glint-arc ratio plus
lip-sync viseme analysis via weighted log-odds — but memory-safe and
zero-cost. This is the production-hardening path: the Python agent wins
the demo, this crate is what you'd ship when the SOC scales.

```bash
cargo test
```
