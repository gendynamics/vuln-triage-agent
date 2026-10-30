//! Synthetic evaluation harness: `cargo run --release --example eval`.
//! Reports empirical false-accept / false-reject on SYNTHETIC sequences only.
//! These numbers do not measure real-world accuracy.
use aura_rs::scan::{scan_window, synth::{sequence, LIVES, SPOOFS}, ScanStatus};

fn main() {
    let seeds = 0..16u64;
    let (mut fa, mut ns) = (0, 0);
    for k in SPOOFS {
        let mut acc = 0;
        for s in seeds.clone() { ns += 1; let r = scan_window(true, sequence(k, 150, 33, s)); if r.status == ScanStatus::Pass || r.legacy_pass { acc += 1; } }
        let r = scan_window(true, sequence(k, 150, 33, 11));
        println!("spoof {:<18} accepted {}/16  vetoes(seed 11) {:?}", format!("{:?}", k), acc, r.vetoes);
        fa += acc;
    }
    let (mut fr, mut nl) = (0, 0);
    for k in LIVES {
        let (mut ha, mut lg) = (0, 0);
        for s in seeds.clone() { nl += 1; let r = scan_window(true, sequence(k, 150, 33, s)); if r.high_assurance_pass { ha += 1; } if r.legacy_pass { lg += 1; } else { fr += 1; } }
        let r = scan_window(true, sequence(k, 150, 33, 11));
        println!("live  {:<18} high-assurance {}/16 legacy {}/16  total_llr(seed 11) {:.2} posterior {:.6}", format!("{:?}", k), ha, lg, r.evidence.total_llr, r.posterior);
    }
    println!("synthetic FAR {}/{}  synthetic FRR(legacy tier) {}/{}  (uncalibrated)", fa, ns, fr, nl);
}
