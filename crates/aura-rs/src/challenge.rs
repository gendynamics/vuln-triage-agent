//! Lip-reading CHALLENGE-RESPONSE channel.
//!
//! AURA issues a random short phrase (4-6 words from a fixed, phonetically
//! diverse vocabulary) bound to a one-time nonce. The app speaks it (TTS) or
//! displays it; the person repeats it while the silent scan runs. We verify
//! that the lip-mesh motion matches *that* phrase, *after* the prompt ended.
//!
//! Scope limit (deliberate): this is closed-vocabulary verification of a
//! prompted phrase, NOT open-ended transcription of arbitrary speech. Only
//! words in `VOCAB` can be decoded, and only a match score + pass/fail leaves
//! this module; decoded words are not returned or stored.
//!
//! Word detection: each frame's lip-mesh features (openness, spread) are
//! snapped to the nearest viseme class, short runs (<2 frames) are dropped as
//! transitions, and the stream is split into word chunks at rest pauses
//! (>=3 rest frames). Each chunk is decoded against every vocabulary word's
//! expected viseme sequence (from a phoneme->viseme map) by dynamic time
//! warping with a feature-distance substitution cost; best cost wins if it is
//! under `MAX_WORD_COST`.
//! Sentence formation: the decoded word list is aligned in order to the
//! challenge phrase by edit distance (Levenshtein); match ratio = matched
//! words / phrase length. Timing: speech onset must come after the prompt
//! ended, and mean per-word duration must be humanly plausible.
//!
//! Production path: implement `VisualSpeechRecognizer` with a real visual
//! speech recognition model (e.g. Auto-AVSR or AV-HuBERT) constrained to the
//! challenge vocabulary. Not bundled. `MeshDtwRecognizer` is a heuristic
//! baseline whose parameters are uncalibrated.

use std::collections::HashSet;

/// Viseme classes and their (openness, spread) targets in lip-mesh units
/// (openness = inner-lip area / mouth width^2, spread = inner/outer width).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Viseme { Rest, P, F, A, O, E, T }

pub const VISEMES: [Viseme; 7] = [Viseme::Rest, Viseme::P, Viseme::F, Viseme::A, Viseme::O, Viseme::E, Viseme::T];

impl Viseme {
    pub fn target(self) -> (f64, f64) {
        match self {
            Viseme::Rest => (0.03, 0.72),
            Viseme::P => (0.00, 0.55),
            Viseme::F => (0.06, 0.88),
            Viseme::A => (0.30, 0.75),
            Viseme::O => (0.18, 0.45),
            Viseme::E => (0.10, 0.95),
            Viseme::T => (0.13, 0.72),
        }
    }
}

fn vdist(a: (f64, f64), b: (f64, f64)) -> f64 { (3.0 * (a.0 - b.0)).hypot(a.1 - b.1) }

/// ARPAbet phoneme -> viseme class.
pub fn phoneme_to_viseme(p: &str) -> Viseme {
    match p {
        "P" | "B" | "M" => Viseme::P,
        "F" | "V" => Viseme::F,
        "AA" | "AE" | "AH" | "AY" | "AW" => Viseme::A,
        "OW" | "UW" | "W" | "OY" | "R" | "ER" | "UH" | "AO" => Viseme::O,
        "IY" | "IH" | "EH" | "EY" | "S" | "Z" | "CH" | "SH" | "JH" | "ZH" => Viseme::E,
        _ => Viseme::T, // T D N L K G TH DH HH Y NG
    }
}

/// Fixed challenge vocabulary with ARPAbet pronunciations.
pub const VOCAB: [(&str, &str); 14] = [
    ("blue", "B L UW"), ("five", "F AY V"), ("seven", "S EH V AH N"), ("tiger", "T AY G ER"),
    ("moon", "M UW N"), ("apple", "AE P AH L"), ("river", "R IH V ER"), ("green", "G R IY N"),
    ("map", "M AE P"), ("window", "W IH N D OW"), ("pizza", "P IY T S AH"), ("violet", "V AY AH L AH T"),
    ("coffee", "K AA F IY"), ("ocean", "OW SH AH N"),
];

/// Collapsed viseme sequence for a vocabulary word.
pub fn word_visemes(i: usize) -> Vec<Viseme> {
    let mut v: Vec<Viseme> = Vec::new();
    for p in VOCAB[i].1.split_whitespace() {
        let x = phoneme_to_viseme(p);
        if v.last() != Some(&x) { v.push(x); }
    }
    v
}

pub const MIN_WORDS: usize = 4;
pub const MAX_WORDS: usize = 6;
pub const CHALLENGE_WINDOW_MS: u64 = 8_000;
pub const MAX_WORD_COST: f64 = 0.08;
pub const MIN_MATCH_RATIO: f64 = 0.75;
pub const MIN_MS_PER_WORD: f64 = 150.0;

#[derive(Clone, Debug)]
pub struct Challenge {
    pub nonce: u64,
    pub words: Vec<usize>,
    /// Time (scan clock, ms) when the spoken/displayed prompt finished.
    pub prompt_end_ms: u64,
}

impl Challenge {
    pub fn phrase(&self) -> String {
        self.words.iter().map(|&i| VOCAB[i].0).collect::<Vec<_>>().join(" ")
    }
}

/// One frame of lip-mesh features (from `scan::build_lip_mesh`).
#[derive(Clone, Copy, Debug)]
pub struct LipFeat { pub t_ms: u64, pub openness: f64, pub spread: f64 }

/// A decoded word span (vocab index, start, end). Internal only.
#[derive(Clone, Copy, Debug)]
pub struct WordSpan { pub word: usize, pub start_ms: u64, pub end_ms: u64 }

/// Pluggable visual speech recognizer, constrained to the challenge vocabulary.
pub trait VisualSpeechRecognizer {
    /// Return the in-order word spans recognised in `feats` (None = unknown word).
    fn decode(&self, feats: &[LipFeat]) -> Vec<(Option<usize>, u64, u64)>;
}

/// Heuristic baseline: viseme segmentation + per-word DTW.
pub struct MeshDtwRecognizer;

fn dtw(a: &[Viseme], b: &[Viseme]) -> f64 {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 { return f64::INFINITY; }
    let mut d = vec![vec![f64::INFINITY; m + 1]; n + 1];
    d[0][0] = 0.0;
    for i in 1..=n { for j in 1..=m {
        let c = vdist(a[i - 1].target(), b[j - 1].target());
        d[i][j] = c + d[i - 1][j].min(d[i][j - 1]).min(d[i - 1][j - 1]);
    }}
    d[n][m] / (n.max(m)) as f64
}

pub fn snap(f: &LipFeat) -> Viseme {
    *VISEMES.iter().min_by(|a, b| vdist(a.target(), (f.openness, f.spread)).partial_cmp(&vdist(b.target(), (f.openness, f.spread))).unwrap()).unwrap()
}

impl VisualSpeechRecognizer for MeshDtwRecognizer {
    fn decode(&self, feats: &[LipFeat]) -> Vec<(Option<usize>, u64, u64)> {
        // run-length segmentation
        let mut runs: Vec<(Viseme, usize, u64, u64)> = Vec::new();
        for f in feats {
            let v = snap(f);
            match runs.last_mut() { Some(r) if r.0 == v => { r.1 += 1; r.3 = f.t_ms; } _ => runs.push((v, 1, f.t_ms, f.t_ms)) }
        }
        let runs: Vec<_> = runs.into_iter().filter(|r| r.1 >= 2).collect();
        // split into word chunks at rest pauses
        let mut out = Vec::new();
        let mut cur: Vec<(Viseme, u64, u64)> = Vec::new();
        let flush = |cur: &mut Vec<(Viseme, u64, u64)>, out: &mut Vec<(Option<usize>, u64, u64)>| {
            if cur.is_empty() { return; }
            let mut seq: Vec<Viseme> = Vec::new();
            for c in cur.iter() { if seq.last() != Some(&c.0) { seq.push(c.0); } }
            let (mut best, mut bc) = (None, f64::INFINITY);
            for w in 0..VOCAB.len() { let c = dtw(&seq, &word_visemes(w)); if c < bc { bc = c; best = Some(w); } }
            out.push((if bc <= MAX_WORD_COST { best } else { None }, cur[0].1, cur[cur.len() - 1].2));
            cur.clear();
        };
        for r in &runs {
            if r.0 == Viseme::Rest { if r.1 >= 3 { flush(&mut cur, &mut out); } } else { cur.push((r.0, r.2, r.3)); }
        }
        flush(&mut cur, &mut out);
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ChallengeCode { Matched, WrongPhrase, NonceReplay, UnknownNonce, EarlyStart, TooFast, NoSpeech }

impl ChallengeCode {
    pub fn is_veto(self) -> bool { !matches!(self, ChallengeCode::Matched) }
}

/// Only scores and codes leave the verifier — never the decoded words.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ChallengeOutcome {
    pub match_ratio: f64,
    pub llr: f64,
    pub onset_after_prompt_ms: i64,
    pub passed: bool,
    pub codes: Vec<ChallengeCode>,
}

/// Issues challenges and remembers spent nonces for the session (in memory).
pub struct ChallengeSession { active: Option<Challenge>, spent: HashSet<u64>, rng: u64 }

impl ChallengeSession {
    /// `seed` = None uses OS-randomised hashing + clock for entropy.
    pub fn new(seed: Option<u64>) -> Self {
        let rng = seed.unwrap_or_else(|| {
            use std::hash::{BuildHasher, Hasher};
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
            h.finish()
        });
        ChallengeSession { active: None, spent: HashSet::new(), rng: rng | 1 }
    }

    fn next(&mut self) -> u64 { // xorshift64*
        self.rng ^= self.rng >> 12; self.rng ^= self.rng << 25; self.rng ^= self.rng >> 27;
        self.rng.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Create a new challenge. `prompt_end_ms` is set once TTS/display ends.
    pub fn issue(&mut self, prompt_end_ms: u64) -> Challenge {
        let n = MIN_WORDS + (self.next() as usize) % (MAX_WORDS - MIN_WORDS + 1);
        let mut words = Vec::new();
        while words.len() < n { let w = (self.next() as usize) % VOCAB.len(); if !words.contains(&w) { words.push(w); } }
        let mut nonce = self.next();
        while self.spent.contains(&nonce) { nonce = self.next(); }
        let c = Challenge { nonce, words, prompt_end_ms };
        self.active = Some(c.clone());
        c
    }

    pub fn verify(&mut self, nonce: u64, feats: &[LipFeat], vsr: &dyn VisualSpeechRecognizer) -> ChallengeOutcome {
        let fail = |codes: Vec<ChallengeCode>| ChallengeOutcome { match_ratio: 0.0, llr: -10.0, onset_after_prompt_ms: 0, passed: false, codes };
        if self.spent.contains(&nonce) { return fail(vec![ChallengeCode::NonceReplay]); }
        let ch = match self.active.take() { Some(c) if c.nonce == nonce => c, other => { self.active = other; return fail(vec![ChallengeCode::UnknownNonce]); } };
        self.spent.insert(nonce); // one-time: consumed whatever the result
        let feats: Vec<LipFeat> = feats.iter().cloned().filter(|f| f.t_ms <= ch.prompt_end_ms + CHALLENGE_WINDOW_MS).collect();
        let spans = vsr.decode(&feats);
        if spans.is_empty() { return fail(vec![ChallengeCode::NoSpeech]); }
        // sentence formation: ordered edit-distance alignment
        let got: Vec<Option<usize>> = spans.iter().map(|s| s.0).collect();
        let (n, m) = (ch.words.len(), got.len());
        let mut d = vec![vec![0usize; m + 1]; n + 1];
        for i in 0..=n { d[i][0] = i; }
        for j in 0..=m { d[0][j] = j; }
        for i in 1..=n { for j in 1..=m {
            let sub = if got[j - 1] == Some(ch.words[i - 1]) { 0 } else { 1 };
            d[i][j] = (d[i - 1][j - 1] + sub).min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
        }}
        // backtrack to count exact matches
        let (mut i, mut j, mut matched) = (n, m, 0usize);
        while i > 0 && j > 0 {
            let sub = if got[j - 1] == Some(ch.words[i - 1]) { 0 } else { 1 };
            if d[i][j] == d[i - 1][j - 1] + sub { if sub == 0 { matched += 1; } i -= 1; j -= 1; }
            else if d[i][j] == d[i - 1][j] + 1 { i -= 1; } else { j -= 1; }
        }
        let ratio = matched as f64 / n as f64;
        let onset = spans[0].1 as i64 - ch.prompt_end_ms as i64;
        let dur = (spans[spans.len() - 1].2 - spans[0].1) as f64;
        let mut codes = Vec::new();
        if onset < 0 { codes.push(ChallengeCode::EarlyStart); }
        if dur / (n as f64) < MIN_MS_PER_WORD { codes.push(ChallengeCode::TooFast); }
        if ratio < MIN_MATCH_RATIO { codes.push(ChallengeCode::WrongPhrase); }
        let passed = codes.is_empty();
        if passed { codes.push(ChallengeCode::Matched); }
        ChallengeOutcome { match_ratio: ratio, llr: (12.0 * (ratio - 0.5)).clamp(-10.0, 6.0), onset_after_prompt_ms: onset, passed, codes }
    }
}

/// Add the challenge channel to a scan result as an extra LLR + vetoes.
pub fn apply_to_scan(r: &mut crate::scan::ScanResult, o: &ChallengeOutcome, cal: &crate::scan::Calibration) {
    use crate::scan::{llr_accept_threshold, ReasonCode, ScanStatus, Tier, HIGH_ASSURANCE_THRESHOLD};
    if r.status == ScanStatus::Inconclusive { return; }
    r.evidence.total_llr += o.llr;
    r.posterior = cal.posterior(r.evidence.total_llr);
    if !o.passed { r.vetoes.push(ReasonCode::ChallengeFailed); r.reasons.push(ReasonCode::ChallengeFailed); }
    r.legacy_pass &= o.passed;
    r.high_assurance_pass = r.vetoes.is_empty() && r.posterior >= HIGH_ASSURANCE_THRESHOLD && r.evidence.total_llr >= llr_accept_threshold()
        && !(r.reasons.contains(&ReasonCode::NoOcularMotion) && r.distance_hint == "close");
    let pass = match r.tier { Tier::Legacy => r.legacy_pass, Tier::HighAssurance => r.high_assurance_pass };
    r.reasons.retain(|c| *c != ReasonCode::Passed && *c != ReasonCode::BelowHighAssurance);
    if !r.high_assurance_pass { r.reasons.push(ReasonCode::BelowHighAssurance); }
    if pass { r.reasons.push(ReasonCode::Passed); }
    r.status = if pass { ScanStatus::Pass } else { ScanStatus::Fail };
}

/// Synthetic lip features for a word sequence (tests only).
pub fn synth_response(words: &[usize], start_ms: u64, frames_per_viseme: usize, seed: u64) -> Vec<LipFeat> {
    let mut s = seed | 1;
    let mut noise = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.01 };
    let mut out = Vec::new();
    let mut t = start_ms;
    let push = |v: Viseme, k: usize, out: &mut Vec<LipFeat>, t: &mut u64, noise: &mut dyn FnMut() -> f64| {
        for _ in 0..k { let (o, p) = v.target(); out.push(LipFeat { t_ms: *t, openness: (o + noise()).max(0.0), spread: p + noise() }); *t += 33; }
    };
    for &w in words {
        for v in word_visemes(w) { push(v, frames_per_viseme, &mut out, &mut t, &mut noise); }
        push(Viseme::Rest, 5, &mut out, &mut t, &mut noise);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_sequences_are_distinct() {
        let all: Vec<_> = (0..VOCAB.len()).map(word_visemes).collect();
        for i in 0..all.len() { for j in i + 1..all.len() { assert_ne!(all[i], all[j], "{} vs {}", VOCAB[i].0, VOCAB[j].0); } }
    }

    #[test]
    fn issue_is_random_and_sized() {
        let mut s = ChallengeSession::new(Some(42));
        let a = s.issue(1000); let b = s.issue(1000);
        assert!((MIN_WORDS..=MAX_WORDS).contains(&a.words.len()));
        assert_ne!(a.nonce, b.nonce);
        assert!(!a.phrase().is_empty());
    }

    #[test]
    fn correct_phrase_passes() {
        let mut s = ChallengeSession::new(Some(1));
        let c = s.issue(2000);
        let o = s.verify(c.nonce, &synth_response(&c.words, 2400, 4, 9), &MeshDtwRecognizer);
        assert!(o.passed, "{:?}", o);
        assert_eq!(o.match_ratio, 1.0);
        assert!(o.llr > 0.0);
    }

    #[test]
    fn wrong_phrase_fails() {
        let mut s = ChallengeSession::new(Some(2));
        let c = s.issue(2000);
        let wrong: Vec<usize> = (0..VOCAB.len()).filter(|w| !c.words.contains(w)).take(c.words.len()).collect();
        let o = s.verify(c.nonce, &synth_response(&wrong, 2400, 4, 9), &MeshDtwRecognizer);
        assert!(!o.passed && o.codes.contains(&ChallengeCode::WrongPhrase), "{:?}", o);
    }

    #[test]
    fn replay_of_old_phrase_fails() {
        let mut s = ChallengeSession::new(Some(3));
        let old = s.issue(2000);
        let rec = synth_response(&old.words, 2400, 4, 9);
        assert!(s.verify(old.nonce, &rec, &MeshDtwRecognizer).passed);
        // same nonce again
        assert_eq!(s.verify(old.nonce, &rec, &MeshDtwRecognizer).codes, vec![ChallengeCode::NonceReplay]);
        // new challenge answered with the old recording
        let new = s.issue(2000);
        let o = s.verify(new.nonce, &rec, &MeshDtwRecognizer);
        assert!(!o.passed, "{:?} new={:?} old={:?}", o, new.words, old.words);
    }

    #[test]
    fn early_start_fails() {
        let mut s = ChallengeSession::new(Some(4));
        let c = s.issue(3000);
        let o = s.verify(c.nonce, &synth_response(&c.words, 1500, 4, 9), &MeshDtwRecognizer);
        assert!(o.codes.contains(&ChallengeCode::EarlyStart), "{:?}", o);
    }

    #[test]
    fn too_fast_fails() {
        let mut s = ChallengeSession::new(Some(5));
        let c = s.issue(1000);
        // squeeze timestamps: 5 ms per frame
        let mut f = synth_response(&c.words, 1200, 4, 9);
        for (i, x) in f.iter_mut().enumerate() { x.t_ms = 1200 + 5 * i as u64; }
        let o = s.verify(c.nonce, &f, &MeshDtwRecognizer);
        assert!(o.codes.contains(&ChallengeCode::TooFast), "{:?}", o);
    }

    #[test]
    fn challenge_feeds_scan_fusion() {
        use crate::scan::{scan_window, synth::{sequence, Kind}, Calibration, ScanStatus, ReasonCode};
        let cal = Calibration::default();
        let mut s = ChallengeSession::new(Some(6));
        let c = s.issue(1000);
        let ok = s.verify(c.nonce, &synth_response(&c.words, 1500, 4, 1), &MeshDtwRecognizer);
        let mut r = scan_window(true, sequence(Kind::Live, 150, 33, 7));
        let before = r.evidence.total_llr;
        apply_to_scan(&mut r, &ok, &cal);
        assert_eq!(r.status, ScanStatus::Pass);
        assert!(r.evidence.total_llr > before);
        let c2 = s.issue(1000);
        let bad = s.verify(c2.nonce, &synth_response(&c2.words, 200, 4, 1), &MeshDtwRecognizer);
        let mut r2 = scan_window(true, sequence(Kind::Live, 150, 33, 7));
        apply_to_scan(&mut r2, &bad, &cal);
        assert_eq!(r2.status, ScanStatus::Fail);
        assert!(r2.vetoes.contains(&ReasonCode::ChallengeFailed));
    }
}
