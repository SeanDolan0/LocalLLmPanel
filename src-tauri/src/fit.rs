//! Hardware-fit scoring for quant variants of a model.
//!
//! Pure functions composing `estimate.rs`. No I/O, fully unit-testable.

use serde::Serialize;
use crate::estimate;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum FitVerdict { Comfortable, Constrained, DoesNotFit }

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum FormatSupport { Native, Experimental }

pub const COMFORTABLE_MAX_RATIO: f64 = 0.60;
pub const CONSTRAINED_MAX_RATIO: f64 = 0.95;
pub const OVERHEAD_MB: f64 = 2500.0;
pub const GPU_UTIL_DEFAULT: f64 = 0.92;

#[derive(Debug, Clone, Serialize)]
pub struct HardwareProfile {
    pub gpu_name: String,
    pub vram_total_mb: u64,
    pub bandwidth_gbs: f64,
    pub bandwidth_known: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelArchInfo {
    pub params_b: Option<f64>,
    pub context: usize,
    pub n_layers: Option<usize>,
    pub n_kv_heads: Option<usize>,
    pub head_dim: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FitResult {
    pub verdict: FitVerdict,
    pub score: u8,
    pub weight_gb: f64,
    pub usable_context: usize,
    pub native_context: usize,
    pub est_tok_s: Option<f64>,
    pub measured_tok_s: Option<f64>,
    pub vram_pct: u8,
    pub format_support: FormatSupport,
    pub reason: String,
}

/// Minimal QuantVariant for scoring (full struct lives in hf.rs, this
/// is the subset fit.rs needs — keeps fit.rs free of HF-API concerns).
#[derive(Debug, Clone, Serialize)]
pub struct VariantInput {
    pub quant_str: String,       // "fp16", "fp8", "awq", "gptq", "bnb", "gguf"
    pub weight_bytes: Option<u64>,
    pub params_b: Option<f64>,   // override if different from base model
    pub is_gguf: bool,
}

pub fn score_variant(
    hw: &HardwareProfile,
    variant: &VariantInput,
    arch: &ModelArchInfo,
    measured_tok_s: Option<f64>,
) -> FitResult {
    let params_b = variant.params_b.or(arch.params_b).unwrap_or(0.0);
    let format_support = if variant.is_gguf {
        FormatSupport::Experimental
    } else {
        FormatSupport::Native
    };

    // Weight in GB: exact from file size when available (bytes to GB / 1024^3), else estimate
    let weight_gb = if let Some(wb) = variant.weight_bytes {
        wb as f64 / (1024.0 * 1024.0 * 1024.0)
    } else {
        estimate::weight_gb(params_b, &variant.quant_str)
    };

    // VRAM ratio
    let total_need_mb = weight_gb * 1024.0 + OVERHEAD_MB;
    let usable_vram_mb = hw.vram_total_mb as f64;
    let vram_ratio = total_need_mb / usable_vram_mb;
    let vram_pct = (vram_ratio * 100.0).round().min(255.0) as u8;

    // Verdict
    let verdict = if vram_ratio <= COMFORTABLE_MAX_RATIO {
        FitVerdict::Comfortable
    } else if vram_ratio <= CONSTRAINED_MAX_RATIO {
        FitVerdict::Constrained
    } else {
        FitVerdict::DoesNotFit
    };

    // Usable context
    let kv_bpt = match (arch.n_layers, arch.n_kv_heads, arch.head_dim) {
        (Some(l), Some(k), Some(h)) => estimate::kv_bytes_per_token(l, k, h),
        _ => estimate::estimate_kv_bytes_per_token(params_b),
    };
    let usable_context = if kv_bpt > 0.0 && (params_b > 0.0 || variant.weight_bytes.is_some()) {
        let ctx = estimate::context_fit_with_weight(
            usable_vram_mb,
            GPU_UTIL_DEFAULT,
            weight_gb,
            kv_bpt,
            OVERHEAD_MB,
        );
        ctx.min(arch.context)
    } else {
        0
    };

    // Tok/s estimate
    let est_tok_s = if hw.bandwidth_gbs > 0.0 {
        if weight_gb > 0.0 {
            Some(hw.bandwidth_gbs * 0.5 / weight_gb)
        } else if params_b > 0.0 {
            Some(estimate::tokens_per_sec(hw.bandwidth_gbs, params_b, &variant.quant_str))
        } else {
            None
        }
    } else {
        None
    };

    let speed_tok_s = measured_tok_s.or(est_tok_s);

    // Composite score: fit (40%), speed (30%), context utilization (30%)
    let fit_pillar = match verdict {
        FitVerdict::Comfortable => 90.0 + 10.0 * (1.0 - vram_ratio / COMFORTABLE_MAX_RATIO),
        FitVerdict::Constrained => {
            let range = CONSTRAINED_MAX_RATIO - COMFORTABLE_MAX_RATIO;
            let pos = (vram_ratio - COMFORTABLE_MAX_RATIO) / range;
            85.0 - pos * 55.0 // 85 → 30
        }
        FitVerdict::DoesNotFit => 0.0,
    };
    let speed_pillar = speed_tok_s
        .map(|t| (t / 80.0 * 100.0).min(100.0).max(0.0))
        .unwrap_or(0.0);
    let ctx_pillar = if arch.context > 0 && usable_context > 0 {
        (usable_context as f64 / arch.context as f64 * 100.0).min(100.0)
    } else {
        0.0
    };

    let mut score = (fit_pillar * 0.4 + speed_pillar * 0.3 + ctx_pillar * 0.3).round() as u8;
    if verdict == FitVerdict::DoesNotFit {
        score = score.min(25);
    }

    let (ctx_room, ctx_limit) = if usable_context < 1000 {
        (
            format!("{} tokens", usable_context),
            format!("{} tokens", usable_context),
        )
    } else {
        (
            format!("{}k", usable_context / 1000),
            format!("{}k tokens", usable_context / 1000),
        )
    };

    let reason = match verdict {
        FitVerdict::Comfortable => format!(
            "Comfortable fit ({vram_pct}% VRAM). Room for {ctx_room} context and fast generation."
        ),
        FitVerdict::Constrained => format!(
            "Constrained fit ({vram_pct}% VRAM). Usable context limited to ~{ctx_limit}."
        ),
        FitVerdict::DoesNotFit => format!(
            "Does not fit — needs ~{:.1} GB but GPU has {:.1} GB VRAM.",
            weight_gb + OVERHEAD_MB / 1024.0,
            hw.vram_total_mb as f64 / 1024.0
        ),
    };

    FitResult {
        verdict,
        score,
        weight_gb,
        usable_context,
        native_context: arch.context,
        est_tok_s,
        measured_tok_s,
        vram_pct,
        format_support,
        reason,
    }
}

pub fn compare_variant_fit(a: (&VariantInput, &FitResult), b: (&VariantInput, &FitResult)) -> std::cmp::Ordering {
    b.1.score.cmp(&a.1.score).then_with(|| {
        let a_native = !a.0.is_gguf;
        let b_native = !b.0.is_gguf;
        b_native.cmp(&a_native)
    })
}

pub fn rank_variants(results: &mut Vec<(VariantInput, FitResult)>) {
    results.sort_by(|a, b| compare_variant_fit((&a.0, &a.1), (&b.0, &b.1)));
}

pub fn best_variant(
    results: &[(VariantInput, FitResult)],
    preferred_format: Option<&str>,
) -> usize {
    if results.is_empty() {
        return 0;
    }
    if let Some(pref) = preferred_format {
        // If preferred format has a "comfortable" or "constrained" variant, pick it
        if let Some(idx) = results.iter().position(|(v, r)| {
            v.quant_str.eq_ignore_ascii_case(pref) && r.verdict != FitVerdict::DoesNotFit
        }) {
            return idx;
        }
    }
    0 // Already sorted by rank_variants, first is best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hw_12gb() -> HardwareProfile {
        HardwareProfile {
            gpu_name: "NVIDIA GeForce RTX 5070 Ti Laptop GPU".into(),
            vram_total_mb: 12227,
            bandwidth_gbs: 672.0,
            bandwidth_known: true,
        }
    }

    fn arch_0_5b() -> ModelArchInfo {
        ModelArchInfo { params_b: Some(0.494), context: 32768, n_layers: Some(24), n_kv_heads: Some(2), head_dim: Some(64) }
    }

    fn arch_7b() -> ModelArchInfo {
        ModelArchInfo { params_b: Some(7.6), context: 32768, n_layers: Some(32), n_kv_heads: Some(8), head_dim: Some(128) }
    }

    fn arch_14b() -> ModelArchInfo {
        ModelArchInfo { params_b: Some(14.7), context: 32768, n_layers: Some(40), n_kv_heads: Some(8), head_dim: Some(128) }
    }

    fn variant(quant: &str, is_gguf: bool) -> VariantInput {
        VariantInput { quant_str: quant.into(), weight_bytes: None, params_b: None, is_gguf }
    }

    fn variant_gguf_sized(quant: &str, size_bytes: u64) -> VariantInput {
        VariantInput { quant_str: quant.into(), weight_bytes: Some(size_bytes), params_b: None, is_gguf: true }
    }

    #[test]
    fn test_comfortable_small_model() {
        let r = score_variant(&hw_12gb(), &variant("fp16", false), &arch_0_5b(), None);
        assert_eq!(r.verdict, FitVerdict::Comfortable);
        assert!(r.score > 70, "score = {}", r.score);
        assert_eq!(r.format_support, FormatSupport::Native);
        assert!(r.usable_context > 0);
    }

    #[test]
    fn test_doesnt_fit_7b_fp16() {
        // 7.6B fp16 = ~15.2 GB weights. On 12 GB GPU → DoesNotFit.
        let r = score_variant(&hw_12gb(), &variant("fp16", false), &arch_7b(), None);
        assert_eq!(r.verdict, FitVerdict::DoesNotFit);
    }

    #[test]
    fn test_constrained_7b_awq() {
        // 7.6B AWQ = ~8.36 GB weights + 2.5 overhead = ~10.86 GB.
        // VRAM ratio = 10.86 / 12.227 = 0.888 → Constrained (0.60 < ratio ≤ 0.95)
        let r = score_variant(&hw_12gb(), &variant("awq", false), &arch_7b(), None);
        assert_eq!(r.verdict, FitVerdict::Constrained);
        assert!(r.score > 30 && r.score < 80, "score = {}", r.score);
    }

    #[test]
    fn test_doesnt_fit_14b_fp16() {
        let r = score_variant(&hw_12gb(), &variant("fp16", false), &arch_14b(), None);
        assert_eq!(r.verdict, FitVerdict::DoesNotFit);
        assert!(r.score <= 25, "score = {} (should be crushed)", r.score);
    }

    #[test]
    fn test_gguf_uses_exact_weight() {
        // GGUF file of 4.5 GB = 4_831_838_208 bytes
        let r = score_variant(&hw_12gb(), &variant_gguf_sized("gguf", 4_831_838_208), &arch_7b(), None);
        // 4.5 GB + 2.5 GB overhead = 7 GB → 7/12.227 = 0.572 → Comfortable
        assert_eq!(r.verdict, FitVerdict::Comfortable);
        assert_eq!(r.format_support, FormatSupport::Experimental);
        assert!(r.usable_context > 0, "GGUF comfortable fit must have positive usable context");
    }

    #[test]
    fn test_gguf_usable_context_non_zero() {
        let r = score_variant(
            &hw_12gb(),
            &variant_gguf_sized("gguf", 4_831_838_208),
            &arch_7b(),
            None,
        );
        assert_eq!(r.verdict, FitVerdict::Comfortable);
        assert!(r.usable_context > 0);
    }

    #[test]
    fn test_context_display_sub_1000() {
        // Weight 8.45 GB leaves ~96 MB KV cache for 7B arch (bpt=131072) -> ~768 tokens
        let r = score_variant(
            &hw_12gb(),
            &variant_gguf_sized("gguf", 9_073_000_000), // ~8.45 GB
            &arch_7b(),
            None,
        );
        assert_eq!(r.verdict, FitVerdict::Constrained);
        assert!(r.usable_context < 1000);
        assert!(r.usable_context >= 512);
        assert!(r.reason.contains(&format!("{} tokens", r.usable_context)));
        assert!(!r.reason.contains("0k"));
    }

    #[test]
    fn test_measured_stats_preferred() {
        let hw = hw_12gb();
        let v = variant("awq", false);
        let arch = arch_7b();

        let r_est = score_variant(&hw, &v, &arch, None);
        let r_faster = score_variant(&hw, &v, &arch, Some(80.0));
        let r_slower = score_variant(&hw, &v, &arch, Some(10.0));

        assert_eq!(r_faster.measured_tok_s, Some(80.0));
        assert!(r_faster.est_tok_s.is_some());
        assert!(
            r_faster.score > r_est.score,
            "higher measured tok/s should yield a higher score ({} > {})",
            r_faster.score,
            r_est.score
        );
        assert!(
            r_est.score > r_slower.score,
            "lower measured tok/s should yield a lower score ({} > {})",
            r_est.score,
            r_slower.score
        );
    }

    #[test]
    fn test_rank_variants_native_before_gguf() {
        let hw = hw_12gb();
        let arch = arch_0_5b();
        let v_fp16 = variant("fp16", false);
        let v_gguf = variant_gguf_sized("gguf", 500_000_000);
        let r_fp16 = score_variant(&hw, &v_fp16, &arch, None);
        let r_gguf = score_variant(&hw, &v_gguf, &arch, None);
        let mut results = vec![(v_gguf, r_gguf), (v_fp16, r_fp16)];
        rank_variants(&mut results);
        assert!(!results[0].0.is_gguf, "Native should rank first");
    }

    #[test]
    fn test_compare_variant_fit() {
        let v_native = variant("fp16", false);
        let v_gguf = variant("gguf", true);
        let mut r1 = score_variant(&hw_12gb(), &v_native, &arch_0_5b(), None);
        let mut r2 = score_variant(&hw_12gb(), &v_gguf, &arch_0_5b(), None);

        // Higher score comes first
        r1.score = 90;
        r2.score = 80;
        assert_eq!(
            compare_variant_fit((&v_native, &r1), (&v_gguf, &r2)),
            std::cmp::Ordering::Less
        );

        // Equal score: native before gguf
        r2.score = 90;
        assert_eq!(
            compare_variant_fit((&v_native, &r1), (&v_gguf, &r2)),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_variant_fit((&v_gguf, &r2), (&v_native, &r1)),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn test_best_variant_preferred_format() {
        let hw = hw_12gb();
        let arch = arch_0_5b();
        let v_fp16 = variant("fp16", false);
        let v_fp8 = variant("fp8", false);
        let r_fp16 = score_variant(&hw, &v_fp16, &arch, None);
        let r_fp8 = score_variant(&hw, &v_fp8, &arch, None);
        let results = vec![(v_fp16, r_fp16), (v_fp8, r_fp8)];
        let idx = best_variant(&results, None);
        assert!(idx < results.len());
        let idx_pref = best_variant(&results, Some("fp8"));
        assert_eq!(results[idx_pref].0.quant_str, "fp8");
    }

    #[test]
    fn test_gguf_usable_context_with_missing_dims() {
        let hw = HardwareProfile {
            gpu_name: "RTX 4090".into(),
            vram_total_mb: 24576,
            bandwidth_gbs: 1008.0,
            bandwidth_known: true,
        };
        // Arch with missing layer dims but known params
        let arch = ModelArchInfo {
            params_b: Some(27.0),
            context: 131072,
            n_layers: None,
            n_kv_heads: None,
            head_dim: None,
        };
        let v_gguf = VariantInput {
            quant_str: "q4_k_m".into(),
            weight_bytes: Some(16_000_000_000), // ~14.9 GB
            params_b: Some(27.0),
            is_gguf: true,
        };
        let res = score_variant(&hw, &v_gguf, &arch, None);
        assert!(res.usable_context > 0, "usable context should not be 0");
        assert!(res.est_tok_s.is_some(), "est speed should be present");
        assert!(res.est_tok_s.unwrap() > 10.0, "est speed should be reasonable");
    }
}
