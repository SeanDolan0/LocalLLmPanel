//! Recommendation engine:
//! - Multi-factor composite scoring (Hardware Fit 40%, Task 25%, Quality 20%, Freshness 10%, Popularity 5%)
//! - Natural language recommendation explanations
//! - Multi-feed discovery across categories
//! - Offline fallback integration

use serde::{Deserialize, Serialize};
use crate::fit::{self, FitResult};
use crate::hf_discovery::{DiscoveryCategory, HfModelRecord, PublisherRegistry, PublisherTier};
use crate::model_intelligence::ModelFamilyInfo;

/// Detailed explanation of why a model is recommended and any relevant trade-offs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecommendationExplanation {
    pub headline: String,
    pub highlights: Vec<String>,
    pub trade_offs: Vec<String>,
}

/// Composite score breakdown.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct CompositeScoreDetails {
    pub total_score: f64,
    pub hardware_fit_score: f64, // 0 - 100
    pub task_relevance_score: f64, // 0 - 100
    pub quality_score: f64, // 0 - 100
    pub freshness_score: f64, // 0 - 100
    pub popularity_score: f64, // 0 - 100
}

/// Compute composite score for a model candidate.
pub fn calculate_composite_score(
    best_fit: &FitResult,
    record: &HfModelRecord,
    category: &DiscoveryCategory,
    family: &ModelFamilyInfo,
) -> (f64, CompositeScoreDetails, RecommendationExplanation) {
    // 1. Hardware Fit Score (0-100)
    let hw_score = match best_fit.verdict {
        fit::FitVerdict::Comfortable => {
            // 90 to 100 based on VRAM headroom
            90.0 + (100.0 - best_fit.vram_pct as f64).clamp(0.0, 10.0)
        }
        fit::FitVerdict::Constrained => {
            // 50 to 85 depending on run mode
            match best_fit.run_mode {
                fit::RunMode::Gpu => 80.0,
                fit::RunMode::GpuRamSwap => 65.0,
                fit::RunMode::CpuOffload => 50.0,
                fit::RunMode::DoesNotFit => 0.0,
            }
        }
        fit::FitVerdict::DoesNotFit => 0.0,
    };

    // 2. Task Relevance Score (0-100)
    let task_score = match category {
        DiscoveryCategory::ForYou => {
            if family.is_embedding { 40.0 } else { 90.0 }
        }
        DiscoveryCategory::NewTrending => 90.0,
        DiscoveryCategory::Coding => {
            if family.is_coding { 100.0 } else { 30.0 }
        }
        DiscoveryCategory::Reasoning => {
            if family.is_reasoning { 100.0 } else { 35.0 }
        }
        DiscoveryCategory::GeneralChat => {
            if !family.is_coding && !family.is_embedding && !family.is_reasoning {
                95.0
            } else {
                80.0
            }
        }
        DiscoveryCategory::SmallFast => {
            if let Some(params) = &family.parameter_class {
                if params == "0.5B" || params == "1.5B" || params == "3B" || params == "4B" {
                    100.0
                } else if params == "7B" || params == "8B" {
                    75.0
                } else {
                    30.0
                }
            } else {
                70.0
            }
        }
        DiscoveryCategory::Embeddings => {
            if family.is_embedding { 100.0 } else { 10.0 }
        }
        DiscoveryCategory::Vision => {
            if family.is_multimodal { 100.0 } else { 15.0 }
        }
    };

    // 3. Quality and Reliability Score (0-100)
    let pub_tier = PublisherRegistry::get_tier(&record.id);
    let quality_score = match pub_tier {
        PublisherTier::OfficialOrg => 100.0,
        PublisherTier::TrustedQuantizer => 90.0,
        PublisherTier::Community => {
            if record.likes > 50 || record.downloads > 5000 {
                75.0
            } else {
                60.0
            }
        }
    };

    // 4. Freshness Score (0-100)
    let mut freshness_score = 50.0;
    if record.trending_score > 0.0 {
        // Logarithmic trending boost up to 40 pts
        freshness_score += (record.trending_score.ln_1p() * 8.0).clamp(0.0, 40.0);
    }
    if let Some(created) = &record.created_at {
        // If created in 2025/2026, award high freshness
        if created.contains("2026") || created.contains("2025") {
            freshness_score = (freshness_score + 20.0).min(100.0);
        }
    }

    // 5. Popularity Score (0-100)
    let dl_log = (record.downloads as f64).ln_1p();
    let likes_log = (record.likes as f64).ln_1p();
    let pop_score = ((dl_log * 5.0) + (likes_log * 10.0)).clamp(0.0, 100.0);

    // Weighted composite calculation:
    // Hardware Fit: 40%
    // Task Relevance: 25%
    // Quality & Reliability: 20%
    // Freshness: 10%
    // Popularity: 5%
    let total_score = if best_fit.verdict == fit::FitVerdict::DoesNotFit {
        0.0
    } else {
        (hw_score * 0.40)
            + (task_score * 0.25)
            + (quality_score * 0.20)
            + (freshness_score * 0.10)
            + (pop_score * 0.05)
    };

    let details = CompositeScoreDetails {
        total_score: (total_score * 10.0).round() / 10.0,
        hardware_fit_score: (hw_score * 10.0).round() / 10.0,
        task_relevance_score: (task_score * 10.0).round() / 10.0,
        quality_score: (quality_score * 10.0).round() / 10.0,
        freshness_score: (freshness_score * 10.0).round() / 10.0,
        popularity_score: (pop_score * 10.0).round() / 10.0,
    };

    // Generate natural explanation
    let mut highlights = Vec::new();
    let mut trade_offs = Vec::new();

    match best_fit.run_mode {
        fit::RunMode::Gpu => {
            highlights.push(format!("Fits 100% in GPU VRAM (using ~{}% VRAM)", best_fit.vram_pct));
        }
        fit::RunMode::GpuRamSwap => {
            highlights.push("GPU accelerated with system RAM swap for extended context".into());
            trade_offs.push(format!("Requires ~{} GB system RAM swap for full context", best_fit.swap_space_gb));
        }
        fit::RunMode::CpuOffload => {
            highlights.push("Runs with partial CPU offload".into());
            trade_offs.push(format!("Requires ~{} GB weight offload to CPU RAM (lower tok/s)", best_fit.cpu_offload_gb));
        }
        fit::RunMode::DoesNotFit => {
            trade_offs.push("Model exceeds available GPU and system memory".into());
        }
    }

    if let Some(tps) = best_fit.est_tok_s {
        if tps > 0.0 {
            highlights.push(format!("Estimated ~{:.1} tok/s generation throughput", tps));
        }
    }

    if pub_tier == PublisherTier::OfficialOrg {
        highlights.push("Official release from model creator".into());
    } else if pub_tier == PublisherTier::TrustedQuantizer {
        highlights.push("Verified quantization from trusted publisher".into());
    }

    if family.is_coding && *category == DiscoveryCategory::Coding {
        highlights.push("Specialized architecture for code generation and agentic tool use".into());
    } else if family.is_reasoning && *category == DiscoveryCategory::Reasoning {
        highlights.push("Trained for multi-step reasoning and mathematical verification".into());
    }

    let headline = format!(
        "Recommended for {} ({:.0}/100 fit score)",
        category.display_name(),
        details.total_score
    );

    let explanation = RecommendationExplanation {
        headline,
        highlights,
        trade_offs,
    };

    (total_score, details, explanation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_score_calculation() {
        let fit = FitResult {
            score: 95,
            verdict: fit::FitVerdict::Comfortable,
            vram_pct: 70,
            usable_context: 32768,
            native_context: 32768,
            est_tok_s: Some(65.0),
            measured_tok_s: None,
            weight_gb: 5.5,
            run_mode: fit::RunMode::Gpu,
            vram_context: 32768,
            extended_context: 32768,
            swap_space_gb: 0,
            cpu_offload_gb: 0,
            ram_pct: 0,
            format_support: fit::FormatSupport::Native,
            runtime: Some("llama.cpp (CUDA)".into()),
            score_components: None,
            notes: vec![],
            reason: "Comfortable fit in VRAM".into(),
        };

        let record = HfModelRecord {
            id: "Qwen/Qwen2.5-Coder-7B-Instruct".to_string(),
            author: Some("Qwen".into()),
            downloads: 250_000,
            downloads_all_time: Some(500_000),
            likes: 1200,
            trending_score: 85.0,
            created_at: Some("2025-10-01".into()),
            last_modified: Some("2026-01-15".into()),
            private: false,
            gated: None,
            pipeline_tag: Some("text-generation".into()),
            tags: vec!["code".into()],
            library_name: Some("transformers".into()),
            sha: None,
            siblings: None,
        };

        let family = ModelFamilyInfo {
            family_name: "Qwen2.5".into(),
            base_model_id: None,
            parameter_class: Some("7B".into()),
            is_multimodal: false,
            is_reasoning: false,
            is_coding: true,
            is_embedding: false,
        };

        let (score, details, explanation) = calculate_composite_score(
            &fit,
            &record,
            &DiscoveryCategory::Coding,
            &family,
        );

        assert!(score > 85.0);
        assert_eq!(details.quality_score, 100.0);
        assert_eq!(details.task_relevance_score, 100.0);
        assert!(!explanation.highlights.is_empty());
    }
}
