/**
 * Weight estimation utilities for models and speculative decoding draft models.
 * Used for estimating weight footprint vs VRAM available and detecting system RAM spill.
 * Note: Context (KV cache) is explicitly excluded from these weight calculations.
 */

export function parseParamsFromName(name: string): number | null {
  const base = name.split(/[/\\]/).pop() || name;
  const match = base.match(/(?:^|[-_ /])(\d+(?:\.\d+)?)[bB](?:[-_ /.]|$)/i);
  return match ? parseFloat(match[1]) : null;
}

export function bytesPerParam(quant?: string): number {
  if (!quant) return 2.0;
  const q = quant.toLowerCase();
  if (q.startsWith("pq") || q.startsWith("ptq")) return 0.28;
  if (
    q.startsWith("q4") ||
    q.includes("q4_") ||
    q.includes("iq4") ||
    q === "awq" ||
    q === "gptq" ||
    q === "int4"
  )
    return 0.55;
  if (q.startsWith("q8") || q.includes("q8_")) return 1.05;
  if (q.startsWith("q5") || q.includes("q5_")) return 0.68;
  if (q.startsWith("q6") || q.includes("q6_")) return 0.80;
  if (q.startsWith("q3") || q.includes("q3_") || q.includes("iq3")) return 0.45;
  if (q.startsWith("q2") || q.includes("q2_") || q.includes("iq2")) return 0.35;
  if (q.startsWith("q1") || q.includes("q1_")) return 0.25;
  if (q === "fp8" || q === "int8") return 1.0;
  if (q === "gguf") return 0.55;
  return 2.0; // fp16 / bf16 / unset
}

export function estimateWeightGb(paramsB: number, quant?: string): number {
  return paramsB * bytesPerParam(quant);
}

export interface ServerWeightBreakdown {
  mainWeightGb: number;
  specWeightGb: number;
  totalWeightGb: number;
}

export function estimateServerWeights(
  def: {
    model_id?: string;
    model_path?: string | null;
    params_b?: number | null;
    quant?: string;
    backend?: string;
    speculative_enabled?: boolean;
    speculative_model?: string | null;
    speculative_model_path?: string | null;
  },
  libraryModels?: {
    model_id?: string;
    model_path?: string | null;
    size_mb?: number;
    params_b?: number | null;
  }[]
): ServerWeightBreakdown {
  let mainWeightGb = 0;
  if (def.params_b && def.params_b > 0) {
    mainWeightGb = estimateWeightGb(def.params_b, def.quant);
  } else if (libraryModels) {
    const match = libraryModels.find(
      (m) =>
        (def.model_path && m.model_path === def.model_path) ||
        (def.model_id && m.model_id === def.model_id)
    );
    if (match && match.size_mb) {
      mainWeightGb = match.size_mb / 1024;
    } else if (match && match.params_b) {
      mainWeightGb = estimateWeightGb(match.params_b, def.quant);
    }
  }
  if (mainWeightGb === 0 && def.model_id) {
    const p = parseParamsFromName(def.model_id);
    if (p) mainWeightGb = estimateWeightGb(p, def.quant);
  }
  if (mainWeightGb === 0 && def.model_path) {
    const p = parseParamsFromName(def.model_path);
    if (p) mainWeightGb = estimateWeightGb(p, def.quant || "gguf");
  }

  let specWeightGb = 0;
  if (def.speculative_enabled) {
    const draftName = def.speculative_model_path || def.speculative_model || "";
    if (libraryModels && draftName) {
      const match = libraryModels.find(
        (m) =>
          (m.model_path && m.model_path === draftName) ||
          (m.model_id && m.model_id === draftName)
      );
      if (match && match.size_mb) {
        specWeightGb = match.size_mb / 1024;
      } else if (match && match.params_b) {
        specWeightGb = estimateWeightGb(
          match.params_b,
          def.backend === "llamacpp" ? "gguf" : "fp16"
        );
      }
    }
    if (specWeightGb === 0 && draftName) {
      const p = parseParamsFromName(draftName);
      if (p) {
        specWeightGb = estimateWeightGb(
          p,
          def.backend === "llamacpp" ? "gguf" : "fp16"
        );
      }
    }
  }

  return {
    mainWeightGb,
    specWeightGb,
    totalWeightGb: mainWeightGb + specWeightGb,
  };
}

export function calculateSystemRamSpill(
  weightGb: number,
  freeVramGb: number
): { loadedIntoRamGb: number; fitsInVram: boolean } {
  const loadedIntoRamGb = Math.max(0, weightGb - freeVramGb);
  const fitsInVram = loadedIntoRamGb <= 0.001;
  return { loadedIntoRamGb, fitsInVram };
}
