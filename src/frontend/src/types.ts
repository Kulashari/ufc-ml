export interface PredictionRequest {
  fighter_a: string;
  fighter_b: string;
  division?: string;
}

export interface Fighter {
  fighter_id: string;
  fighter_name: string;
  display_name?: string | null;
}

export interface PredictionWarning {
  code: string;
  severity: "info" | "warning" | "critical" | string;
  message: string;
  details?: Record<string, unknown>;
}

export interface ConfidenceAssessment {
  tier: string;
  score: number;
  orientation_disagreement: number;
  warnings: PredictionWarning[];
}

export interface PredictionResponse {
  fighter_a: Fighter;
  fighter_b: Fighter;
  predicted_at: string;
  model_cutoff?: string | null;
  dataset_cutoff?: string | null;
  division?: string | null;
  probability_a: number;
  probability_b: number;
  prior_ufc_fights_a: number;
  prior_ufc_fights_b: number;
  snapshot_date_a?: string | null;
  snapshot_date_b?: string | null;
  predicted_winner_id?: string | null;
  predicted_winner_name?: string | null;
  is_even_probability?: boolean;
  orientation_disagreement?: number;
  confidence_tier?: string;
  confidence?: ConfidenceAssessment;
  warnings?: PredictionWarning[];
}
