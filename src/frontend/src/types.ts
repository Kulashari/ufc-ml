export interface PredictionRequest {
  fighter_a: string;
  fighter_b: string;
  prediction_date: string;
  division?: string;
}

export interface Fighter {
  fighter_id: string;
  fighter_name: string;
  display_name?: string | null;
  division?: string | null;
  dob?: string | null;
  as_of_date?: string | null;
  aliases?: string[];
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

export interface ArtifactSummary {
  artifact_version?: string;
  created_at_utc?: string;
  cutoff_date?: string;
  feature_count?: number;
}

export interface PredictionResponse {
  fighter_a: Fighter;
  fighter_b: Fighter;
  prediction_date: string;
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
  artifact?: ArtifactSummary;
}
