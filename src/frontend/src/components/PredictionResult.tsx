import {
  fighterDisplayName,
  formatDatasetLabel,
  formatDate,
  formatTimestamp,
  predictedWinnerName,
} from "../displayLabels";
import type { PredictionResponse, PredictionWarning } from "../types";

interface PredictionResultProps {
  prediction: PredictionResponse | null;
  error: string | null;
  isLoading: boolean;
}

function formatPercentage(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    maximumFractionDigits: 1,
  }).format(Math.max(0, Math.min(1, value)));
}

function confidenceClass(tier?: string): string {
  switch (tier?.toLowerCase()) {
    case "standard":
      return "confidence-standard";
    case "reduced":
      return "confidence-reduced";
    case "low":
      return "confidence-low";
    case "unsupported":
      return "confidence-unsupported";
    default:
      return "confidence-reduced";
  }
}

function severityClass(severity: string): string {
  return `warning-${severity.toLowerCase().replace(/[^a-z]+/g, "-")}`;
}

function PredictionWarnings({ warnings }: { warnings: PredictionWarning[] }) {
  if (!warnings.length) {
    return <p className="no-warnings">No model-applicability warnings were returned.</p>;
  }

  return (
    <ul className="warning-list">
      {warnings.map((warning, index) => (
        <li className={`warning-item ${severityClass(warning.severity)}`} key={`${warning.code}-${index}`}>
          <span className="warning-icon" aria-hidden="true">!</span>
          <div>
            <strong>{formatDatasetLabel(warning.code, "warning")}</strong>
            <p>{warning.message}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}

function EmptyState() {
  return (
    <section className="panel result-panel empty-state" aria-live="polite">
      <div className="empty-icon" aria-hidden="true">⌁</div>
      <h2>Your prediction will appear here</h2>
      <p>Enter two fighters to see their estimated win probabilities.</p>
    </section>
  );
}

export function PredictionResult({ prediction, error, isLoading }: PredictionResultProps) {
  if (isLoading && !prediction) {
    return (
      <section className="panel result-panel loading-state" aria-live="polite">
        <span className="spinner large-spinner" aria-hidden="true" />
        <h2>Building the matchup</h2>
        <p>Loading the saved model and constructing pre-fight features.</p>
      </section>
    );
  }

  if (!prediction) {
    return error ? (
      <section className="panel result-panel error-state" role="alert">
        <p className="section-kicker">Prediction unavailable</p>
        <h2>We couldn&apos;t build that matchup.</h2>
        <p>{error}</p>
      </section>
    ) : <EmptyState />;
  }

  const nameA = fighterDisplayName(prediction.fighter_a);
  const nameB = fighterDisplayName(prediction.fighter_b);
  const warnings = prediction.warnings ?? prediction.confidence?.warnings ?? [];
  const confidenceTier = prediction.confidence_tier ?? prediction.confidence?.tier ?? "reduced";
  const confidenceScore = prediction.confidence?.score;
  const winnerName = predictedWinnerName(
    prediction.predicted_winner_name,
    prediction.predicted_winner_id,
    [prediction.fighter_a, prediction.fighter_b],
  );

  return (
    <section className="panel result-panel" aria-live="polite" aria-labelledby="prediction-title">
      {error ? <p className="inline-error" role="alert">Latest request failed: {error}</p> : null}

      <div className="result-header">
        <div>
          <p className="section-kicker">Model result</p>
          <h2 id="prediction-title">{nameA} <span>vs</span> {nameB}</h2>
          <p className="muted">Prediction generated: {formatTimestamp(prediction.predicted_at)}</p>
        </div>
        <span className={`confidence-badge ${confidenceClass(confidenceTier)}`}>
          {formatDatasetLabel(confidenceTier, "confidence")} confidence
        </span>
      </div>

      <div className="winner-callout">
        <p>Projected winner</p>
        <strong>{prediction.is_even_probability ? "Even matchup" : winnerName}</strong>
        <span>
          {prediction.is_even_probability
            ? "The model sees an even probability."
            : `${formatPercentage(Math.max(prediction.probability_a, prediction.probability_b))} estimated win probability`}
        </span>
      </div>

      <div className="probability-grid" aria-label="Estimated win probabilities">
        <article className="probability-card fighter-a-card">
          <p>{nameA}</p>
          <strong>{formatPercentage(prediction.probability_a)}</strong>
          <div className="probability-track" aria-hidden="true">
            <span style={{ width: formatPercentage(prediction.probability_a) }} />
          </div>
        </article>
        <article className="probability-card fighter-b-card">
          <p>{nameB}</p>
          <strong>{formatPercentage(prediction.probability_b)}</strong>
          <div className="probability-track" aria-hidden="true">
            <span style={{ width: formatPercentage(prediction.probability_b) }} />
          </div>
        </article>
      </div>

      <div className="details-grid">
        <div className="detail-group">
          <p className="detail-label">Fight context</p>
          <dl>
            <div><dt>Division</dt><dd>{formatDatasetLabel(prediction.division, "division", "Inferred")}</dd></div>
            <div><dt>Model cutoff</dt><dd>{formatDate(prediction.model_cutoff ?? prediction.dataset_cutoff)}</dd></div>
            <div><dt>Orientation gap</dt><dd>{formatPercentage(prediction.orientation_disagreement ?? prediction.confidence?.orientation_disagreement ?? 0)}</dd></div>
          </dl>
        </div>
        <div className="detail-group">
          <p className="detail-label">Available history</p>
          <dl>
            <div><dt>{nameA} UFC fights</dt><dd>{prediction.prior_ufc_fights_a}</dd></div>
            <div><dt>{nameB} UFC fights</dt><dd>{prediction.prior_ufc_fights_b}</dd></div>
            <div><dt>Snapshot dates</dt><dd>{formatDate(prediction.snapshot_date_a)} / {formatDate(prediction.snapshot_date_b)}</dd></div>
          </dl>
        </div>
      </div>

      <div className="warning-section">
        <div className="warning-heading">
          <div>
            <p className="detail-label">Applicability notes</p>
            <h3>Data quality &amp; confidence</h3>
          </div>
          {typeof confidenceScore === "number" ? <span>{formatPercentage(confidenceScore)} support score</span> : null}
        </div>
        <PredictionWarnings warnings={warnings} />
      </div>

    </section>
  );
}
