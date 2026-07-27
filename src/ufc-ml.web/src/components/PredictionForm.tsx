import { useState } from "react";
import type { FormEvent } from "react";

import { DIVISION_OPTIONS } from "../displayLabels";
import type { PredictionRequest } from "../types";

interface PredictionFormProps {
  isLoading: boolean;
  onSubmit: (request: PredictionRequest) => Promise<void>;
}

export function PredictionForm({ isLoading, onSubmit }: PredictionFormProps) {
  const [fighterA, setFighterA] = useState("");
  const [fighterB, setFighterB] = useState("");
  const [division, setDivision] = useState("");
  const [formError, setFormError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const first = fighterA.trim();
    const second = fighterB.trim();

    if (!first || !second) {
      setFormError("Enter both fighter names.");
      return;
    }

    if (first.localeCompare(second, undefined, { sensitivity: "accent" }) === 0) {
      setFormError("Choose two different fighters.");
      return;
    }

    setFormError(null);
    await onSubmit({
      fighter_a: first,
      fighter_b: second,
      ...(division.trim() ? { division: division.trim() } : {}),
    });
  }

  return (
    <section className="panel form-panel" aria-labelledby="matchup-form-title">
      <div className="panel-heading">
        <p className="section-kicker">New prediction</p>
        <h2 id="matchup-form-title">Build a matchup</h2>
        <p className="muted">Use the fighters&apos; names as they appear in the dataset.</p>
      </div>

      <form className="matchup-form" onSubmit={submit} noValidate>
        <label className="field-label" htmlFor="fighter-a">
          Fighter A
          <input
            id="fighter-a"
            name="fighterA"
            type="text"
            value={fighterA}
            onChange={(event) => setFighterA(event.target.value)}
            placeholder="e.g. Ilia Topuria"
            autoComplete="off"
            autoCapitalize="words"
            enterKeyHint="next"
            spellCheck={false}
            disabled={isLoading}
            required
          />
        </label>

        <div className="versus" aria-hidden="true"><span>VS</span></div>

        <label className="field-label" htmlFor="fighter-b">
          Fighter B
          <input
            id="fighter-b"
            name="fighterB"
            type="text"
            value={fighterB}
            onChange={(event) => setFighterB(event.target.value)}
            placeholder="e.g. Max Holloway"
            autoComplete="off"
            autoCapitalize="words"
            enterKeyHint="done"
            spellCheck={false}
            disabled={isLoading}
            required
          />
        </label>

        <label className="field-label" htmlFor="division">
          Division <span className="optional">optional</span>
          <select
            id="division"
            name="division"
            value={division}
            onChange={(event) => setDivision(event.target.value)}
            disabled={isLoading}
          >
            <option value="">Infer from fighter records</option>
            {DIVISION_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        {formError ? <p className="form-error" role="alert">{formError}</p> : null}

        <button className="predict-button" type="submit" disabled={isLoading}>
          {isLoading ? <><span className="spinner" aria-hidden="true" />Calculating...</> : "Predict matchup"}
        </button>
      </form>

      <p className="form-note">
        The model uses the latest available fighter snapshot and may flag limited history or
        stale data.
      </p>
    </section>
  );
}
