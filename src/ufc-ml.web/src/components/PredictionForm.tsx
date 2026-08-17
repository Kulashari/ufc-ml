import { useState } from "react";
import type { FormEvent } from "react";

import { DIVISION_OPTIONS } from "../displayLabels";
import type { FighterOption, PredictionRequest } from "../types";
import { FighterAutocomplete } from "./FighterAutocomplete";

interface PredictionFormProps {
  isLoading: boolean;
  onSubmit: (request: PredictionRequest) => Promise<void>;
}

export function PredictionForm({ isLoading, onSubmit }: PredictionFormProps) {
  const [fighterA, setFighterA] = useState<FighterOption | null>(null);
  const [fighterB, setFighterB] = useState<FighterOption | null>(null);
  const [division, setDivision] = useState("");
  const [formError, setFormError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!fighterA || !fighterB) {
      setFormError("Select both fighters from the suggestions.");
      return;
    }

    if (fighterA.id === fighterB.id) {
      setFormError("Choose two different fighters.");
      return;
    }

    setFormError(null);
    await onSubmit({
      fighter_a: fighterA.name,
      fighter_b: fighterB.name,
      fighter_a_id: fighterA.id,
      fighter_b_id: fighterB.id,
      ...(division.trim() ? { division: division.trim() } : {}),
    });
  }

  return (
    <section className="panel form-panel" aria-labelledby="matchup-form-title">
      <div className="panel-heading">
        <p className="section-kicker">New prediction</p>
        <h2 id="matchup-form-title">Build a matchup</h2>
        <p className="muted">Search for and select two fighters.</p>
      </div>

      <form className="matchup-form" onSubmit={submit} noValidate>
        <FighterAutocomplete
          id="fighter-a"
          name="fighterA"
          label="Fighter A"
          value={fighterA}
          onChange={(fighter) => {
            setFighterA(fighter);
            setFormError(null);
          }}
          excludeFighterId={fighterB?.id}
          placeholder="e.g. Ilia Topuria"
          enterKeyHint="next"
          disabled={isLoading}
        />

        <div className="versus" aria-hidden="true"><span>VS</span></div>

        <FighterAutocomplete
          id="fighter-b"
          name="fighterB"
          label="Fighter B"
          value={fighterB}
          onChange={(fighter) => {
            setFighterB(fighter);
            setFormError(null);
          }}
          excludeFighterId={fighterA?.id}
          placeholder="e.g. Max Holloway"
          enterKeyHint="done"
          disabled={isLoading}
        />

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
