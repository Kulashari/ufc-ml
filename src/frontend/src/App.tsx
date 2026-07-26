import { useState } from "react";

import { PredictionForm } from "./components/PredictionForm";
import { PredictionResult } from "./components/PredictionResult";
import { requestPrediction } from "./api";
import type { PredictionRequest, PredictionResponse } from "./types";

function App() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  async function handlePrediction(request: PredictionRequest) {
    setIsLoading(true);
    setError(null);

    try {
      const result = await requestPrediction(request);
      setPrediction(result);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Unable to make a prediction.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="hero" aria-labelledby="app-title">
        <div className="eyebrow"><span aria-hidden="true">◉</span> UFC prediction lab</div>
        <h1 id="app-title">Matchup probability, made clear.</h1>
        <p>
          Compare two fighters using the trained model&apos;s latest available pre-fight data.
        </p>
      </header>

      <section className="prediction-layout" aria-label="Fight prediction workspace">
        <PredictionForm
          isLoading={isLoading}
          onSubmit={handlePrediction}
        />
        <PredictionResult prediction={prediction} error={error} isLoading={isLoading} />
      </section>

      <footer className="app-footer">
        This research model estimates probabilities, not guaranteed outcomes or betting advice.
      </footer>
    </main>
  );
}

export default App;
