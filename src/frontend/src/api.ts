import type { PredictionRequest, PredictionResponse } from "./types";

export class PredictionApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "PredictionApiError";
  }
}

function errorMessage(payload: unknown): string | undefined {
  if (typeof payload === "string" && payload.trim()) {
    return payload;
  }

  if (payload && typeof payload === "object") {
    const candidate = payload as Record<string, unknown>;
    for (const key of ["detail", "message", "error"]) {
      if (typeof candidate[key] === "string" && candidate[key].trim()) {
        return candidate[key];
      }
    }
  }

  return undefined;
}

export async function requestPrediction(
  request: PredictionRequest,
): Promise<PredictionResponse> {
  let response: Response;

  try {
    response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch {
    throw new PredictionApiError(
      "Could not reach the prediction service. Make sure the Python API is running.",
    );
  }

  const payload: unknown = await response.json().catch(() => undefined);
  if (!response.ok) {
    throw new PredictionApiError(
      errorMessage(payload) ?? "The prediction service could not complete this request.",
      response.status,
    );
  }

  if (!payload || typeof payload !== "object") {
    throw new PredictionApiError("The prediction service returned an unexpected response.");
  }

  return payload as PredictionResponse;
}
