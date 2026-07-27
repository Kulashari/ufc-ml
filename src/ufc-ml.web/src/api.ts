import { formatUserFacingError } from "./displayLabels";
import type { PredictionRequest, PredictionResponse } from "./types";

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const apiBaseUrl = configuredApiBaseUrl ? configuredApiBaseUrl.replace(/\/+$/, "") : "";

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
    response = await fetch(`${apiBaseUrl}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch {
    throw new PredictionApiError(
      "Could not reach the prediction service. Check your connection and try again.",
    );
  }

  const payload: unknown = await response.json().catch(() => undefined);
  if (!response.ok) {
    throw new PredictionApiError(
      formatUserFacingError(
        errorMessage(payload) ?? "The prediction service could not complete this request.",
      ),
      response.status,
    );
  }

  if (!payload || typeof payload !== "object") {
    throw new PredictionApiError("The prediction service returned an unexpected response.");
  }

  return payload as PredictionResponse;
}
