import { formatUserFacingError } from "./displayLabels";
import type { FighterOption, PredictionRequest, PredictionResponse } from "./types";

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const apiBaseUrl = configuredApiBaseUrl ? configuredApiBaseUrl.replace(/\/+$/, "") : "";

export class PredictionApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "PredictionApiError";
  }
}

export class FighterSearchApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = "FighterSearchApiError";
  }
}

function isAbortError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    (error as { name: unknown }).name === "AbortError"
  );
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

export async function searchFighters(
  query: string,
  signal?: AbortSignal,
  limit = 8,
): Promise<FighterOption[]> {
  const trimmedQuery = query.trim();
  if (!trimmedQuery) {
    return [];
  }

  const params = new URLSearchParams({
    query: trimmedQuery,
    limit: String(limit),
  });

  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}/api/fighters?${params.toString()}`, {
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (error) {
    if (isAbortError(error)) {
      throw error;
    }
    throw new FighterSearchApiError(
      "Could not reach the fighter search service. Check your connection and try again.",
    );
  }

  const payload: unknown = await response.json().catch(() => undefined);
  if (!response.ok) {
    throw new FighterSearchApiError(
      formatUserFacingError(
        errorMessage(payload) ?? "The fighter search service could not complete this request.",
      ),
      response.status,
    );
  }

  if (!Array.isArray(payload)) {
    throw new FighterSearchApiError("The fighter search service returned an unexpected response.");
  }

  return payload.filter(
    (candidate): candidate is FighterOption =>
      Boolean(candidate) &&
      typeof candidate === "object" &&
      typeof (candidate as Record<string, unknown>).id === "string" &&
      typeof (candidate as Record<string, unknown>).name === "string",
  );
}
