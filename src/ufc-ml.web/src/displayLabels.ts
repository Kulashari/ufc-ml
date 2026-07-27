import type { Fighter } from "./types";

export type DatasetLabelKind =
  | "confidence"
  | "division"
  | "method"
  | "outcome"
  | "stance"
  | "warning"
  | "weight_class";

type LabelMap = Readonly<Record<string, string>>;

const DIVISION_LABELS: LabelMap = {
  CATCH: "Catchweight",
  M_BANTAM: "Male Bantamweight",
  M_FEATHER: "Male Featherweight",
  M_FLY: "Male Flyweight",
  M_HEAVY: "Male Heavyweight",
  M_LIGHT: "Male Lightweight",
  M_LIGHT_HEAVY: "Male Light Heavyweight",
  M_MIDDLE: "Male Middleweight",
  M_WELTER: "Male Welterweight",
  OPEN: "Openweight",
  UNKNOWN: "Unknown division",
  UNSPECIFIED: "Unspecified division",
  W_BANTAM: "Female Bantamweight",
  W_FEATHER: "Female Featherweight",
  W_FLY: "Female Flyweight",
  W_STRAW: "Female Strawweight",
};

const OUTCOME_LABELS: LabelMap = {
  D: "Draw",
  DRAW: "Draw",
  L: "Loss",
  LOSS: "Loss",
  NC: "No contest",
  NO_CONTEST: "No contest",
  OVERTURNED: "Overturned",
  W: "Win",
  WIN: "Win",
};

const METHOD_LABELS: LabelMap = {
  COULD_NOT_CONTINUE: "Could not continue",
  DECISION_MAJORITY: "Majority decision",
  DECISION_SPLIT: "Split decision",
  DECISION_UNANIMOUS: "Unanimous decision",
  DQ: "Disqualification",
  KO: "Knockout",
  "KO/TKO": "KO/TKO",
  M_DEC: "Majority decision",
  OTHER: "Other method",
  S_DEC: "Split decision",
  SUB: "Submission",
  SUBMISSION: "Submission",
  TKO: "Technical knockout",
  "TKO_DOCTOR'S_STOPPAGE": "Technical knockout (doctor's stoppage)",
  U_DEC: "Unanimous decision",
};

const STANCE_LABELS: LabelMap = {
  OPEN_STANCE: "Open stance",
  ORTHODOX: "Orthodox",
  OTHER: "Other stance",
  SIDEWAYS: "Sideways stance",
  SOUTHPAW: "Southpaw",
  SWITCH: "Switch",
  UNKNOWN: "Unknown stance",
};

const CONFIDENCE_LABELS: LabelMap = {
  LOW: "Low",
  REDUCED: "Reduced",
  STANDARD: "Standard",
  UNSUPPORTED: "Unsupported",
};

const WARNING_LABELS: LabelMap = {
  AFTER_CUTOFF: "Date after data cutoff",
  DEBUT: "UFC debut",
  LIMITED_HISTORY: "Limited UFC history",
  ORIENTATION_DISAGREEMENT: "Model orientation disagreement",
  OUT_OF_RANGE: "Outside training range",
  STALE: "Stale fighter data",
  UNSUPPORTED: "Unsupported matchup",
};

const WEIGHT_CLASS_LABELS: LabelMap = {
  CATCH_WEIGHT_BOUT: "Catchweight",
  OPEN_WEIGHT_BOUT: "Openweight",
};

const LABELS: Readonly<Record<DatasetLabelKind, LabelMap>> = {
  confidence: CONFIDENCE_LABELS,
  division: DIVISION_LABELS,
  method: METHOD_LABELS,
  outcome: OUTCOME_LABELS,
  stance: STANCE_LABELS,
  warning: WARNING_LABELS,
  weight_class: WEIGHT_CLASS_LABELS,
};

const ALL_LABELS: LabelMap = Object.assign({}, ...Object.values(LABELS));
const INTERNAL_ID = /^[0-9a-f]{12,}$/i;

export const DIVISION_OPTIONS = Object.entries(DIVISION_LABELS)
  .filter(([value]) => !["OPEN", "UNKNOWN", "UNSPECIFIED"].includes(value))
  .map(([value, label]) => ({ value, label }));

function normalizeLabelKey(value: string): string {
  return value.trim().toUpperCase().replace(/[ -]+/g, "_");
}

function humanizeDatasetValue(value: string): string {
  return value
    .trim()
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function labelForWeightClass(value: string): string | undefined {
  const normalized = value.trim();
  if (/\s+bout$/i.test(normalized)) {
    return normalized.replace(/\s+bout$/i, "");
  }
  return undefined;
}

function cleanFighterName(value: string | null | undefined): string | undefined {
  const cleaned = value?.trim();
  return cleaned && !INTERNAL_ID.test(cleaned) ? cleaned : undefined;
}

export function formatDatasetLabel(
  value: string | number | null | undefined,
  kind: DatasetLabelKind,
  fallback = "Unknown",
): string {
  if (value === null || value === undefined || String(value).trim() === "") {
    return fallback;
  }

  const raw = String(value).trim();
  const key = normalizeLabelKey(raw);
  const mapped = LABELS[kind][key] ?? ALL_LABELS[key];
  if (mapped) {
    return mapped;
  }
  if (kind === "weight_class") {
    return labelForWeightClass(raw) ?? humanizeDatasetValue(raw);
  }
  return humanizeDatasetValue(raw);
}

export function formatDate(value: string | null | undefined, fallback = "Not available"): string {
  const isoDate = value?.slice(0, 10);
  if (!isoDate || !/^\d{4}-\d{2}-\d{2}$/.test(isoDate)) {
    return fallback;
  }
  const parsed = new Date(`${isoDate}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) {
    return fallback;
  }
  return new Intl.DateTimeFormat("en-US", {
    day: "numeric",
    month: "short",
    timeZone: "UTC",
    year: "numeric",
  }).format(parsed);
}

export function formatTimestamp(
  value: string | null | undefined,
  fallback = "Not available",
): string {
  if (!value) {
    return fallback;
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return fallback;
  }
  return new Intl.DateTimeFormat("en-US", {
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    month: "short",
    timeZone: "UTC",
    timeZoneName: "short",
    year: "numeric",
  }).format(parsed);
}

export function fighterDisplayName(fighter: Fighter | null | undefined): string {
  return (
    cleanFighterName(fighter?.display_name) ??
    cleanFighterName(fighter?.fighter_name) ??
    "Unknown Fighter"
  );
}

export function predictedWinnerName(
  winnerName: string | null | undefined,
): string {
  return cleanFighterName(winnerName) ?? "Unknown Fighter";
}

export function formatUserFacingError(message: string): string {
  return Object.entries(ALL_LABELS)
    .sort(([left], [right]) => right.length - left.length)
    .reduce(
      (formatted, [raw, label]) => formatted.replace(new RegExp(`\\b${raw}\\b`, "gi"), label),
      message.replace(/\bfighter[_ -]?id\b/gi, "fighter"),
    );
}
