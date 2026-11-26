"""
Trend analysis over a person's stored tests.

Every test is timestamped, so any two tests can be compared:
  - risk probability and level,
  - memory and thinking score (MMSE scale), everyday tasks score, self-care score,
  - number of memory/behaviour symptoms, and BMI.
An overall verdict (Improving / Stable / Worsening) is given in plain words.

The rules are simple and transparent (documented below) so they can be explained
in a viva; they are screening summaries, not clinical judgements.
"""
from src import config

LEVELS = {"Low": 0, "Medium": 1, "High": 2}
# (key, readable name, higher_is_better, smallest change that counts, unit)
MEASURES = [
    ("probability", "Risk of Alzheimer's pattern", False, 0.05, "%"),
    ("MMSE", "Memory and thinking score", True, 1.0, "/30"),
    ("FunctionalAssessment", "Everyday tasks score", True, 0.5, "/10"),
    ("ADL", "Self-care score", True, 0.5, "/10"),
    ("symptoms", "Memory and behaviour symptoms", False, 1, ""),
    ("BMI", "BMI", None, 0.5, ""),
]
RISK_JUMP = 0.10          # a change of 10 percentage points or more decides the verdict
SCORE_JUMP = {"MMSE": 2.0, "FunctionalAssessment": 1.0, "ADL": 1.0}


def record(pred) -> dict:
    """One test as a flat dictionary."""
    obs, i = pred.observation, pred.observation.inputs
    return {
        "prediction_id": pred.id, "when": obs.recorded_at, "kind": obs.kind_label,
        "simulated": obs.is_simulated, "probability": pred.probability, "risk_level": pred.risk_level,
        "MMSE": i.get("MMSE"), "FunctionalAssessment": i.get("FunctionalAssessment"), "ADL": i.get("ADL"),
        "symptoms": sum(int(i.get(s) or 0) for s in config.SYMPTOM_FEATURES), "BMI": i.get("BMI"),
        "scores_from": "clinic" if (i.get("_assessment") or {}).get("mode") != "questions" else "home estimate",
    }


def records(patient) -> list[dict]:
    preds = sorted(patient.predictions, key=lambda p: (p.observation.recorded_at, p.id))
    return [record(p) for p in preds]


def _fmt(key, v):
    if v is None:
        return "-"
    if key == "probability":
        return f"{100 * v:.1f}%"
    return f"{v:g}" if key == "symptoms" else f"{v:.1f}"


def compare(a: dict, b: dict) -> dict:
    """Change from test a (earlier) to test b (later)."""
    days = (b["when"] - a["when"]).days
    rows, better, worse = [], 0, 0
    for key, name, higher_better, min_change, unit in MEASURES:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            continue
        d = vb - va
        if abs(d) < min_change:
            direction = "No real change"
        elif higher_better is None:
            direction = "Higher" if d > 0 else "Lower"
        else:
            good = (d > 0) == higher_better
            direction = "Better" if good else "Worse"
        if key in SCORE_JUMP and abs(d) >= SCORE_JUMP[key]:
            better, worse = better + (direction == "Better"), worse + (direction == "Worse")
        change = f"{100 * d:+.1f} points" if key == "probability" else f"{d:+.1f}" if key != "symptoms" else f"{d:+g}"
        per_year = None
        if key == "MMSE" and days >= 30:
            per_year = round(d / days * 365, 1)
        rows.append({"key": key, "name": name, "before": _fmt(key, va), "after": _fmt(key, vb),
                     "change": change, "direction": direction, "unit": unit, "per_year": per_year})

    la, lb = LEVELS[a["risk_level"]], LEVELS[b["risk_level"]]
    dp = b["probability"] - a["probability"]
    if lb > la or dp >= RISK_JUMP:
        verdict, why = "Worsening", "the estimated risk went up"
    elif lb < la or dp <= -RISK_JUMP:
        verdict, why = "Improving", "the estimated risk went down"
    elif worse > better:
        verdict, why = "Possibly worsening", "the risk is similar but memory or daily-life scores fell"
    elif better > worse:
        verdict, why = "Possibly improving", "the risk is similar but memory or daily-life scores rose"
    else:
        verdict, why = "Stable", "there was no meaningful change"
    notes = []
    if a["scores_from"] != b["scores_from"]:
        notes.append("One test used home-estimated scores and the other used clinic scores, so part of the "
                     "change may come from the different way of measuring.")
    if a["simulated"] or b["simulated"]:
        notes.append("Includes simulated demo data.")
    return {"a": a, "b": b, "days": days, "rows": rows, "verdict": verdict, "why": why, "notes": notes}


def summary(patient) -> dict:
    """Latest vs previous test, latest vs first test, and days since the last test."""
    recs = records(patient)
    out = {"records": recs, "count": len(recs), "since_previous": None, "since_first": None}
    if len(recs) >= 2:
        out["since_previous"] = compare(recs[-2], recs[-1])
    if len(recs) >= 3:
        out["since_first"] = compare(recs[0], recs[-1])
    return out
