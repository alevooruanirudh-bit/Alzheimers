"""
Personal profile: details asked ONCE and edited when they change.

  name, date of birth (or approximate age), gender, ethnicity, education,
  height and weight (any common unit) -> BMI, family and medical history,
  lifestyle, and clinic numbers (blood pressure, cholesterol) when available.

Tests (home checks, clinical reports) take these from the profile at the time
of the test, so age is always current and a weight change after "Edit profile"
is used from the next test onwards. Every height/weight change is also stored
as a timestamped body measurement for the BMI trend.
"""
from datetime import date, datetime

from src import body
from src.schema import CHILD_MESSAGE, ETHNICITY_BY_KEY, FIELD_BY_NAME

HISTORY_FIELDS = ["FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression", "HeadInjury",
                  "Hypertension"]
SELECT_FIELDS = ["Gender", "EducationLevel"]
LIFESTYLE_FIELDS = ["Smoking", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality"]
CLINIC_FIELDS = ["SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
                 "CholesterolTriglycerides"]


def age_on(profile: dict, when: datetime | date | None = None) -> float:
    """Age in whole years at a given moment (today by default)."""
    born = date.fromisoformat(profile["birth_date"])
    d = (when or datetime.now())
    d = d.date() if isinstance(d, datetime) else d
    return float(d.year - born.year - ((d.month, d.day) < (born.month, born.day)))


def validate_profile(form: dict, today: date | None = None) -> tuple[dict, dict]:
    """Return (profile, errors) from the profile form."""
    today = today or date.today()
    p, errors = {}, {}
    p["name"] = str(form.get("name", "")).strip()
    if not p["name"]:
        errors["name"] = "Please enter the person's name."

    # date of birth, or approximate age for people who do not know it
    if form.get("age_mode") == "age":
        try:
            age = int(float(form.get("age_years", "")))
            p["birth_date"] = date(today.year - age, today.month, min(today.day, 28)).isoformat()
            p["birth_date_approx"] = True
        except (TypeError, ValueError):
            errors["age_years"] = "Please enter the age in years."
    else:
        try:
            p["birth_date"] = date.fromisoformat(str(form.get("birth_date", ""))).isoformat()
            p["birth_date_approx"] = False
        except ValueError:
            errors["birth_date"] = "Please enter the date of birth."
    if "birth_date" in p:
        age = age_on(p, today)
        if age < 18:
            errors["age"] = CHILD_MESSAGE
        elif age > 105:
            errors["age"] = "Please check the date of birth or age."

    for name in SELECT_FIELDS + HISTORY_FIELDS:
        f, raw = FIELD_BY_NAME[name], form.get(name, "")
        if f.get("optional") and str(raw).strip() == "":
            p[name] = None
            continue
        try:
            v = int(raw)
            assert v in [o[0] for o in f["options"]]
            p[name] = v
        except (TypeError, ValueError, AssertionError):
            errors[name] = "Please choose an answer."
    key = form.get("EthnicityDetail", "")
    if key in ETHNICITY_BY_KEY:
        p["EthnicityDetail"], p["Ethnicity"] = key, ETHNICITY_BY_KEY[key][1]
    elif not key:
        p["EthnicityDetail"], p["Ethnicity"] = None, None
    else:
        errors["EthnicityDetail"] = "Please choose an option."

    for name in LIFESTYLE_FIELDS + CLINIC_FIELDS:
        f, raw = FIELD_BY_NAME[name], str(form.get(name, "")).strip()
        if raw == "":
            p[name] = None
            continue
        try:
            v = float(raw)
        except ValueError:
            errors[name] = "Please enter a number."
            continue
        if f["kind"] == "select":
            if v not in [o[0] for o in f["options"]]:
                errors[name] = "Choose one of the listed options."
                continue
            v = int(v)
        elif not (f["min"] <= v <= f["max"]):
            errors[name] = f"Must be between {f['min']:g} and {f['max']:g}."
            continue
        p[name] = v
    # blood pressure needs both numbers (same rule as the test forms)
    upper, lower = p.get("SystolicBP"), p.get("DiastolicBP")
    if not ({"SystolicBP", "DiastolicBP"} & set(errors)):
        if (upper is None) != (lower is None):
            errors["SystolicBP" if upper is None else "DiastolicBP"] = "Enter both blood pressure numbers, or leave both empty."
        elif upper is not None and upper <= lower:
            errors["SystolicBP"] = "The upper number should be larger than the lower number."

    # height and weight in the units the person chose
    p["height_unit"] = form.get("height_unit", "ft_in")
    p["weight_unit"] = form.get("weight_unit", "kg")
    cm, h_text, h_err = body.height_from_units(p["height_unit"], form.get("height_value"),
                                               form.get("height_ft"), form.get("height_inch"))
    kg, w_text, w_err = body.weight_from_units(p["weight_unit"], form.get("weight_value"))
    if h_err:
        errors["height"] = h_err
    if w_err:
        errors["weight"] = w_err
    if cm and kg:
        p.update(height_cm=cm, weight_kg=kg, bmi=round(kg / (cm / 100) ** 2, 1),
                 height_text=h_text, weight_text=w_text)
        # exactly what was typed, so "Edit profile" shows it back unchanged
        p["entered"] = {k: str(form.get(k, "")).strip() for k in ("height_value", "height_ft", "height_inch", "weight_value")}
    return p, errors


def model_inputs(profile: dict, when: datetime | None = None) -> dict:
    """The profile part of the model inputs, as they are at the time of a test."""
    out = {k: profile[k] for k in SELECT_FIELDS + HISTORY_FIELDS + ["Ethnicity", "EthnicityDetail"]}
    out.update({k: profile.get(k) for k in LIFESTYLE_FIELDS + CLINIC_FIELDS})
    out["Age"] = age_on(profile, when)
    out["BMI"] = profile["bmi"]
    out["_body"] = {"height_cm": profile["height_cm"], "weight_kg": profile["weight_kg"],
                    "height_entered": profile.get("height_text"), "weight_entered": profile.get("weight_text"),
                    "weight_unit": profile.get("weight_unit")}
    return out


def form_values(profile: dict) -> dict:
    """Pre-fill the edit-profile form from a saved profile."""
    v = {k: profile.get(k) for k in ["name", "EthnicityDetail", "height_unit", "weight_unit"] + SELECT_FIELDS + HISTORY_FIELDS
         + LIFESTYLE_FIELDS + CLINIC_FIELDS}
    if profile.get("birth_date_approx"):
        v.update(age_mode="age", age_years=int(age_on(profile)))
    else:
        v.update(age_mode="dob", birth_date=profile.get("birth_date"))
    if profile.get("entered"):
        v.update(profile["entered"])
    elif profile.get("height_cm"):
        v.update(body.height_fields(profile["height_cm"], profile.get("height_unit", "ft_in")))
    if profile.get("weight_kg") and not profile.get("entered"):
        kg = profile["weight_kg"]
        v["weight_value"] = round(kg / body.KG_PER_LB, 1) if profile.get("weight_unit") == "lb" else kg
    return v


def apply_to_patient(patient, profile: dict) -> bool:
    """Save the profile on the twin. Returns True if height or weight changed."""
    old = patient.profile or {}
    patient.profile = profile
    patient.name = profile["name"]
    patient.age = age_on(profile)
    patient.gender, patient.ethnicity = profile["Gender"], profile["Ethnicity"] or 0
    patient.education_level, patient.family_history = profile["EducationLevel"] or 0, profile["FamilyHistoryAlzheimers"]
    patient.updated_at = datetime.now()
    return (old.get("height_cm"), old.get("weight_kg")) != (profile["height_cm"], profile["weight_kg"])


def stored_from_inputs(inputs: dict) -> dict:
    """Lifestyle and clinic numbers from a test record, in the form the profile keeps them."""
    out = {k: inputs.get(k) for k in LIFESTYLE_FIELDS + CLINIC_FIELDS}
    out["Smoking"] = None if out["Smoking"] is None else int(out["Smoking"])
    return out


def profile_from_inputs(name: str, inputs: dict, today: date | None = None) -> dict:
    """Build a profile from an older record (used for existing twins and demo accounts)."""
    today = today or date.today()
    age = int(inputs["Age"])
    bmi = float(inputs["BMI"])
    height_cm = 165.0                                        # not recorded in older records: assume 165 cm
    weight_kg = round(bmi * (height_cm / 100) ** 2, 1)      # and derive the weight that gives the same BMI
    p = {"name": name, "birth_date": date(today.year - age, today.month, min(today.day, 28)).isoformat(),
         "birth_date_approx": True, "height_unit": "cm", "weight_unit": "kg", "height_cm": height_cm,
         "weight_kg": weight_kg, "bmi": round(bmi, 1), "height_text": f"{height_cm:g} cm",
         "weight_text": f"{weight_kg:g} kg", "height_assumed": True}
    for k in SELECT_FIELDS + HISTORY_FIELDS + ["Ethnicity"]:
        p[k] = int(inputs[k])
    from src.schema import DEFAULT_KEY_FOR_CODE
    p["EthnicityDetail"] = inputs.get("EthnicityDetail") or DEFAULT_KEY_FOR_CODE[p["Ethnicity"]]
    p.update(stored_from_inputs(inputs))
    return p
