"""
Height / weight parsing and BMI calculation for people who don't know their BMI.

Height can be typed the way people actually say it:
    5'4"   5' 4   5 ft 4 in   5 feet 4 inches   6 feet   6'
    5.4    (read as 5 ft 4 in, the common way of writing it in India; 5.10 / 5.11 = 5 ft 10 / 11 in)
    165 cm   165   1.65 m   1.65   64 in   64"
Weight: a number plus kg or lb (from the unit selector), or typed with its unit:
    70   70 kg   154 lb   154 lbs   154 pounds

Every parse returns a human-readable interpretation (e.g. "5 ft 4 in = 162.6 cm"),
which the form shows live so the person can see how their entry was understood.
"""
import re

CM_PER_INCH = 2.54
KG_PER_LB = 0.45359237
HEIGHT_RANGE_CM = (100, 250)   # plausible adult heights
WEIGHT_RANGE_KG = (25, 300)    # plausible adult weights

_NUM = r"(\d+(?:\.\d+)?)"


def _clean_height(text: str) -> str:
    t = str(text).strip().lower()
    t = t.replace("’", "'").replace("‘", "'").replace("′", "'").replace("″", '"')
    t = t.replace("“", '"').replace("”", '"').replace("''", '"')
    t = re.sub(r"\b(feet|foot|ft)\b\.?", "'", t)
    t = re.sub(r"\b(inches|inch|ins|in)\b\.?", '"', t)
    t = re.sub(r"\b(centimetres|centimeters|centimetre|centimeter|cms|cm)\b", "cm", t)
    t = re.sub(r"\b(metres|meters|metre|meter|mtr|m)\b", "m", t)
    return re.sub(r"\s+", " ", t).strip()


def _fmt_ft_in(total_inches: float) -> str:
    ft, inch = divmod(round(total_inches), 12)
    return f"{ft} ft {inch} in"


def parse_height(text: str) -> tuple[float | None, str, str | None]:
    """Return (height in cm, interpretation, error)."""
    t = _clean_height(text)
    if not t:
        return None, "", "Please enter the height."
    cm, how = None, ""
    if m := re.fullmatch(_NUM + r" ?' ?(?:" + _NUM + r' ?"?)?', t):          # 5'4"  5'  5 ' 4 "
        ft, inch = float(m.group(1)), float(m.group(2) or 0)
        if inch >= 12:
            return None, "", "Inches should be less than 12 (e.g. 5'11\")."
        cm, how = (ft * 12 + inch) * CM_PER_INCH, f"{ft:g} ft {inch:g} in"
    elif (m := re.fullmatch(r"(\d) (\d{1,2})", t)) and 3 <= int(m.group(1)) <= 7 and int(m.group(2)) <= 11:
        # spoken "five four" -> "5 4" -> 5 ft 4 in
        ft, inch = int(m.group(1)), int(m.group(2))
        cm, how = (ft * 12 + inch) * CM_PER_INCH, f"{ft} ft {inch} in"
    elif m := re.fullmatch(_NUM + r' ?"', t):                                # 64"
        cm, how = float(m.group(1)) * CM_PER_INCH, f"{float(m.group(1)):g} in"
    elif m := re.fullmatch(_NUM + r" ?cm", t):
        cm, how = float(m.group(1)), f"{float(m.group(1)):g} cm"
    elif m := re.fullmatch(_NUM + r" ?m", t):
        cm, how = float(m.group(1)) * 100, f"{float(m.group(1)):g} m"
    elif re.fullmatch(_NUM, t):                                              # bare number: guess the unit
        x = float(t)
        if 3 <= x < 8:
            whole, _, frac = t.partition(".")
            if frac and len(frac) <= 2 and int(frac) <= 11:                  # 5.4 -> 5 ft 4 in
                cm, how = (int(whole) * 12 + int(frac)) * CM_PER_INCH, f"{int(whole)} ft {int(frac)} in"
            else:                                                            # 6 -> 6 ft, 5.75 -> 5.75 ft
                cm, how = x * 12 * CM_PER_INCH, _fmt_ft_in(x * 12)
        elif 1 <= x <= 2.5:
            cm, how = x * 100, f"{x:g} m"
        elif 40 <= x < 100:
            cm, how = x * CM_PER_INCH, f"{x:g} in"
        elif 100 <= x <= 250:
            cm, how = x, f"{x:g} cm"
    if cm is None:
        return None, "", "Could not understand this height. Try 5'4\", 5.4, 165 cm or 1.65 m."
    if not HEIGHT_RANGE_CM[0] <= cm <= HEIGHT_RANGE_CM[1]:
        return None, "", f"{how} = {cm:.0f} cm looks unusual for an adult. Please check it."
    return round(cm, 1), (how if how.endswith(" cm") else f"{how} = {cm:.1f} cm"), None


def parse_weight(text: str, unit: str = "kg") -> tuple[float | None, str, str | None]:
    """Return (weight in kg, interpretation, error). Units typed in the text win over `unit`."""
    t = str(text).strip().lower()
    if not t:
        return None, "", "Please enter the weight."
    m = re.fullmatch(_NUM + r" ?(kgs?|kilos?|kilograms?|lbs?|pounds?)?\.?", t)
    if not m:
        return None, "", "Could not understand this weight. Try 70, 70 kg or 154 lb."
    x, typed = float(m.group(1)), m.group(2)
    is_lb = (typed or unit or "kg").startswith(("lb", "pound"))
    kg = x * KG_PER_LB if is_lb else x
    how = f"{x:g} lb = {kg:.1f} kg" if is_lb else f"{x:g} kg"
    if not WEIGHT_RANGE_KG[0] <= kg <= WEIGHT_RANGE_KG[1]:
        return None, "", f"{how} looks unusual for an adult. Please check the number and the unit."
    return round(kg, 1), how, None


def bmi_from(height_text: str, weight_text: str, weight_unit: str = "kg") -> dict:
    """Parse both values and compute BMI = kg / m^2."""
    cm, h_how, h_err = parse_height(height_text)
    kg, w_how, w_err = parse_weight(weight_text, weight_unit)
    out = {"height_cm": cm, "height_text": h_how, "height_error": h_err,
           "weight_kg": kg, "weight_text": w_how, "weight_error": w_err, "bmi": None}
    if cm and kg:
        out["bmi"] = round(kg / (cm / 100) ** 2, 1)
    return out


# ---------------------------------------------------------------------------
# Unit-based entry (profile form): the person picks a unit and types numbers
# ---------------------------------------------------------------------------
HEIGHT_UNITS = [("ft_in", "Feet and inches"), ("cm", "Centimetres (cm)"), ("in", "Inches only"), ("m", "Metres (m)")]
WEIGHT_UNITS = [("kg", "Kilograms (kg)"), ("lb", "Pounds (lb)")]


def _num(v) -> float | None:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def height_from_units(unit: str, value=None, ft=None, inch=None) -> tuple[float | None, str, str | None]:
    """Height in cm from a chosen unit. Returns (cm, readable text, error)."""
    if unit == "ft_in":
        f, i = _num(ft), _num(inch) if str(inch or "").strip() else 0.0
        if f is None or i is None:
            return None, "", "Enter feet (and inches if needed), e.g. 5 feet 4 inches."
        if not 0 <= i < 12:
            return None, "", "Inches should be between 0 and 11."
        cm, how = (f * 12 + i) * CM_PER_INCH, f"{f:g} ft {i:g} in"
    else:
        x = _num(value)
        if x is None:
            return None, "", "Please enter the height."
        cm, how = {"cm": (x, f"{x:g} cm"), "in": (x * CM_PER_INCH, f"{x:g} in"),
                   "m": (x * 100, f"{x:g} m")}.get(unit, (None, ""))
        if cm is None:
            return None, "", "Choose a height unit."
    if not HEIGHT_RANGE_CM[0] <= cm <= HEIGHT_RANGE_CM[1]:
        return None, "", f"{how} = {cm:.0f} cm looks unusual for an adult. Please check the number and the unit."
    return round(cm, 1), (how if unit == "cm" else f"{how} = {cm:.1f} cm"), None


def weight_from_units(unit: str, value) -> tuple[float | None, str, str | None]:
    x = _num(value)
    if x is None:
        return None, "", "Please enter the weight."
    return parse_weight(f"{x:g}", unit)


def bmi_category(bmi: float | None) -> str:
    if bmi is None:
        return ""
    return ("Underweight" if bmi < 18.5 else "Healthy weight" if bmi < 25 else "Overweight" if bmi < 30
            else "Obese") + " (WHO adult categories)"


def spoken_height(text: str) -> dict:
    """Understand a spoken/typed height and say which unit fields it belongs in (used by voice help)."""
    cm, how, err = parse_height(text)
    if err:
        return {"error": err}
    if "ft" in how:
        ft, inch = re.match(r"([\d.]+) ft ([\d.]+) in", how).groups()
        return {"unit": "ft_in", "ft": float(ft), "inch": float(inch), "cm": cm, "text": how}
    unit = "cm" if how.split(" = ")[0].endswith(" cm") else "m" if how.split(" = ")[0].endswith(" m") else "in"
    return {"unit": unit, "value": float(how.split(" ")[0]), "cm": cm, "text": how}


def height_fields(cm: float, unit: str) -> dict:
    """Pre-fill values for the profile form, in the unit the person chose last time."""
    if unit == "ft_in":
        ft, inch = divmod(round(cm / CM_PER_INCH), 12)
        return {"height_ft": ft, "height_inch": inch}
    return {"height_value": {"cm": round(cm, 1), "in": round(cm / CM_PER_INCH, 1), "m": round(cm / 100, 2)}[unit]}
