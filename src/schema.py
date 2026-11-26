"""
Input field definitions (ranges, units, help text) taken from the dataset
documentation. Used by the web form, server-side validation and the what-if
sliders, so all three always agree.
"""
from src import config

YES_NO = [(0, "No"), (1, "Yes")]

SECTIONS = [
    ("About the person", "demographics"),
    ("Lifestyle", "lifestyle"),
    ("Health history", "history"),
    ("Health check numbers (optional)", "clinical"),
    ("Memory and behaviour", "symptoms"),
    ("Memory, thinking and daily life scores", "cognitive"),
]

FIELDS = [
    # name, label, section, kind, min, max, step, unit, help, options
    dict(name="Age", say="How old is the person, in years?", label="Age", section="demographics", kind="number", min=18, max=105, step=1, unit="years",
         help="For adults (18 and over). The research data covers ages 60 to 90; results for younger "
              "or older people are less certain."),
    dict(name="Gender", say="Is the person male or female?", label="Gender", section="demographics", kind="select", options=[(0, "Male"), (1, "Female")]),
    dict(name="Ethnicity", label="Ethnicity", section="demographics", kind="ethnicity",
         options=[(0, "Caucasian"), (1, "African American"), (2, "Asian"), (3, "Other")],
         help="Choose the closest match. The model was trained on 4 broad groups, so your choice is "
              "mapped to the \"Model group\" shown above. Ethnicity has almost no effect on this model's risk score."),
    dict(name="EducationLevel", say="What is their highest education? No schooling, school, a graduate degree, or a higher degree?", label="Education level", section="demographics", kind="select",
         options=[(0, "None"), (1, "High school"), (2, "Bachelor's"), (3, "Higher")]),

    dict(name="BMI", say="What is their BMI, the body mass index number?", label="Body mass index (BMI)", section="lifestyle", kind="number", min=10, max=60, step=0.1,
         unit="kg/m2", help="Don't know it? Choose \"Enter height and weight\" and it is calculated for you."),
    dict(name="Smoking", say="Do they smoke now? Please say yes or no.", label="Current smoker", section="lifestyle", kind="select", options=YES_NO),
    dict(name="AlcoholConsumption", say="About how many alcoholic drinks do they have in a week? Say zero if none.", label="Alcohol consumption", section="lifestyle", kind="number", min=0, max=20,
         step=0.1, unit="units/week", help="Weekly alcohol units, 0 to 20."),
    dict(name="PhysicalActivity", say="About how many hours of exercise or walking do they do in a week?", label="Physical activity", section="lifestyle", kind="number", min=0, max=10,
         step=0.1, unit="hours/week", help="Weekly hours of physical activity, 0 to 10."),
    dict(name="DietQuality", say="On a scale of zero to ten, how healthy is their food? Zero is very poor, ten is excellent.", label="Diet quality", section="lifestyle", kind="number", min=0, max=10, step=0.1,
         unit="score", help="0 (poor) to 10 (excellent)."),
    dict(name="SleepQuality", say="On a scale of four to ten, how well do they sleep? Four is poor, ten is excellent.", label="Sleep quality", section="lifestyle", kind="number", min=4, max=10, step=0.1,
         unit="score", help="4 (poor) to 10 (excellent)."),

    dict(name="FamilyHistoryAlzheimers", label="Family history of Alzheimer's", section="history", kind="select", options=YES_NO,
         question="Has a parent, brother or sister had Alzheimer's or dementia?"),
    dict(name="CardiovascularDisease", label="Cardiovascular disease", section="history", kind="select", options=YES_NO,
         question="Has a doctor ever said they have heart disease (e.g. heart attack, angina, heart failure)?"),
    dict(name="Diabetes", label="Diabetes", section="history", kind="select", options=YES_NO,
         question="Has a doctor ever said they have diabetes (high blood sugar)?"),
    dict(name="Depression", label="Depression", section="history", kind="select", options=YES_NO,
         question="Have they been diagnosed with or treated for depression?"),
    dict(name="HeadInjury", label="History of head injury", section="history", kind="select", options=YES_NO,
         question="Have they ever had a serious head injury (e.g. knocked out, or needed hospital care)?"),
    dict(name="Hypertension", label="Hypertension", section="history", kind="select", options=YES_NO,
         question="Has a doctor ever said they have high blood pressure?"),

    dict(name="SystolicBP", say="What is the upper blood pressure number? For example one hundred and thirty. Say skip if you don't know.", label="Blood pressure: upper number (systolic)", section="clinical", kind="number",
         min=90, max=180, step=1, unit="mmHg", optional=True, group="bp",
         help="The first, larger number on a BP machine, e.g. 130 in 130/85. Range 90 to 180."),
    dict(name="DiastolicBP", say="What is the lower blood pressure number? For example eighty five. Say skip if you don't know.", label="Blood pressure: lower number (diastolic)", section="clinical", kind="number",
         min=60, max=120, step=1, unit="mmHg", optional=True, group="bp",
         help="The second, smaller number, e.g. 85 in 130/85. Range 60 to 120."),
    dict(name="CholesterolTotal", say="From the blood report, what is the total cholesterol? Say skip if you don't have it.", label="Total cholesterol", section="clinical", kind="number", min=150, max=300,
         step=0.1, unit="mg/dL", optional=True, group="blood", help="From a lipid profile blood report. Range 150 to 300."),
    dict(name="CholesterolLDL", say="What is the L D L cholesterol? Say skip if you don't have it.", label="LDL cholesterol", section="clinical", kind="number", min=50, max=200,
         step=0.1, unit="mg/dL", optional=True, group="blood", help="Range 50 to 200."),
    dict(name="CholesterolHDL", say="What is the H D L cholesterol? Say skip if you don't have it.", label="HDL cholesterol", section="clinical", kind="number", min=20, max=100,
         step=0.1, unit="mg/dL", optional=True, group="blood", help="Range 20 to 100."),
    dict(name="CholesterolTriglycerides", say="What is the triglycerides number? Say skip if you don't have it.", label="Triglycerides", section="clinical", kind="number", min=50, max=400,
         step=0.1, unit="mg/dL", optional=True, group="blood", help="Range 50 to 400."),

    dict(name="MMSE", say="What is the M M S E score from the doctor, out of thirty?", label="Memory and thinking score (MMSE)", section="cognitive", kind="number", min=0, max=30,
         step=0.1, unit="/ 30",
         help="Mini-Mental State Examination from a doctor or memory clinic, 0 to 30. Below 24 usually means impairment."),
    dict(name="FunctionalAssessment", say="What is the functional assessment score from the doctor, out of ten?", label="Everyday tasks score (functional assessment)", section="cognitive",
         kind="number", min=0, max=10, step=0.1, unit="/ 10",
         help="Clinician's functional assessment, 0 to 10. Lower means more difficulty with everyday tasks."),
    dict(name="ADL", say="What is the A D L, daily living score from the doctor, out of ten?", label="Self-care score (ADL)", section="cognitive", kind="number", min=0, max=10, step=0.1,
         unit="/ 10", help="Activities of daily living score, 0 to 10. Lower means more help needed with self-care."),

    dict(name="MemoryComplaints", label="Memory complaints", section="symptoms", kind="select", options=YES_NO,
         question="Do they (or people close to them) say their memory has been getting worse?"),
    dict(name="Forgetfulness", label="Forgetfulness", section="symptoms", kind="select", options=YES_NO,
         question="Do they often forget recent things?",
         help="For example: repeating the same question, forgetting a recent conversation, or misplacing keys or spectacles."),
    dict(name="Confusion", label="Confusion", section="symptoms", kind="select", options=YES_NO,
         question="Do they get confused about what is happening around them?",
         help="For example: not following a simple conversation, or mixing up people or events."),
    dict(name="Disorientation", label="Disorientation", section="symptoms", kind="select", options=YES_NO,
         question="Do they get lost or lose track of time?",
         help="For example: getting lost on a familiar road, or not knowing the day, month or year."),
    dict(name="DifficultyCompletingTasks", label="Difficulty completing tasks", section="symptoms", kind="select",
         options=YES_NO, question="Do they struggle to finish familiar tasks they used to do easily?",
         help="For example: following a recipe they know, using the TV remote, or counting money."),
    dict(name="PersonalityChanges", label="Personality changes", section="symptoms", kind="select", options=YES_NO,
         question="Has their personality changed?",
         help="For example: becoming withdrawn, suspicious, unusually anxious or easily upset."),
    dict(name="BehavioralProblems", label="Behavioural problems", section="symptoms", kind="select", options=YES_NO,
         question="Have there been changes in behaviour that are hard to manage?",
         help="For example: anger outbursts, agitation, wandering, or acting out of character."),
]
# The 11 inputs every assessment needs. All other fields may be left blank: the
# predictor then uses the typical training-set value and says so in the result.
CORE_INPUTS = ["Age", "Gender", "BMI", "FamilyHistoryAlzheimers", "MMSE", "FunctionalAssessment", "ADL",
               "MemoryComplaints", "BehavioralProblems", "Forgetfulness", "DifficultyCompletingTasks"]
for _f in FIELDS:
    if _f["name"] not in CORE_INPUTS:
        _f["optional"] = True

# ---------------------------------------------------------------------------
# Detailed ethnicity options
# ---------------------------------------------------------------------------
# The training data has only 4 ethnicity codes (0 Caucasian, 1 African American,
# 2 Asian, 3 Other). Users pick a more specific option; we store their exact
# choice for display and map it to the dataset code the model was trained on.
# The model cannot learn categories that were not in its training data.
ETHNICITY_GROUPS = {0: "Caucasian", 1: "African American", 2: "Asian", 3: "Other"}
ETHNICITY_OPTIONS = [
    # (key, label, dataset code, section in the dropdown)
    ("indian", "Indian", 2, "South Asian"),
    ("pakistani", "Pakistani", 2, "South Asian"),
    ("bangladeshi", "Bangladeshi", 2, "South Asian"),
    ("sri_lankan", "Sri Lankan", 2, "South Asian"),
    ("nepali", "Nepali / Bhutanese", 2, "South Asian"),
    ("south_asian_other", "Other South Asian", 2, "South Asian"),
    ("chinese", "Chinese", 2, "East and Southeast Asian"),
    ("japanese", "Japanese", 2, "East and Southeast Asian"),
    ("korean", "Korean", 2, "East and Southeast Asian"),
    ("southeast_asian", "Southeast Asian (e.g. Filipino, Vietnamese, Thai, Malay)", 2, "East and Southeast Asian"),
    ("asian_other", "Other Asian", 2, "East and Southeast Asian"),
    ("white", "White / Caucasian (e.g. European)", 0, "Other backgrounds"),
    ("black", "Black / African / African American / Caribbean", 1, "Other backgrounds"),
    ("middle_eastern", "Middle Eastern / North African", 3, "Other backgrounds"),
    ("hispanic", "Hispanic / Latino", 3, "Other backgrounds"),
    ("mixed", "Mixed / multiple backgrounds", 3, "Other backgrounds"),
    ("other", "Other / prefer not to say", 3, "Other backgrounds"),
]
ETHNICITY_BY_KEY = {k: (label, code) for k, label, code, _ in ETHNICITY_OPTIONS}
# Default detailed option for records that only have a dataset code (e.g. demo twins)
DEFAULT_KEY_FOR_CODE = {0: "white", 1: "black", 2: "asian_other", 3: "other"}


def ethnicity_display(inputs: dict) -> str:
    """'Indian (model group: Asian)' if a detailed choice was stored, else the dataset group."""
    if inputs.get("Ethnicity") is None:
        return "Not provided"
    group = ETHNICITY_GROUPS.get(int(inputs["Ethnicity"]), str(inputs["Ethnicity"]))
    key = inputs.get("EthnicityDetail")
    if key in ETHNICITY_BY_KEY:
        return f"{ETHNICITY_BY_KEY[key][0]} (model group: {group})"
    return group


FIELD_BY_NAME = {f["name"]: f for f in FIELDS}
assert set(FIELD_BY_NAME) == set(config.RAW_FEATURES), "form must cover every model input"


CHILD_MESSAGE = ("This screening is for adults only. Children do not get Alzheimer's disease; the conditions "
                 "sometimes called \"childhood Alzheimer's\" (such as Niemann-Pick type C or Sanfilippo syndrome) "
                 "are different rare genetic illnesses. If a child has memory loss or is losing skills, please see "
                 "a paediatrician or paediatric neurologist.")


def validate(form: dict) -> tuple[dict, dict]:
    """
    Return (clean values, errors).

    - Fields marked optional (everything outside CORE_INPUTS) may be left empty: they
      are stored as None and listed in clean["_not_provided"]; the predictor then
      uses the typical training-set value.
    - assessment_mode == "questions": MMSE, FunctionalAssessment and ADL are
      ESTIMATED from the home questionnaire (src/questionnaire.py).
      Otherwise (doctor's scores, API calls, what-if) they are entered as numbers.
    """
    from src import questionnaire as Q

    clean, errors, not_provided = {}, {}, []
    mode = form.get("assessment_mode", "scores")
    estimated = {}
    if mode == "questions":
        adl, adl_ans, e1 = Q.score_self_care(form)
        fa, fa_ans, e2 = Q.score_everyday_tasks(form)
        mmse, bc, e3 = Q.score_brain_check(form)
        errors.update(e1 | e2 | e3)
        estimated = {"ADL": adl, "FunctionalAssessment": fa, "MMSE": mmse}
        clean["_assessment"] = {"mode": "questions", "self_care_answers": adl_ans,
                                "everyday_task_answers": fa_ans, "brain_check": bc}
    form = dict(form)
    if form.get("bmi_mode") == "calc":
        # BMI calculated from height and weight typed in everyday units (src/body.py)
        from src.body import bmi_from
        body = bmi_from(form.get("height", ""), form.get("weight", ""), form.get("weight_unit", "kg"))
        if body["height_error"]:
            errors["height"] = body["height_error"]
        if body["weight_error"]:
            errors["weight"] = body["weight_error"]
        if body["bmi"] is not None:
            form["BMI"] = body["bmi"]
            clean["_body"] = {"height_cm": body["height_cm"], "weight_kg": body["weight_kg"],
                              "height_entered": form.get("height", ""), "weight_entered": form.get("weight", ""),
                              "weight_unit": form.get("weight_unit", "kg")}
        else:
            form["BMI"] = ""
    age_raw = str(form.get("Age", "")).strip()
    try:
        if age_raw and float(age_raw) < 18:
            errors["Age"] = CHILD_MESSAGE
    except ValueError:
        pass
    for f in FIELDS:
        name = f["name"]
        if name == "Age" and "Age" in errors:
            continue
        if name == "BMI" and form.get("bmi_mode") == "calc" and not form["BMI"]:
            continue  # the height/weight error already explains the problem
        if name in estimated:
            if estimated[name] is not None:
                clean[name] = estimated[name]
            continue
        if f["kind"] == "ethnicity" and form.get("EthnicityDetail"):
            # detailed choice from the web form -> store it and map to the dataset code
            key = form.get("EthnicityDetail")
            if key not in ETHNICITY_BY_KEY:
                errors["Ethnicity"] = "Choose one of the listed options."
            else:
                clean["EthnicityDetail"] = key
                clean["Ethnicity"] = ETHNICITY_BY_KEY[key][1]
            continue
        raw = form.get(name, "")
        if f.get("optional") and (raw is None or str(raw).strip() == ""):
            clean[name] = None
            not_provided.append(name)
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors[name] = "Please answer this question." if f["kind"] != "number" else "Please enter a number."
            continue
        if f["kind"] in ("select", "ethnicity"):
            if int(v) not in [o[0] for o in f["options"]] or v != int(v):
                errors[name] = "Choose one of the listed options."
                continue
            v = int(v)
        elif not (f["min"] <= v <= f["max"]):
            errors[name] = f"Must be between {f['min']:g} and {f['max']:g}."
            continue
        clean[name] = v
    # blood pressure needs both numbers (pulse pressure uses the pair)
    bp = [clean.get("SystolicBP"), clean.get("DiastolicBP")]
    if (bp[0] is None) != (bp[1] is None):
        errors["SystolicBP" if bp[0] is None else "DiastolicBP"] = "Enter both blood pressure numbers, or leave both empty."
    elif bp[0] is not None and bp[0] <= bp[1]:
        errors["SystolicBP"] = "The upper number should be larger than the lower number."
    clean["_not_provided"] = not_provided
    voice = form.get("voice_used") == "1"  # answered with voice help (spoken questions and answers)
    if mode != "questions":
        clean["_assessment"] = {"mode": "scores"}
    clean["_assessment"]["voice_used"] = voice
    return clean, errors


def missing_inputs(inputs: dict) -> list[str]:
    """Model inputs that were not provided (the predictor uses typical values for them)."""
    return [f for f in config.RAW_FEATURES if inputs.get(f) is None]


def display_value(name: str, value) -> str:
    """Readable value for a stored input (option label or number with unit)."""
    f = FIELD_BY_NAME[name]
    if value is None:
        return "Not provided"
    if f["kind"] in ("select", "ethnicity"):
        return dict(f["options"]).get(int(value), str(value))
    return f"{float(value):g} {f.get('unit', '')}".strip()
