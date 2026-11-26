"""
Page routes and JSON APIs.

Flow for a person (one account can hold several profiles, e.g. yourself, your mother and your father):
  1. Register / log in            (app/auth.py)
  2. Profiles: add one per person; set up once, edit when something changes (e.g. weight).
     One profile is selected at a time and the tests below act on it.
  3. Level 1  Home check: lifestyle, health history, symptoms, daily life, brain check (repeat regularly)
  4. Level 2  Clinical report: blood pressure, blood tests, doctor's memory/daily-life scores (verifies Level 1)
     + Scan reports (MRI/CT/PET) kept in the record
  5. Dashboard: latest risk, trend over time, compare any two tests, full history, what-if

Every number shown comes from a live model prediction (src.predictor) or the saved
evaluation files in reports/ (written by src.evaluate).
"""
import json
import random
from datetime import date, datetime, time

import pandas as pd
from flask import (Blueprint, abort, flash, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

from app.auth import can_view, current_user, login_required, owns, researcher_required
from src import body, config, trends
from src import profile as P
from src import questionnaire as Q
from src.predictor import get_predictor
from src.schema import (CHILD_MESSAGE, ETHNICITY_BY_KEY, ETHNICITY_GROUPS, ETHNICITY_OPTIONS, FIELD_BY_NAME,
                        FIELDS, SECTIONS, display_value, ethnicity_display, validate)
from src.twin import (RELATION_LABELS, RELATIONS, UPLOAD_DIR, BodyMeasurement, Patient, Prediction, ScanReport,
                      SessionLocal, User, list_patients, profiles_of, record_visit)

bp = Blueprint("main", __name__)
RETEST_DAYS = 90            # suggested interval between home checks (a design choice, not medical advice)
SCAN_TYPES = ["MRI", "CT", "PET", "Other"]
ALLOWED_UPLOADS = {".pdf", ".jpg", ".jpeg", ".png"}
CLINICAL_FIELDS = ["SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL", "CholesterolHDL",
                   "CholesterolTriglycerides", "MMSE", "FunctionalAssessment", "ADL"]
DOCTOR_SCORES = ["MMSE", "FunctionalAssessment", "ADL"]


def _load_json(name: str) -> dict | None:
    p = config.REPORTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def _assessment_info(inputs: dict) -> dict:
    """How the memory/daily-life scores were obtained, and which values were missing."""
    a = inputs.get("_assessment") or {}
    mode = a.get("mode")
    if mode == "mixed":
        label = ("Doctor's scores for " + ", ".join(FIELD_BY_NAME[f]["label"] for f in a.get("doctor_scores", []))
                 + "; home estimates for the rest")
    else:
        label = {"questions": "Estimated from home questions and the brain check",
                 "scores": "Scores from a doctor or clinic"}.get(mode, "Scores from the research dataset record")
    missing = [FIELD_BY_NAME[f]["label"] for f in FIELD_BY_NAME if inputs.get(f) is None]
    return {"mode": mode, "label": label, "missing": missing, "brain_check": a.get("brain_check"),
            "voice_used": a.get("voice_used")}


def _own_patient(s) -> Patient | None:
    """The selected profile of the logged-in account (only if the account holds it)."""
    u = current_user()
    p = s.get(Patient, u.patient_id) if u and u.patient_id else None
    return p if p is not None and p.owner_id == u.id else None


def _select_profile(s, patient_id: int | None) -> None:
    u = s.get(User, current_user().id)
    u.patient_id = patient_id
    current_user().patient_id = patient_id


def _visible_prediction(s, prediction_id: int) -> Prediction:
    pred = s.get(Prediction, prediction_id) or abort(404)
    if not can_view(pred.patient_id):
        abort(403)
    return pred


def _form_context(**extra) -> dict:
    """Shared template data for forms that ask profile-type questions."""
    return dict(fields=FIELDS, eth_options=ETHNICITY_OPTIONS, eth_groups=ETHNICITY_GROUPS, eth_by_key=ETHNICITY_BY_KEY,
                eth_sections=list(dict.fromkeys(o[3] for o in ETHNICITY_OPTIONS)), child_message=CHILD_MESSAGE,
                height_units=body.HEIGHT_UNITS, weight_units=body.WEIGHT_UNITS, **extra)


# ---------------------------------------------------------------- home
@bp.route("/")
def home():
    return render_template("home.html", active="home", results=_load_json("results.json"),
                           valid=(_load_json("label_noise_eval.json") or {}).get("valid_labels"))


@bp.route("/me")
@login_required
def dashboard():
    u = current_user()
    if u.is_researcher:
        return redirect(url_for("main.twins"))
    with SessionLocal() as s:
        mine = profiles_of(s, u)
        active = _own_patient(s)
    if not mine:
        return redirect(url_for("main.profile_new"))
    if active is None and len(mine) > 1:
        return redirect(url_for("main.profiles"))
    return redirect(url_for("main.patient_view", patient_id=(active or mine[0]).id))


# ---------------------------------------------------------------- profiles (one per person)
@bp.route("/me/profiles")
@login_required
def profiles():
    u = current_user()
    if u.is_researcher:
        return redirect(url_for("main.twins"))
    with SessionLocal() as s:
        mine = profiles_of(s, u)
        cards = []
        for p in mine:
            latest = p.latest_prediction
            cards.append({"p": p, "relation": RELATION_LABELS.get(p.relation or "", "Profile"),
                          "age": int(P.age_on(p.profile)) if p.profile else None, "latest": latest,
                          "tests": len(p.predictions), "verdict": trends.summary(p)["since_previous"]})
        return render_template("profiles.html", active="profiles", cards=cards, active_id=u.patient_id)


@bp.route("/me/profiles/new", methods=["GET", "POST"])
@login_required
def profile_new():
    return _profile_form(new=True)


@bp.route("/me/profile", methods=["GET", "POST"])
@login_required
def profile():
    return _profile_form(new=False)


def _profile_form(new: bool):
    u = current_user()
    if u.is_researcher:
        abort(403)
    with SessionLocal() as s:
        patient = None if new else _own_patient(s)
        if not new and patient is None:
            return redirect(url_for("main.profile_new") if not profiles_of(s, u) else url_for("main.profiles"))
        errors = {}
        values = P.form_values(patient.profile) if patient and patient.profile else {}
        values["relation"] = (patient.relation if patient else None) or ""   # always asked for a new profile
        if request.method == "POST":
            prof, errors = P.validate_profile(request.form)
            relation = request.form.get("relation", "")
            if relation not in RELATION_LABELS:
                errors["relation"] = "Choose who this profile is for."
            if not errors:
                if patient is None:
                    patient = Patient(name=prof["name"], age=0, gender=0, ethnicity=0, education_level=0,
                                      family_history=0, source="Registered account", owner_id=u.id)
                    s.add(patient)
                    s.flush()
                    _select_profile(s, patient.id)
                patient.relation = relation
                if P.apply_to_patient(patient, prof):   # height/weight changed: keep the BMI history
                    s.add(BodyMeasurement(patient=patient, height_cm=prof["height_cm"], weight_kg=prof["weight_kg"],
                                          bmi=prof["bmi"], entered_as=f"{prof['height_text']}, {prof['weight_text']}"))
                s.commit()
                flash(f"Profile of {patient.name} saved. It will be used for every test from now on.", "info")
                return redirect(url_for("main.patient_view", patient_id=patient.id))
            values = request.form
            flash("Please correct the highlighted answers.", "error")
        return render_template("profile.html", active="profiles" if new else "profile", values=values, errors=errors,
                               is_new=patient is None or not patient.profile, adding=new, patient=patient,
                               relations=RELATIONS, **_form_context(history_fields=P.HISTORY_FIELDS))


@bp.route("/me/profiles/<int:patient_id>/select", methods=["POST"])
@login_required
def profile_select(patient_id: int):
    if not owns(patient_id):
        abort(403)
    with SessionLocal() as s:
        _select_profile(s, patient_id)
        s.commit()
    target = request.form.get("next", "")
    return redirect(target if target.startswith("/") and not target.startswith("//")
                    else url_for("main.patient_view", patient_id=patient_id))


@bp.route("/me/profiles/<int:patient_id>/delete", methods=["POST"])
@login_required
def profile_delete(patient_id: int):
    """Delete one profile with all its tests, BMI history and scan reports (cannot be undone)."""
    if not owns(patient_id):
        abort(403)
    with SessionLocal() as s:
        p = s.get(Patient, patient_id)
        name = p.name
        for sc in p.scans:
            if sc.file_name:
                (UPLOAD_DIR / sc.file_name).unlink(missing_ok=True)
        s.delete(p)
        if current_user().patient_id == patient_id:
            rest = [x for x in profiles_of(s, current_user()) if x.id != patient_id]
            _select_profile(s, rest[0].id if rest else None)
        s.commit()
    flash(f"The profile of {name} and all its records were deleted.", "info")
    return redirect(url_for("main.profiles"))


def _require_profile(s):
    patient = _own_patient(s)
    if patient is None or not patient.profile:
        flash("Please choose or add a profile first.", "info")
        return None
    return patient


# ---------------------------------------------------------------- Level 1: home check
@bp.route("/me/test/home", methods=["GET", "POST"])
@login_required
def home_test():
    with SessionLocal() as s:
        patient = _require_profile(s)
        if patient is None:
            return redirect(url_for("main.dashboard"))
        prof = patient.profile
        values, errors = {}, {}
        last_home = patient.latest_of_kind("home")
        if request.method == "POST":
            now = datetime.now()
            base = {k: v for k, v in P.model_inputs(prof, now).items() if not k.startswith("_")}
            clean, errors = validate({**base, **request.form, "assessment_mode": "questions"})
            if not errors:
                clean["_body"] = P.model_inputs(prof, now)["_body"]
                pred = record_visit(s, patient, clean, get_predictor(), recorded_at=now, kind="home")
                s.commit()
                return redirect(url_for("main.result", prediction_id=pred.id))
            values = {**request.form}
            flash("Please correct the highlighted answers.", "error")
        return render_template(
            "home_test.html", active="test", values=values, errors=errors, patient=patient, prof=prof,
            age_now=P.age_on(prof), start_step=_home_error_step(errors) if errors else 1, last_home=last_home,
            extras_open=any(FIELD_BY_NAME.get(k, {}).get("optional") for k in errors),
            self_care_items=Q.SELF_CARE_ITEMS, everyday_items=Q.EVERYDAY_TASK_ITEMS,
            help_answers=Q.HELP_ANSWERS, na_answer=Q.NA_ANSWER, word_sets=Q.WORD_SETS,
            wordset_idx=random.randrange(len(Q.WORD_SETS)), place_items=Q.PLACE_ITEMS,
            copy_sentence=Q.COPY_SENTENCE, **_form_context())


def _home_error_step(errors: dict) -> int:
    steps = []
    for key in errors:
        optional = FIELD_BY_NAME.get(key, {}).get("optional")
        steps.append(4 if optional else 2 if key.startswith(("adl_", "iadl_")) else 3 if key.startswith("bc_") else 1)
    return min(steps)


# ---------------------------------------------------------------- Level 2: clinical report
@bp.route("/me/test/clinical", methods=["GET", "POST"])
@login_required
def clinical_test():
    with SessionLocal() as s:
        patient = _require_profile(s)
        if patient is None:
            return redirect(url_for("main.dashboard"))
        last_home = patient.latest_of_kind("home")
        if last_home is None:
            flash("Please do a home check (Level 1) first. The clinical report is then used to verify it.", "info")
            return redirect(url_for("main.home_test"))
        values, errors = {"report_date": date.today().isoformat()}, {}
        if request.method == "POST":
            values = request.form
            try:
                report_day = date.fromisoformat(request.form.get("report_date", ""))
                if report_day > date.today():
                    errors["report_date"] = "The report date cannot be in the future."
            except ValueError:
                errors["report_date"] = "Please enter the report date."
                report_day = date.today()
            given = {f: request.form.get(f, "").strip() for f in CLINICAL_FIELDS if request.form.get(f, "").strip()}
            if not given:
                errors["_any"] = "Enter at least one value from the report."
            when = datetime.now() if report_day == date.today() else datetime.combine(report_day, time(12, 0))
            base = {k: v for k, v in last_home.inputs.items() if not k.startswith("_") and v is not None}
            base.update({k: v for k, v in P.model_inputs(patient.profile, when).items() if not k.startswith("_")})
            form = {**base, **given}
            for f in CLINICAL_FIELDS[:6]:
                form.setdefault(f, "")
            clean, verr = validate({**form, "assessment_mode": "scores"})
            errors.update({k: v for k, v in verr.items() if k in CLINICAL_FIELDS})
            if not errors:
                doctor = [f for f in DOCTOR_SCORES if f in given]
                clean["_assessment"] = {
                    "mode": "scores" if len(doctor) == 3 else "mixed" if doctor else "questions",
                    "doctor_scores": doctor, "based_on_home_test": last_home.id,
                    "home_test_date": last_home.recorded_at.isoformat(timespec="minutes"),
                    "brain_check": (last_home.inputs.get("_assessment") or {}).get("brain_check") if len(doctor) < 3 else None}
                clean["_body"] = P.model_inputs(patient.profile, when)["_body"]
                note = request.form.get("notes", "").strip() or None
                pred = record_visit(s, patient, clean, get_predictor(), recorded_at=when, note=note, kind="clinical")
                reported = {k: clean[k] for k in P.CLINIC_FIELDS if k in given}
                if reported and when >= last_home.recorded_at:   # newest numbers are kept on the profile
                    P.apply_to_patient(patient, {**patient.profile, **reported})
                s.commit()
                return redirect(url_for("main.result", prediction_id=pred.id))
            flash("Please correct the highlighted values.", "error")
        return render_template("clinical.html", active="test", values=values, errors=errors, patient=patient,
                               last_home=last_home, clinical_fields=[FIELD_BY_NAME[f] for f in CLINICAL_FIELDS])


# ---------------------------------------------------------------- scan reports
@bp.route("/me/scans/new", methods=["GET", "POST"])
@login_required
def scan_new():
    with SessionLocal() as s:
        patient = _require_profile(s)
        if patient is None:
            return redirect(url_for("main.dashboard"))
        values, errors = {"scan_date": date.today().isoformat()}, {}
        if request.method == "POST":
            values = request.form
            try:
                scan_day = date.fromisoformat(request.form.get("scan_date", ""))
            except ValueError:
                errors["scan_date"] = "Please enter the scan date."
            scan_type = request.form.get("scan_type", "")
            if scan_type not in SCAN_TYPES:
                errors["scan_type"] = "Choose the type of scan."
            upload = request.files.get("file")
            file_name = None
            if upload and upload.filename:
                ext = "." + upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
                if ext not in ALLOWED_UPLOADS:
                    errors["file"] = "Upload a PDF, JPG or PNG file."
                else:
                    file_name = f"{patient.id}_{datetime.now():%Y%m%d%H%M%S}_{secure_filename(upload.filename)}"
            if not errors:
                if file_name:
                    upload.save(UPLOAD_DIR / file_name)
                s.add(ScanReport(patient=patient, scan_date=scan_day, scan_type=scan_type,
                                 facility=request.form.get("facility", "").strip() or None,
                                 findings=request.form.get("findings", "").strip() or None, file_name=file_name))
                s.commit()
                flash("Scan report saved to the record.", "info")
                return redirect(url_for("main.patient_view", patient_id=patient.id) + "#history")
        return render_template("scan.html", active="test", values=values, errors=errors, scan_types=SCAN_TYPES)


@bp.route("/scans/<int:scan_id>/file")
@login_required
def scan_file(scan_id: int):
    with SessionLocal() as s:
        scan = s.get(ScanReport, scan_id) or abort(404)
        if not can_view(scan.patient_id) or not scan.file_name:
            abort(403)
        return send_from_directory(UPLOAD_DIR, scan.file_name)


# ---------------------------------------------------------------- dashboard (one person's Digital Twin)
@bp.route("/patients/<int:patient_id>")
@login_required
def patient_view(patient_id: int):
    if not can_view(patient_id):
        abort(403)
    with SessionLocal() as s:
        p = s.get(Patient, patient_id) or abort(404)
        if p.owner_id == current_user().id and current_user().patient_id != p.id:
            _select_profile(s, p.id)   # opening a profile selects it, so the next test is for this person
            s.commit()
        latest = p.latest_prediction
        summary = trends.summary(p)
        recs = summary["records"]
        compare = None
        if len(recs) >= 2:  # any two tests can be compared; default: previous vs latest
            ids = [r["prediction_id"] for r in recs]
            a = request.args.get("a", type=int, default=ids[-2])
            b = request.args.get("b", type=int, default=ids[-1])
            ra = next((r for r in recs if r["prediction_id"] == a), recs[-2])
            rb = next((r for r in recs if r["prediction_id"] == b), recs[-1])
            if ra["when"] > rb["when"]:
                ra, rb = rb, ra
            compare = trends.compare(ra, rb)
        timeline = ([{"type": "test", "when": r.observation.recorded_at, "pred": r} for r in p.predictions]
                    + [{"type": "scan", "when": datetime.combine(sc.scan_date, time(0)), "scan": sc} for sc in p.scans]
                    + [{"type": "body", "when": m.recorded_at, "m": m} for m in p.body_measurements])
        timeline.sort(key=lambda e: e["when"], reverse=True)          # newest first
        chart = [{"date": r["when"].strftime("%Y-%m-%d"), "probability": r["probability"], "kind": r["kind"],
                  "simulated": r["simulated"], "MMSE": r["MMSE"], "FA": r["FunctionalAssessment"], "ADL": r["ADL"]}
                 for r in recs]
        bmi_chart = [{"date": m.recorded_at.strftime("%Y-%m-%d"), "bmi": m.bmi} for m in p.body_measurements]
        ctx = {}
        if latest:
            obs = latest.observation
            medians = get_predictor().medians
            ctx = dict(obs=obs, assessment=_assessment_info(obs.inputs),
                       sliders=[FIELD_BY_NAME[f] for f in config.CHANGEABLE_FEATURES],
                       slider_start={f: (medians[f] if obs.inputs.get(f) is None else obs.inputs[f])
                                     for f in config.CHANGEABLE_FEATURES})
        last_home = p.latest_of_kind("home")
        levels = {"home": sum(1 for o in p.observations if o.kind_label == "home"),
                  "clinical": sum(1 for o in p.observations if o.kind_label == "clinical"), "scans": len(p.scans),
                  "last_home": last_home.recorded_at if last_home else None,
                  "due": (not last_home) or (datetime.now() - last_home.recorded_at).days >= RETEST_DAYS}
        prof = p.profile
        prof_rows = []
        if prof:
            prof_rows = [("Age", f"{int(P.age_on(prof))} years" + (" (approximate)" if prof.get("birth_date_approx") else "")),
                         ("Gender", display_value("Gender", prof["Gender"])),
                         ("Ethnicity", ethnicity_display(prof)), ("Education", display_value("EducationLevel", prof["EducationLevel"])),
                         ("Height", prof.get("height_text", "").split(" = ")[0] + (" (assumed)" if prof.get("height_assumed") else "")),
                         ("Weight", prof.get("weight_text", "").split(" = ")[0]), ("BMI", f"{prof['bmi']} ({body.bmi_category(prof['bmi']).split(' (')[0]})")]
            prof_rows += [(FIELD_BY_NAME[f]["label"], display_value(f, prof[f])) for f in P.HISTORY_FIELDS]
        owner = s.get(User, p.owner_id) if p.owner_id else None
        return render_template("dashboard.html", active="dashboard", p=p, prof=prof, prof_rows=prof_rows,
                               latest=latest, summary=summary, compare=compare, recs=recs, timeline=timeline,
                               chart=chart, bmi_chart=bmi_chart, levels=levels, retest_days=RETEST_DAYS,
                               is_owner=p.owner_id == current_user().id, owner=owner,
                               relation=RELATION_LABELS.get(p.relation or ""), **ctx)


@bp.route("/twins")
@researcher_required
def twins():
    with SessionLocal() as s:
        users = {u.id: u.username for u in s.query(User)}
        owners = {p.id: users.get(p.owner_id) for p in s.query(Patient) if p.owner_id}
        patients = list_patients(s)
        ages = {p.id: int(P.age_on(p.profile)) if p.profile else int(p.latest_observation.inputs["Age"]) if p.observations else None
                for p in patients}
        relations = {p.id: RELATION_LABELS.get(p.relation or "") for p in patients}
        return render_template("twins.html", active="twins", patients=patients, owners=owners, ages=ages, relations=relations)


@bp.route("/twins/<int:patient_id>")
def twin_detail(patient_id: int):  # older links
    return redirect(url_for("main.patient_view", patient_id=patient_id))


@bp.route("/predict")
def predict():  # older links: the assessment now lives in the logged-in area
    return redirect(url_for("main.home_test"))


# ---------------------------------------------------------------- result and explanation
@bp.route("/result/<int:prediction_id>")
@login_required
def result(prediction_id: int):
    with SessionLocal() as s:
        pred = _visible_prediction(s, prediction_id)
        res = get_predictor().predict(pred.observation.inputs)
        return render_template("result.html", active="dashboard", pred=pred, res=res, patient=pred.patient,
                               assessment=_assessment_info(pred.observation.inputs), brain_domains=Q.BRAIN_DOMAINS)


@bp.route("/explain/<int:prediction_id>")
@login_required
def explain(prediction_id: int):
    with SessionLocal() as s:
        pred = _visible_prediction(s, prediction_id)
        predictor = get_predictor()
        inputs = pred.observation.inputs
        res = predictor.predict(inputs)
        return render_template("explain.html", active="dashboard", pred=pred, res=res, patient=pred.patient,
                               attention=predictor.attention(inputs)[:15],
                               plain_shap=predictor.plain_xgb_shap(inputs),
                               global_info=_load_json("explainability.json"))


@bp.route("/explain/<int:prediction_id>/<kind>.png")
@login_required
def explain_plot(prediction_id: int, kind: str):
    with SessionLocal() as s:
        pred = _visible_prediction(s, prediction_id)
        predictor = get_predictor()
        res = predictor.predict(pred.observation.inputs)
    if kind == "waterfall":
        png = predictor.waterfall_png(res)
    elif kind == "force":
        png = predictor.force_png(res)
    else:
        abort(404)
    return png, 200, {"Content-Type": "image/png", "Cache-Control": "no-store"}


# ---------------------------------------------------------------- APIs
@bp.route("/api/whatif", methods=["POST"])
@login_required
def whatif():
    """Recompute risk with changed modifiable factors; non-modifiable ones stay fixed."""
    data = request.get_json(force=True)
    pid = int(data.get("patient_id", 0))
    if not can_view(pid):
        abort(403)
    with SessionLocal() as s:
        p = s.get(Patient, pid) or abort(404)
        base_inputs = dict(p.latest_observation.inputs)
        current = p.latest_prediction.probability
    changes = {k: v for k, v in (data.get("changes") or {}).items() if k in config.CHANGEABLE_FEATURES}
    clean, errors = validate({**base_inputs, **changes})
    if errors:
        return jsonify({"errors": errors}), 400
    res = get_predictor().predict(clean, explain=False)
    return jsonify({"probability": res["probability"], "risk_level": res["risk_level"],
                    "current_probability": current, "delta": res["probability"] - current})


@bp.route("/api/height")
def height_api():
    """Height from unit fields (live preview) or from free/spoken text (voice help)."""
    if request.args.get("text"):
        return jsonify(body.spoken_height(request.args["text"]))
    cm, text, err = body.height_from_units(request.args.get("unit", "ft_in"), request.args.get("value"),
                                           request.args.get("ft"), request.args.get("inch"))
    kg, wtext, werr = body.weight_from_units(request.args.get("weight_unit", "kg"), request.args.get("weight"))
    bmi = round(kg / (cm / 100) ** 2, 1) if cm and kg else None
    return jsonify({"cm": cm, "text": text, "error": err, "kg": kg, "weight_text": wtext, "weight_error": werr,
                    "bmi": bmi, "category": body.bmi_category(bmi)})


@bp.route("/api/voice-log", methods=["POST"])
def voice_log():
    """Local troubleshooting log of microphone events (what the browser heard, and when).
    Written to database/voice_debug.log on this computer only; delete the file to clear it."""
    entry = request.get_json(silent=True) or {}
    log = config.DATABASE_DIR / "voice_debug.log"
    if log.exists() and log.stat().st_size > 2_000_000:
        log.write_text("")
    with log.open("a") as fh:
        fh.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"),
                             "agent": request.headers.get("User-Agent", "")[:120], **entry})[:4000] + "\n")
    return ("", 204)


# ---------------------------------------------------------------- performance
_DATA_SUMMARY = None


def _data_summary() -> dict | None:
    """How much data was used, and how it was split (from the saved split indices; computed once)."""
    global _DATA_SUMMARY
    if _DATA_SUMMARY is None and config.RAW_CSV.exists() and config.SPLIT_FILE.exists():
        df = pd.read_csv(config.RAW_CSV)
        split = json.loads(config.SPLIT_FILE.read_text())
        parts = {}
        for name in ("train", "test"):
            d = df.loc[split[name]]
            parts[name] = {"n": len(d), "ad": int(d["Diagnosis"].sum()), "no_ad": int((d["Diagnosis"] == 0).sum())}
        n_train = parts["train"]["n"]
        _DATA_SUMMARY = {"n": len(df), "ad": int(df["Diagnosis"].sum()), "no_ad": int((df["Diagnosis"] == 0).sum()),
                         "n_features": len(config.RAW_FEATURES), "age_min": int(df["Age"].min()), "age_max": int(df["Age"].max()),
                         "train": parts["train"], "test": parts["test"], "folds": config.N_FOLDS,
                         "fold_fit": round(n_train * (config.N_FOLDS - 1) / config.N_FOLDS), "fold_val": round(n_train / config.N_FOLDS)}
    return _DATA_SUMMARY


@bp.route("/performance")
def performance():
    results = _load_json("results.json")
    if results is None:
        return render_template("performance.html", active="performance", results=None)
    comparison = pd.read_csv(config.REPORTS_DIR / "model_comparison.csv").to_dict(orient="records")
    return render_template("performance.html", active="performance", results=results,
                           comparison=comparison, training=_load_json("training_summary.json"),
                           pretest=_load_json("pretest_results.json"), explain_info=_load_json("explainability.json"),
                           data=_data_summary(), noise=_load_json("label_noise_eval.json"),
                           noise_cut=_load_json("label_noise_cutoff.json"))


@bp.route("/figures/<path:name>")
def figure(name: str):
    return send_from_directory(config.FIGURES_DIR, name)
