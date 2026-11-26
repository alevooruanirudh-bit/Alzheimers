"""
Digital Twin persistence layer (SQLite + SQLAlchemy).

Every person has one Digital Twin: a profile plus every test they ever did, each
stored with a timestamp so changes can be followed over time.

  users              login accounts; one account can hold several profiles (e.g. yourself, your mother and
                     your father); "researcher" accounts can see everyone
  patients           the twin: profile (name, age, height/weight, history...), owner account, relation, demo flag
  body_measurements  every height/weight/BMI update from the profile, timestamped
  observations       every test (home check or clinical report) with all model inputs
  predictions        risk probability, level, model version, timestamp, top SHAP factors
  scan_reports       brain scan reports (MRI/CT/PET) kept in the record (not used by the model)
"""
from datetime import date, datetime

from sqlalchemy import (JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine, inspect, select, text)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker
from werkzeug.security import check_password_hash, generate_password_hash

from src import config


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(60), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="patient")      # patient | researcher
    relation: Mapped[str] = mapped_column(String(20), default="self")     # older field, no longer asked
    # the profile currently selected in this account (tests and "My profile" act on it)
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("patients.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    patient: Mapped["Patient | None"] = relationship(foreign_keys=[patient_id])

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)   # salted hash, never the password itself

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def is_researcher(self) -> bool:
        return self.role == "researcher"


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    age: Mapped[float] = mapped_column(Float)                 # age when the twin was created
    gender: Mapped[int] = mapped_column(Integer)
    ethnicity: Mapped[int] = mapped_column(Integer)
    education_level: Mapped[int] = mapped_column(Integer)
    family_history: Mapped[int] = mapped_column(Integer)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String(200), default="Entered via web form")
    dataset_label: Mapped[int | None] = mapped_column(Integer, nullable=True)  # reference only (demo twins)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    # Profile: stable personal details asked once and edited when they change (see src/profile.py)
    profile: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # the account that holds this profile, and who the person is to the account holder
    # (plain integer, not a foreign key, so users <-> patients does not become a circular constraint)
    owner_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    relation: Mapped[str | None] = mapped_column(String(30), nullable=True)   # see RELATIONS

    observations: Mapped[list["Observation"]] = relationship(
        back_populates="patient", order_by="Observation.recorded_at", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="patient", order_by="Prediction.created_at", cascade="all, delete-orphan")
    body_measurements: Mapped[list["BodyMeasurement"]] = relationship(
        back_populates="patient", order_by="BodyMeasurement.recorded_at", cascade="all, delete-orphan")
    scans: Mapped[list["ScanReport"]] = relationship(
        back_populates="patient", order_by="ScanReport.scan_date", cascade="all, delete-orphan")

    @property
    def latest_prediction(self) -> "Prediction | None":
        return max(self.predictions, key=lambda p: (p.observation.recorded_at, p.id)) if self.predictions else None

    @property
    def latest_observation(self) -> "Observation | None":
        return self.observations[-1] if self.observations else None

    def latest_of_kind(self, kind: str) -> "Observation | None":
        obs = [o for o in self.observations if o.kind_label == kind]
        return obs[-1] if obs else None


RELATIONS = [("self", "Myself"), ("mother", "Mother"), ("father", "Father"), ("spouse", "Husband / wife / partner"),
             ("grandparent", "Grandparent"), ("relative", "Other relative"), ("other", "Someone I care for")]
RELATION_LABELS = dict(RELATIONS)


class Observation(Base):
    """One test: a home check (questionnaire + brain check) or a clinical report."""
    __tablename__ = "observations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    inputs: Mapped[dict] = mapped_column(JSON)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    kind: Mapped[str | None] = mapped_column(String(20), nullable=True)   # home | clinical (None = older record)

    patient: Mapped[Patient] = relationship(back_populates="observations")
    prediction: Mapped["Prediction"] = relationship(back_populates="observation", uselist=False)

    @property
    def kind_label(self) -> str:
        """home / clinical; older records are classified by how their scores were obtained."""
        if self.kind:
            return self.kind
        mode = (self.inputs.get("_assessment") or {}).get("mode")
        return "home" if mode == "questions" else "clinical"


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    observation_id: Mapped[int] = mapped_column(ForeignKey("observations.id"))
    probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(10))
    predicted_label: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(60))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    top_factors: Mapped[list] = mapped_column(JSON)

    patient: Mapped[Patient] = relationship(back_populates="predictions")
    observation: Mapped[Observation] = relationship(back_populates="prediction")


class BodyMeasurement(Base):
    __tablename__ = "body_measurements"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    height_cm: Mapped[float] = mapped_column(Float)
    weight_kg: Mapped[float] = mapped_column(Float)
    bmi: Mapped[float] = mapped_column(Float)
    entered_as: Mapped[str | None] = mapped_column(String(80), nullable=True)   # e.g. "5 ft 4 in, 150 lb"

    patient: Mapped[Patient] = relationship(back_populates="body_measurements")


class ScanReport(Base):
    __tablename__ = "scan_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    scan_date: Mapped[date] = mapped_column(Date)
    scan_type: Mapped[str] = mapped_column(String(40))
    facility: Mapped[str | None] = mapped_column(String(160), nullable=True)
    findings: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_name: Mapped[str | None] = mapped_column(String(200), nullable=True)   # stored under database/uploads/
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    patient: Mapped[Patient] = relationship(back_populates="scans")


engine = create_engine(config.DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
UPLOAD_DIR = config.DATABASE_DIR / "uploads"

# columns added after the first version: (table, column, SQL type)
_NEW_COLUMNS = [("patients", "profile", "JSON"), ("patients", "updated_at", "DATETIME"),
                ("observations", "kind", "VARCHAR(20)"), ("patients", "owner_id", "INTEGER"),
                ("patients", "relation", "VARCHAR(30)")]


def init_db(reset: bool = False) -> None:
    """Create missing tables and add new columns to existing ones, keeping all existing rows."""
    if reset:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, column, sql_type in _NEW_COLUMNS:
            if column not in {c["name"] for c in insp.get_columns(table)}:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))
        # accounts from the one-profile version: their profile becomes the first profile they own
        conn.execute(text("UPDATE patients SET owner_id = (SELECT u.id FROM users u WHERE u.patient_id = patients.id), "
                          "relation = COALESCE(relation, 'self') "
                          "WHERE owner_id IS NULL AND EXISTS (SELECT 1 FROM users u WHERE u.patient_id = patients.id)"))
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Service functions
# ---------------------------------------------------------------------------
def create_patient(s: Session, name: str, inputs: dict, is_demo: bool = False,
                   source: str = "Entered via web form", dataset_label: int | None = None) -> Patient:
    p = Patient(name=name, age=inputs["Age"], gender=int(inputs["Gender"]), ethnicity=int(inputs["Ethnicity"]),
                education_level=int(inputs["EducationLevel"]),
                family_history=int(inputs["FamilyHistoryAlzheimers"]),
                is_demo=is_demo, source=source, dataset_label=dataset_label)
    s.add(p)
    s.flush()
    return p


def record_visit(s: Session, patient: Patient, inputs: dict, predictor, recorded_at: datetime | None = None,
                 is_simulated: bool = False, note: str | None = None, kind: str | None = None) -> Prediction:
    """Store a new test (observation), run the model on it, store the prediction."""
    obs = Observation(patient=patient, inputs=inputs, recorded_at=recorded_at or datetime.now(),
                      is_simulated=is_simulated, note=note, kind=kind)
    s.add(obs)
    s.flush()
    res = predictor.predict(inputs)
    pred = Prediction(patient=patient, observation=obs, probability=res["probability"],
                      risk_level=res["risk_level"], predicted_label=res["predicted_label"],
                      model_version=res["model_version"], top_factors=predictor.top_factors_json(res))
    s.add(pred)
    s.flush()
    return pred


def list_patients(s: Session) -> list[Patient]:
    """All twins, most recently assessed first (latest visit date, then newest twin)."""
    patients = list(s.scalars(select(Patient)))
    return sorted(patients, key=lambda p: (p.latest_observation.recorded_at if p.observations else p.created_at,
                                           p.created_at, p.id), reverse=True)


def profiles_of(s: Session, user: User) -> list[Patient]:
    """The profiles an account holds, in the order they were added."""
    return list(s.scalars(select(Patient).where(Patient.owner_id == user.id).order_by(Patient.created_at, Patient.id)))


def find_user(s: Session, username: str) -> User | None:
    return s.scalar(select(User).where(User.username == username.strip().lower()))
