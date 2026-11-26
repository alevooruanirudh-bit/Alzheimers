"""
Demo and researcher accounts:  python -m src.accounts

Safe to run any time: it never deletes data. It
  - moves lifestyle and clinic numbers from the latest test into older profiles,
  - gives every twin that has no profile a profile built from its latest test
    (height is not in the research data, so 165 cm is assumed and the weight is
    chosen to keep the recorded BMI; the profile page says so),
  - creates the accounts below if they do not exist yet.

  researcher / research123   sees every Digital Twin (for project review)
  demo       / demo123       Demo patient 01 (declining over simulated visits)
  demo2      / demo123       Demo patient 04 (improving over simulated visits)

These are local research-demo passwords. Change or remove them before any real use.
"""
from src.profile import profile_from_inputs, stored_from_inputs
from src.twin import Patient, SessionLocal, User, find_user, init_db

DEMO_ACCOUNTS = [("researcher", "research123", "researcher", None),
                 ("demo", "demo123", "patient", "Demo patient 01"),
                 ("demo2", "demo123", "patient", "Demo patient 04")]


def ensure_profiles(s) -> int:
    n = 0
    for p in s.query(Patient).all():
        if not p.profile and p.latest_observation:
            p.profile = profile_from_inputs(p.name, p.latest_observation.inputs)
            n += 1
        elif p.profile and "SleepQuality" not in p.profile and p.latest_observation:
            # older profiles: lifestyle and clinic numbers were only on the test records
            p.profile = {**p.profile, **stored_from_inputs(p.latest_observation.inputs)}
            n += 1
    return n


def ensure_demo_accounts(s) -> list[str]:
    created = []
    for username, password, role, patient_name in DEMO_ACCOUNTS:
        if find_user(s, username):
            continue
        patient = s.query(Patient).filter(Patient.name == patient_name).first() if patient_name else None
        if patient_name and patient is None:
            continue  # demo twins not seeded
        u = User(username=username, role=role, relation="self", patient_id=patient.id if patient else None)
        u.set_password(password)
        s.add(u)
        s.flush()
        if patient:
            patient.owner_id, patient.relation = u.id, "self"
        created.append(username)
    return created


def main() -> None:
    init_db()
    with SessionLocal() as s:
        n = ensure_profiles(s)
        created = ensure_demo_accounts(s)
        s.commit()
    print(f"Profiles added to {n} existing twins; accounts created: {created or 'none (already exist)'}")


if __name__ == "__main__":
    main()
