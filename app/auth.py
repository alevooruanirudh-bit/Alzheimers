"""
Accounts: register, log in, log out.

- A "patient" account can hold several profiles (Digital Twins), for example yourself, your mother and your
  father, and can only see the profiles it holds. One profile is "selected" at a time; tests act on it.
- "researcher" accounts can see every twin (for project review and the viva demo).
- Passwords are stored only as salted hashes (werkzeug.security).
"""
import re
from functools import wraps

from flask import Blueprint, abort, flash, g, redirect, render_template, request, session, url_for

from src.twin import Patient, SessionLocal, User, find_user

auth = Blueprint("auth", __name__)
USERNAME_RE = re.compile(r"^[a-z0-9._-]{3,30}$")


def current_user() -> User | None:
    if "user" not in g:
        g.user = None
        if uid := session.get("user_id"):
            with SessionLocal() as s:
                g.user = s.get(User, uid)
    return g.user


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            flash("Please log in first.", "info")
            return redirect(url_for("auth.login", next=request.full_path))
        return view(*args, **kwargs)
    return wrapped


def researcher_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if not current_user().is_researcher:
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def owns(patient_id: int) -> bool:
    """True if the logged-in account holds this profile."""
    u = current_user()
    if u is None or not patient_id:
        return False
    with SessionLocal() as s:
        p = s.get(Patient, patient_id)
        return p is not None and p.owner_id == u.id


def can_view(patient_id: int) -> bool:
    """Accounts see only the profiles they hold; researchers see all."""
    u = current_user()
    return u is not None and (u.is_researcher or owns(patient_id))


def _safe_next(target: str | None) -> str | None:
    return target if target and target.startswith("/") and not target.startswith("//") else None


@auth.route("/register", methods=["GET", "POST"])
def register():
    errors, values = {}, {}
    if request.method == "POST":
        values = request.form
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        if not USERNAME_RE.match(username):
            errors["username"] = "Use 3 to 30 letters, numbers, dots, dashes or underscores (no spaces)."
        if len(password) < 6:
            errors["password"] = "Use at least 6 characters."
        elif password != request.form.get("password2", ""):
            errors["password2"] = "The two passwords do not match."
        with SessionLocal() as s:
            if not errors and find_user(s, username):
                errors["username"] = "This username is already taken."
            if not errors:
                u = User(username=username, role="patient", relation="self")
                u.set_password(password)
                s.add(u)
                s.commit()
                session.clear()
                session["user_id"] = u.id
                flash("Account created. Next, add the first profile: yourself, or the person you are checking.", "info")
                return redirect(url_for("main.profile_new"))
    return render_template("register.html", active="register", errors=errors, values=values)


@auth.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        with SessionLocal() as s:
            u = find_user(s, request.form.get("username", ""))
            if u and u.check_password(request.form.get("password", "")):
                session.clear()
                session["user_id"] = u.id
                return redirect(_safe_next(request.args.get("next")) or url_for("main.dashboard"))
        error = "Wrong username or password."
    return render_template("login.html", active="login", error=error)


@auth.route("/logout", methods=["POST", "GET"])
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("main.home"))


@auth.app_context_processor
def inject_user():
    """The logged-in account and its selected profile (shown in the menu on every page)."""
    u, selected = current_user(), None
    if u is not None and u.patient_id and not u.is_researcher:
        with SessionLocal() as s:
            p = s.get(Patient, u.patient_id)
            if p is not None and p.owner_id == u.id:
                selected = {"id": p.id, "name": p.name}
    return {"user": u, "selected_profile": selected}
