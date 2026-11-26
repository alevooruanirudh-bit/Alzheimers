"""
Web application entry point:  python app/app.py   ->  http://127.0.0.1:5050

Loads the saved models once (no retraining) and serves the dashboard.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[0] = str(ROOT)  # make `src` and the `app` package importable

from flask import Flask  # noqa: E402

from app.auth import auth  # noqa: E402
from app.routes import bp  # noqa: E402
from src import config  # noqa: E402
from src.twin import init_db  # noqa: E402


def create_app() -> Flask:
    flask_app = Flask(__name__, template_folder="templates", static_folder="static")
    # session-signing key: generated once and kept in database/ (never committed or hard-coded)
    key_file = config.DATABASE_DIR / ".secret_key"
    if not key_file.exists():
        import secrets
        key_file.write_text(secrets.token_hex(32))
    flask_app.config.update(SECRET_KEY=key_file.read_text().strip(), MAX_CONTENT_LENGTH=10 * 1024 * 1024,
                            SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                            TEMPLATES_AUTO_RELOAD=True)
    init_db()
    flask_app.register_blueprint(auth)
    flask_app.register_blueprint(bp)

    @flask_app.template_filter("pct")
    def pct(v, digits=1):
        return f"{100 * float(v):.{digits}f}%"

    @flask_app.url_defaults
    def static_version(endpoint, values):
        # ?v=<last change> on static files, so browsers load a new voice.js / app.css straight after an update
        if endpoint == "static" and "filename" in values:
            f = Path(flask_app.static_folder) / values["filename"]
            if f.exists():
                values["v"] = int(f.stat().st_mtime)

    @flask_app.context_processor
    def globals_():
        return {"RISK_LOW_MAX": config.RISK_LOW_MAX, "RISK_HIGH_MIN": config.RISK_HIGH_MIN}

    return flask_app


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))  # 5000 is taken by AirPlay on macOS
    create_app().run(host="127.0.0.1", port=port, debug=False)
