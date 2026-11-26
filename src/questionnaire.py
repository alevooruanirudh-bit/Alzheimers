"""
Plain-language home questionnaire that ESTIMATES the three clinical scores the
model needs, so people without a clinic visit can still be screened.

  Self-care questions (6)         -> estimated ADL score (0-10)
  Everyday-task questions (8)     -> estimated Functional Assessment score (0-10)
  Quick brain check (11 tasks)    -> estimated memory-and-thinking score on the MMSE scale (0-30)

The items follow well-known domains (basic self-care as in the Katz index,
everyday tasks as in the Lawton scale, and the usual memory/orientation/
attention/language domains of brief cognitive screens), written in our own
words. They are NOT the copyrighted MMSE and have NOT been validated against
the dataset's scores: the model's reported accuracy applies to clinician
scores, and estimates from these questions are less certain.

Scoring happens on the server, so it is transparent and testable. Only the
scores are stored, never the typed answers (e.g. the person's city).
"""
import re
from difflib import SequenceMatcher
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Daily living
# ---------------------------------------------------------------------------
HELP_ANSWERS = [("2", "Does it without help"), ("1", "Needs some help"), ("0", "Cannot do it / needs full help")]
NA_ANSWER = ("na", "Never did this task")

SELF_CARE_ITEMS = [  # -> ADL
    ("adl_bathing", "Bathing or washing themselves"),
    ("adl_dressing", "Getting dressed (choosing clothes, buttons, zips)"),
    ("adl_toilet", "Using the toilet and cleaning themselves"),
    ("adl_moving", "Getting in and out of bed or a chair"),
    ("adl_continence", "Controlling bladder and bowels"),
    ("adl_eating", "Eating a meal (once it is served)"),
]
EVERYDAY_TASK_ITEMS = [  # -> Functional Assessment
    ("iadl_phone", "Using the phone (finding a number, making a call)"),
    ("iadl_shopping", "Buying everyday things (groceries, vegetables, medicines)"),
    ("iadl_cooking", "Preparing a simple meal or tea"),
    ("iadl_housework", "Light housework (tidying, washing dishes)"),
    ("iadl_laundry", "Washing or arranging their own clothes"),
    ("iadl_travel", "Travelling alone (bus, auto, taxi, walking to a nearby place)"),
    ("iadl_medicines", "Taking their medicines in the right dose at the right time"),
    ("iadl_money", "Handling money (counting change, paying bills)"),
]


def _score_items(form: dict, items, allow_na: bool) -> tuple[float | None, dict, dict]:
    """0-10 score = points / maximum points of the answered (non-'na') items."""
    answers, errors, points, maximum = {}, {}, 0, 0
    for key, _ in items:
        a = form.get(key, "")
        if a not in {"0", "1", "2"} | ({"na"} if allow_na else set()):
            errors[key] = "Please choose an answer."
            continue
        answers[key] = a
        if a != "na":
            points += int(a)
            maximum += 2
    if errors:
        return None, answers, errors
    if maximum == 0:
        return None, answers, {items[0][0]: "Answer at least one task the person used to do."}
    return round(10 * points / maximum, 2), answers, {}


def score_self_care(form: dict):
    return _score_items(form, SELF_CARE_ITEMS, allow_na=False)


def score_everyday_tasks(form: dict):
    return _score_items(form, EVERYDAY_TASK_ITEMS, allow_na=True)


# ---------------------------------------------------------------------------
# Quick brain check (30 points)
# ---------------------------------------------------------------------------
WORD_SETS = [("river", "lemon", "chair"), ("market", "tiger", "candle"),
             ("window", "mango", "drum"), ("bridge", "sugar", "shoe")]
COPY_SENTENCE = "The children are playing in the park"
TAP_ORDER = ["triangle", "big-circle", "small-square"]   # instruction shown in the app
READ_CORRECT = "green"                                    # card says: CHOOSE THE WORD GREEN
SHAPE_CORRECT = "b"                                       # the matching figure
PLACE_ITEMS = [("country", "Which country are you in?"), ("state", "Which state are you in?"),
               ("city", "Which city, town or village are you in?"),
               ("area", "Which area or neighbourhood is this?"),
               ("building", "What kind of place is this (home, hospital, clinic, office...)?")]
DAY_PHASES = {"morning": (5, 12), "afternoon": (12, 17), "evening": (17, 21), "night": (21, 29)}

BRAIN_DOMAINS = [  # (key, readable name, max points)
    ("date", "Knowing today's date and time", 5),
    ("place", "Knowing where they are", 5),
    ("words_now", "Repeating 3 words", 3),
    ("counting", "Counting backwards by 7", 5),
    ("recall", "Remembering the 3 words later", 3),
    ("naming", "Naming 2 pictures", 2),
    ("copy", "Typing a sentence exactly", 1),
    ("steps", "Following a 3-step instruction", 3),
    ("reading", "Reading and following an instruction", 1),
    ("writing", "Writing a sentence", 1),
    ("shape", "Matching a shape", 1),
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", str(text).lower())).strip()


def _same_word(a: str, w: str) -> bool:
    """Typed word matches the target allowing a plural or a one-letter slip ("shoes", "mangoe")."""
    a = a[:-1] if a.endswith("s") and not w.endswith("s") else a
    return a == w or (len(w) >= 5 and SequenceMatcher(None, a, w).ratio() >= 0.85)


def _words_found(typed: list[str], target: tuple) -> int:
    said = [_norm(t) for t in typed if _norm(t)]
    return sum(1 for w in target if any(_same_word(a, w) for a in said))


def _int(v):
    try:
        return int(str(v).strip())
    except ValueError:
        return None


def _phase_ok(phase: str, now: datetime) -> bool:
    """Correct if the chosen part of the day fits the time, with 45 minutes of tolerance."""
    if phase not in DAY_PHASES:
        return False
    lo, hi = DAY_PHASES[phase]
    for t in (now, now - timedelta(minutes=45), now + timedelta(minutes=45)):
        h = t.hour + (24 if t.hour < 5 else 0)
        if lo <= h < hi:
            return True
    return False


def score_brain_check(form: dict) -> tuple[float | None, dict, dict]:
    """Return (estimated score 0-30, points per domain, errors)."""
    if form.get("bc_completed") != "1":
        return None, {}, {"bc_completed": "Please complete the quick brain check (about 3 minutes)."}
    try:
        now = datetime.fromisoformat(form.get("bc_client_now", ""))  # the person's local time
    except ValueError:
        now = datetime.now()
    s = {}
    s["date"] = sum([
        _int(form.get("bc_year")) == now.year,
        _int(form.get("bc_month")) == now.month,
        _int(form.get("bc_day")) == now.day,
        _int(form.get("bc_weekday")) == now.weekday(),
        _phase_ok(form.get("bc_dayphase", ""), now),
    ])
    # Place answers cannot be checked by a computer: a helper (or the person) ticks each correct answer.
    s["place"] = sum(1 for key, _ in PLACE_ITEMS
                     if str(form.get(f"bc_place_{key}", "")).strip() and form.get(f"bc_place_ok_{key}") == "1")
    idx = _int(form.get("bc_wordset"))
    words = WORD_SETS[idx] if idx is not None and 0 <= idx < len(WORD_SETS) else WORD_SETS[0]
    s["words_now"] = _words_found([form.get(f"bc_now_{i}", "") for i in (1, 2, 3)], words)
    s["recall"] = _words_found([form.get(f"bc_recall_{i}", "") for i in (1, 2, 3)], words)
    # Serial sevens: each answer scores if it is 7 less than the previous answer
    prev, counting = 100, 0
    for i in range(1, 6):
        a = _int(form.get(f"bc_serial_{i}"))
        counting += int(a is not None and a == prev - 7)
        prev = a if a is not None else prev - 7
    s["counting"] = counting
    n1, n2 = _norm(form.get("bc_name_1", "")), _norm(form.get("bc_name_2", ""))
    s["naming"] = int("clock" in n1 or "watch" in n1) + int("pencil" in n2 or "pen" in n2.split() or "crayon" in n2)
    s["copy"] = int(_norm(form.get("bc_copy", "")) == _norm(COPY_SENTENCE))
    taps = [t for t in str(form.get("bc_taps", "")).split(",") if t][:3]
    s["steps"] = sum(1 for i, t in enumerate(taps) if t == TAP_ORDER[i])
    s["reading"] = int(form.get("bc_read") == READ_CORRECT)
    s["writing"] = int(len([w for w in _norm(form.get("bc_write", "")).split() if len(w) > 1]) >= 3)
    s["shape"] = int(form.get("bc_shape") == SHAPE_CORRECT)
    return float(sum(s.values())), s, {}
