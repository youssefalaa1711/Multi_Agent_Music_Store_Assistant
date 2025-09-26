"""
verify_info node:
- Parses user input for phone, email, or numeric customer id
- Resolves to CustomerId from Chinook DB (normalizes phone formats)
- Updates the (dynamic) profile and decides whether human input is needed
- Adds a light "intent" classification to help later routing (optional)
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import re
from sqlalchemy import text
from src.memory.user_profile import user_profile  # current in-memory profile dict
from src.database.chinook_loader import get_engine_for_chinook_db


# -----------------------------
# Helpers
# -----------------------------

_phone_re = re.compile(
    r"(?:\+?\d{1,3}\s*)?(?:\(?\d{1,4}\)?[\s-]*)?\d{3,4}[\s-]?\d{4}"
)
_email_re = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_id_re = re.compile(r"(?:^|[^0-9])(?P<cid>\d{1,5})(?:[^0-9]|$)")  # loose capture of a small integer


def _normalize_phone(s: str) -> str:
    """Keep only digits for comparison."""
    return re.sub(r"\D", "", s or "")


def _classify_intent(text: str) -> Optional[str]:
    t = text.lower()
    music_keywords = [
        "song", "songs", "track", "tracks", "album", "albums",
        "artist", "artists", "genre", "genres", "recommend", "music"
    ]
    invoice_keywords = [
        "invoice", "invoices", "purchase", "purchases", "order",
        "receipt", "support rep", "employee", "last purchase", "most recent"
    ]
    if any(k in t for k in invoice_keywords) and not any(k in t for k in music_keywords):
        return "invoice"
    if any(k in t for k in music_keywords) and not any(k in t for k in invoice_keywords):
        return "music"
    return None


def _find_customer_id_by_phone(conn, raw_phone: str) -> Optional[int]:
    digits = _normalize_phone(raw_phone)
    if not digits:
        return None
    # Compare against a digits-only version of Customer.Phone
    row = conn.execute(
        text("""
            SELECT CustomerId
            FROM Customer
            WHERE REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(Phone,' ',''),'(',''),')',''),'+',''),'-','')
                  LIKE :digits
            LIMIT 1
        """),
        {"digits": f"%{digits}%"}
    ).fetchone()
    return int(row[0]) if row else None


def _find_customer_id_by_email(conn, email: str) -> Optional[int]:
    row = conn.execute(
        text("""
            SELECT CustomerId
            FROM Customer
            WHERE lower(Email) = lower(:email)
            LIMIT 1
        """),
        {"email": email.strip()}
    ).fetchone()
    return int(row[0]) if row else None


def _verify_or_lookup_customer_id(text: str) -> Optional[int]:
    """Try phone → email → explicit number, return CustomerId or None."""
    engine = get_engine_for_chinook_db()
    with engine.connect() as conn:
        # 1) phone
        m = _phone_re.search(text)
        if m:
            cid = _find_customer_id_by_phone(conn, m.group(0))
            if cid:
                return cid

        # 2) email
        m = _email_re.search(text)
        if m:
            cid = _find_customer_id_by_email(conn, m.group(0))
            if cid:
                return cid

        # 3) explicit id (very lenient: takes any small int in text)
        m = _id_re.search(text)
        if m:
            try:
                cid = int(m.group("cid"))
                # verify it exists
                row = conn.execute(
                    text("SELECT 1 FROM Customer WHERE CustomerId=:cid LIMIT 1"),
                    {"cid": cid}
                ).fetchone()
                if row:
                    return cid
            except Exception:
                pass

    return None


# -----------------------------
# Node
# -----------------------------

def verify_info(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
      state["input"]: user utterance

    Output (added/updated):
      - profile (with possibly resolved customer_id)
      - intent (optional)
      - need_human (bool)
    """
    text = state.get("input", "") or ""

    # Try to resolve customer id from the utterance
    resolved_cid = _verify_or_lookup_customer_id(text)

    # Build a dynamic profile view (start from current user_profile)
    dynamic_profile = dict(user_profile)  # shallow copy
    if resolved_cid:
        dynamic_profile["customer_id"] = str(resolved_cid)

    # If still no id, we need human input
    need_human = not bool(dynamic_profile.get("customer_id"))

    # Optional: store intent hint (supervisor still decides final routing)
    intent = _classify_intent(text)

    # Also update the global user_profile so legacy code that imports it still sees the id
    # (You can remove this line once your supervisor fully uses the state.profile.)
    if resolved_cid:
        user_profile["customer_id"] = str(resolved_cid)

    if need_human:
        message = (
            "I couldn’t verify your account. "
            "Please provide your customer ID, email, or phone number."
        )
    else:
        message = None

    return {
        **state,
        "profile": dynamic_profile,
        "intent": intent,
        "need_human": need_human,
        "output": message or state.get("output"),
    }
