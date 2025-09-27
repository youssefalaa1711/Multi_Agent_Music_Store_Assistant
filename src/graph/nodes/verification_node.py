"""
verify_info node:
- Parses user input and state for phone, email, or numeric customer id
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

# phones like +55 (12) 3923-5555, 555-1234, etc. (lenient but not too short)
_PHONE_RE = re.compile(r"\+?\d[\d\-\(\)\s]{6,}")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# a small integer, captured as cid
_ID_RE = re.compile(r"(?:^|[^0-9])(?P<cid>\d{1,5})(?:[^0-9]|$)")

def _normalize_phone(s: str) -> str:
    return re.sub(r"\D", "", s or "")

def _classify_intent(text: str) -> Optional[str]:
    t = text.lower()
    music_keywords = [
        "song", "songs", "track", "tracks", "album", "albums",
        "artist", "artists", "genre", "genres", "recommend", "music"
    ]
    invoice_keywords = [
        "invoice", "invoices", "purchase", "purchases", "order",
        "receipt", "support rep", "employee", "last purchase", "most recent", "billing"
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

def _verify_or_lookup_customer_id(text: str, phone_from_state: Optional[str] = None) -> tuple[Optional[int], Optional[str]]:
    """
    Try, in order:
    1) phone_from_state (if present)
    2) phone found in text
    3) email in text
    4) explicit small integer in text (and verify existence)
    Returns (customer_id, phone_used_if_any)
    """
    engine = get_engine_for_chinook_db()
    with engine.connect() as conn:
        # 1) state phone
        if phone_from_state:
            cid = _find_customer_id_by_phone(conn, phone_from_state)
            if cid:
                return cid, phone_from_state

        # 2) phone in text
        m = _PHONE_RE.search(text or "")
        if m:
            phone = m.group(0)
            cid = _find_customer_id_by_phone(conn, phone)
            if cid:
                return cid, phone

        # 3) email in text
        m = _EMAIL_RE.search(text or "")
        if m:
            cid = _find_customer_id_by_email(conn, m.group(0))
            if cid:
                return cid, None

        # 4) explicit id in text
        m = _ID_RE.search(text or "")
        if m:
            try:
                cid_candidate = int(m.group("cid"))
                row = conn.execute(
                    text("SELECT 1 FROM Customer WHERE CustomerId=:cid LIMIT 1"),
                    {"cid": cid_candidate}
                ).fetchone()
                if row:
                    return cid_candidate, None
            except Exception:
                pass

    return None, None

# -----------------------------
# Node
# -----------------------------

def verify_info(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
      state["input"]: user utterance

    Output (added/updated):
      - profile (with possibly resolved customer_id and phone)
      - intent (optional)
      - need_human (bool; only True for invoice-ish queries when id still unknown)
      - output (optional message)
    """
    text = state.get("input", "") or ""
    phone_in_state = state.get("phone")

    # Classify for routing/need_human purposes
    intent = _classify_intent(text)

    # Try to resolve customer id (prefer saved phone if we have it)
    resolved_cid, phone_used = _verify_or_lookup_customer_id(text, phone_from_state=phone_in_state)

    # Build dynamic profile from the global (legacy) profile
    dynamic_profile = dict(user_profile)  # shallow copy

    if phone_used and not dynamic_profile.get("phone"):
        dynamic_profile["phone"] = phone_used
    if resolved_cid:
        dynamic_profile["customer_id"] = str(resolved_cid)

    # Update legacy global user_profile for backward-compat paths
    if phone_used:
        user_profile["phone"] = phone_used
    if resolved_cid:
        user_profile["customer_id"] = str(resolved_cid)

    # Decide if human input is required:
    # - Only if it's an invoice/purchase/billing query AND we still don't know the customer.
    missing_customer = not bool(dynamic_profile.get("customer_id"))
    need_human = bool(intent == "invoice" and missing_customer)

    message = None
    if need_human:
        message = "I couldn’t verify your account. Please provide your phone number, email, or customer ID."

    return {
        **state,
        "profile": dynamic_profile,
        "intent": intent,
        "need_human": need_human,
        "output": message or state.get("output"),
    }
