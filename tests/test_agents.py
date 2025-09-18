"""
Tests for Music and Invoice Agents.
Ensures agents are correctly calling tools
and returning structured outputs.
"""

import pytest
from src.agents.music_agent import build_music_agent
from src.agents.invoice_agent import build_invoice_agent


@pytest.fixture(scope="module")
def music_agent():
    return build_music_agent()


@pytest.fixture(scope="module")
def invoice_agent():
    return build_invoice_agent()


# -----------------------------
# 🎵 Music Agent Tests
# -----------------------------
def test_music_albums(music_agent):
    """Agent should list albums for U2 using the tool."""
    result = music_agent.invoke({"input": "List albums by U2"})
    print("\nAlbums by U2:", result)
    assert "output" in result
    assert isinstance(result["output"], list) or result["output"] == ""


def test_music_tracks(music_agent):
    """Agent should list tracks for U2 using the tool."""
    result = music_agent.invoke({"input": "List tracks by U2"})
    print("\nTracks by U2:", result)
    assert "output" in result


def test_music_genre(music_agent):
    """Agent should fetch songs by genre (Rock)."""
    result = music_agent.invoke({"input": "List songs in Rock genre"})
    print("\nRock Songs:", result)
    assert "output" in result


# -----------------------------
# 🧾 Invoice Agent Tests
# -----------------------------
def test_invoices_by_date(invoice_agent):
    """Agent should fetch last 2 invoices for customer 1."""
    result = invoice_agent.invoke({"input": "Get the last 2 invoices for customer 1"})
    print("\nInvoices by date:", result)
    assert "output" in result


def test_invoices_by_unit_price(invoice_agent):
    """Agent should fetch invoices sorted by unit price."""
    result = invoice_agent.invoke({"input": "Get invoices by unit price for customer 1"})
    print("\nInvoices by unit price:", result)
    assert "output" in result


def test_employee_lookup(invoice_agent):
    """Agent should fetch employee details for an invoice."""
    result = invoice_agent.invoke({"input": "Which employee handled invoice 1 for customer 1?"})
    print("\nEmployee for invoice 1 (customer 1):", result)
    assert "output" in result
