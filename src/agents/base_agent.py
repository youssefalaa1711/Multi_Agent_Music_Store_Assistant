"""
Base agent utilities: wraps tools into LangChain StructuredTools
and provides shared logic for all agents.
"""

from langchain.tools import StructuredTool
from src.tools.music_tools import (
    get_albums_by_artist,
    get_tracks_by_artist,
    get_songs_by_genre,
    check_for_songs,
)
from src.tools.invoice_tools import (
    get_invoices_by_customer_sorted_by_date,
    get_invoices_sorted_by_unit_price,
    get_employee_by_invoice_and_customer,
)

# -----------------------------
# 🎵 Music Tools
# -----------------------------
music_tools = [
    StructuredTool.from_function(
        func=get_albums_by_artist,
        name="GetAlbumsByArtist",
        description="Fetch albums for a given artist. Args: artist (string). Output: JSON list of album titles."
    ),
    StructuredTool.from_function(
        func=get_tracks_by_artist,
        name="GetTracksByArtist",
        description="Fetch tracks for a given artist. Args: artist (string). Output: JSON list of track names."
    ),
    StructuredTool.from_function(
        func=get_songs_by_genre,
        name="GetSongsByGenre",
        description="Fetch songs for a given genre. Args: genre (string). Output: JSON list of song titles."
    ),
    StructuredTool.from_function(
        func=check_for_songs,
        name="CheckForSongs",
        description="Check if a song exists by title. Args: song_title (string). Output: { 'exists': true/false }."
    ),
]

# -----------------------------
# 🧾 Invoice Tools
# -----------------------------
invoice_tools = [
    StructuredTool.from_function(
        func=get_invoices_by_customer_sorted_by_date,
        name="GetInvoicesByCustomerSortedByDate",
        description="Fetch invoices for a customer sorted by date. Args: customer_id (int). Output: JSON list of invoices."
    ),
    StructuredTool.from_function(
        func=get_invoices_sorted_by_unit_price,
        name="GetInvoicesSortedByUnitPrice",
        description="Fetch invoices for a customer sorted by unit price. Args: customer_id (int). Output: JSON list with InvoiceId & UnitPrice."
    ),
    StructuredTool.from_function(
        func=get_employee_by_invoice_and_customer,
        name="GetEmployeeByInvoiceAndCustomer",
        description="Fetch the employee who handled a specific invoice. Args: invoice_id (int), customer_id (int). Output: JSON object with employee details."
    ),
]
