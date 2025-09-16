"""
Base agent utilities: wraps tools into LangChain Tools
and provides shared logic for all agents.
"""

from langchain.tools import Tool
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

#Music Tools
music_tools = [
    Tool(
        name="GetAlbumsByArtist",
        func=get_albums_by_artist,
        description="Retrieve albums by a given artist. Input: artist name (str)."
    ),
    Tool(
        name="GetTracksByArtist",
        func=get_tracks_by_artist,
        description="Retrieve tracks by a given artist. Input: artist name (str)."
    ),
    Tool(
        name="GetSongsByGenre",
        func=get_songs_by_genre,
        description="Retrieve songs by a given genre. Input: genre name (str)."
    ),
    Tool(
        name="CheckForSongs",
        func=check_for_songs,
        description="Check if a song exists by title. Input: song title (str)."
    ),
]

#Invoice Tools
invoice_tools = [
    Tool(
        name="GetInvoicesByCustomerSortedByDate",
        func=get_invoices_by_customer_sorted_by_date,
        description="Retrieve invoices for a customer sorted by date. Input: customer_id (int)."
    ),
    Tool(
        name="GetInvoicesSortedByUnitPrice",
        func=get_invoices_sorted_by_unit_price,
        description="Retrieve invoices for a customer sorted by unit price. Input: customer_id (int)."
    ),
    Tool(
        name="GetEmployeeByInvoiceAndCustomer",
        func=get_employee_by_invoice_and_customer,
        description="Retrieve the employee handling a given invoice for a customer. Input: invoice_id (int), customer_id (int)."
    ),
]
