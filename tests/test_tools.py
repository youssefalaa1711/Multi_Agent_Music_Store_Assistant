
from src.tools.music_tools import get_albums_by_artist, get_tracks_by_artist, get_songs_by_genre, check_for_songs
from src.tools.invoice_tools import get_invoices_by_customer_sorted_by_date, get_invoices_sorted_by_unit_price, get_employee_by_invoice_and_customer


def run_music_tests():
    print("Albums by U2:", get_albums_by_artist("U2"))
    print("Tracks by U2:", get_tracks_by_artist("U2"))
    print("Songs in Rock genre:", get_songs_by_genre("Rock")[:5])
    print("Does song 'Let There Be Rock' exist?", check_for_songs("Let There Be Rock"))


def run_invoice_tests():
    print("Invoices for customer 1:", get_invoices_by_customer_sorted_by_date(1)[:2])
    print("Invoices by unit price for customer 1:", get_invoices_sorted_by_unit_price(1)[:3])
    print("Employee for invoice 1 (customer 1):", get_employee_by_invoice_and_customer(1, 1))


if __name__ == "__main__":
    run_music_tests()
    run_invoice_tests()
