from sqlalchemy import text
from src.database.chinook_loader import get_chinook_db

db = get_chinook_db()


def get_invoices_by_customer_sorted_by_date(customer_id: int) -> list[dict]:
    """Retrieve all invoices for a customer, sorted by date (latest first)."""
    query = text("""
        SELECT InvoiceId, InvoiceDate, Total
        FROM Invoice
        WHERE CustomerId = :customer_id
        ORDER BY InvoiceDate DESC
    """)
    with db._engine.connect() as conn:
        result = conn.execute(query, {"customer_id": customer_id}).fetchall()
    invoices = [{"InvoiceId": row[0], "InvoiceDate": row[1], "Total": row[2]} for row in result]
    return str(invoices)


def get_invoices_sorted_by_unit_price(customer_id: int) -> list[dict]:
    """Retrieve invoices for a customer, sorted by unit price (highest first)."""
    query = text("""
        SELECT Invoice.InvoiceId, Track.UnitPrice
        FROM Invoice
        JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId
        JOIN Track ON InvoiceLine.TrackId = Track.TrackId
        WHERE Invoice.CustomerId = :customer_id
        ORDER BY Track.UnitPrice DESC
    """)
    with db._engine.connect() as conn:
        result = conn.execute(query, {"customer_id": customer_id}).fetchall()
    invoices = [{"InvoiceId": row[0], "UnitPrice": row[1]} for row in result]
    return str(invoices)


def get_employee_by_invoice_and_customer(invoice_id: int, customer_id: int) -> dict | None:
    """Retrieve the employee who handled a given invoice for a customer."""
    query = text("""
        SELECT Employee.FirstName, Employee.LastName, Employee.Title
        FROM Employee
        JOIN Customer ON Employee.EmployeeId = Customer.SupportRepId
        JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId
        WHERE Invoice.InvoiceId = :invoice_id AND Customer.CustomerId = :customer_id
    """)
    with db._engine.connect() as conn:
        result = conn.execute(query, {"invoice_id": invoice_id, "customer_id": customer_id}).fetchall()
    if result:
        row = result[0]
        return {"FirstName": row[0], "LastName": row[1], "Title": row[2]}
    return None
