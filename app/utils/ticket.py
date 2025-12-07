# app/pipeline/helpers/ticket_helper.py

import random

def create_ticket_api(summary: str) -> str:
    """
    Mock ticket creation API.
    Returns a fake ticket ID for demonstration.
    """
    ticket_id = f"TICKET-{random.randint(1000, 9999)}"
    print(f"[Ticket Created] {ticket_id}: {summary}")
    return ticket_id
