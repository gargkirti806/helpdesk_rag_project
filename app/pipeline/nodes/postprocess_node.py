# app/pipeline/nodes/postprocess_node.py

from app.original.langraph_pipeline_typed_original import PipelineState
from app.utils.ticket import create_ticket_api

def postprocess(state: PipelineState) -> PipelineState:
    print("\n[DEBUG] ENTER postprocess_node")
    print("[DEBUG] Incoming state =", state.model_dump())

    response = {}

    # --- KB Answer Handling ---
    if state.eval_sufficient:
        response["answer"] = state.kb_answer
    else:
        response["answer"] = "KB answer insufficient. Escalating to human/HR."
    print(f"[DEBUG] KB sufficient? {state.eval_sufficient}")

    # --- Ticket Logic ---
    create_ticket = False
    ticket_summary = None

    if state.intent == "IT_guidelines":
        create_ticket = True
        ticket_summary = f"IT Ticket for user query: {state.user_query}"
        print("[DEBUG] Creating IT ticket")
    elif state.intent == "HR_Policy":
        if not state.eval_sufficient:
            create_ticket = True
            ticket_summary = f"HR Ticket for user query: {state.user_query}"
            print("[DEBUG] Creating HR ticket (insufficient KB answer)")
        else:
            print("[DEBUG] HR policy but KB answer sufficient → no ticket")
    else:
        print(f"[DEBUG] Unknown intent '{state.intent}', no ticket generated")

    # --- Create Ticket If Needed ---
    if create_ticket:
        try:
            ticket_id = create_ticket_api(ticket_summary)
            response["ticket_id"] = ticket_id
            response["ticket_summary"] = ticket_summary
            print(f"[DEBUG] Ticket created with ID {ticket_id}")
        except Exception as e:
            print("[ERROR] Ticket creation failed:", str(e))
            response["ticket_error"] = str(e)

    # --- Escalation ---
    if not state.eval_sufficient:
        response["escalation"] = "Human/HR team assigned"
        response["reason"] = state.eval_reason
        print("[DEBUG] Escalated to human/HR team")

    # --- Finalize ---
    state.final_response = response
    print("[DEBUG] Final response =", response)
    print("[DEBUG] EXIT postprocess_node\n")

    return state
