# -------------------------
# intent_node.py
# -------------------------

from app.llm.llm_factory import get_intent_llm
from app.llm.prompts import STRICT_INTENT_PROMPT  # define a prompt template for intent classification
import json
def classify_intent(state):
    print("\n[DEBUG] ENTER classify_intent_node")
    print("[DEBUG] incoming state =", state.model_dump())

    llm = get_intent_llm()
    prompt = STRICT_INTENT_PROMPT.format(question=state.user_query)
    
    response = llm.invoke(prompt)  # returns Intentclassify

    # Directly access the Intent field
    state.intent = response.Intent

    print("[DEBUG] EXIT classify_intent_node")
    print("[DEBUG] returning state =", state.model_dump())
    return state
