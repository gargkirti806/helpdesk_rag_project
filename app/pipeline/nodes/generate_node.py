from app.llm.llm_factory import get_answer_generation_llm
from app.llm.prompts import STRICT_RAG_PROMPT

MAX_DOCS = 3  # max docs to include in context to avoid LLM freezing

def generate_answer(state):
    try:
        print("\n[DEBUG] ENTER generate_answer_node")
        # Safely limit the number of docs
        docs_to_use = (state.compressed_docs or [])[:MAX_DOCS]
        context = "\n\n".join([doc.page_content for doc in docs_to_use])
        print(f"[DEBUG] Using {len(docs_to_use)} docs for context. Total characters: {len(context)}")

        # Build prompt safely
        prompt = STRICT_RAG_PROMPT.format(context=context, question=state.user_query)
        print("[DEBUG] Prompt constructed. Sending to LLM...")

        llm = get_answer_generation_llm()
        response = llm.invoke(prompt)

        # Ensure structured response
        if hasattr(response, "answer"):
            state.kb_answer = response.answer
            print("[DEBUG] LLM returned answer:", state.kb_answer[:200], "...")  # print first 200 chars
        else:
            print("[WARNING] LLM response missing 'answer' field. Returning empty string.")
            state.kb_answer = ""
        if isinstance(response.answer, bytes):
            state.kb_answer = response.answer.decode("utf-8", errors="ignore")
        else:
            state.kb_answer = str(response.answer)
            print("[DEBUG] EXIT generate_answer_node")
        return state

    except Exception as e:
        print("[ERROR] generate_answer_node failed:", e)
        state.kb_answer = ""
        return state

