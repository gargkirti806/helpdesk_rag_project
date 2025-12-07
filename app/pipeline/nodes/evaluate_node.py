from app.llm.llm_factory import evaluation_llm
from langchain.prompts import ChatPromptTemplate
from app.pipeline.nodes.retrieve_node import retrieve_docs
from app.pipeline.nodes.generate_node import generate_answer

prompt = ChatPromptTemplate.from_template("""
Evaluate if the ANSWER fully and correctly matches CONTEXT.
No hallucinations allowed. Return structured fields: confidence (0-1), sufficient (bool), reason (short).
User: {question}
Answer: {answer}
""")

def evaluate_answer(state):
    chain = prompt | evaluation_llm
    result = chain.invoke({"question": state.user_query, "answer": state.kb_answer})
    state.eval_confidence = result.confidence
    state.eval_sufficient = result.sufficient
    state.eval_reason = result.reason

    # Reflection loop: if insufficient or low confidence, try one re-retrieval with higher k and regenerate
    try:
        if not state.eval_sufficient or (state.eval_confidence is not None and state.eval_confidence < 0.8):
            # increase retrieval breadth
            state = retrieve_docs(state, override_k=20)
            state = generate_answer(state)
            # re-evaluate once
            result2 = chain.invoke({"question": state.user_query, "answer": state.kb_answer})
            state.eval_confidence = result2.confidence
            state.eval_sufficient = result2.sufficient
            state.eval_reason = result2.reason
    except Exception as e:
        # Log and continue with original evaluation
        print('Reflection loop failed:', e)

    return state