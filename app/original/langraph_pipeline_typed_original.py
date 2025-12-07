# -------------------------
# langraph_pipeline_typed_original.py (user's original)
# -------------------------
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
import os
# persist_dir='data/vector_db'
class Intentclassify(BaseModel):
    Intent: Literal["HR_Policy", "IT_guidelines"] = Field(
        ..., description="Classify the user query as HR_Policy or IT_guidelines."
    )

class EvaluationResult(BaseModel):
    confidence: float = Field(..., description="0.0 to 1.0")
    sufficient: bool = Field(..., description="Whether KB answer fully answers user")
    reason: str = Field(..., description="Short explanation")
class AnswerGeneration(BaseModel):
    answer: str = Field(..., description="Generated answer based on context and user query")
    
class PipelineState(BaseModel):
    user_query: str
    intent: Optional[str] = None
    compressed_docs: Optional[List] = None
    kb_answer: Optional[str] = None
    eval_confidence: Optional[float] = None
    eval_sufficient: Optional[bool] = None
    eval_reason: Optional[str] = None
    final_response: Optional[dict] = None

llm = ChatOllama(model="qwen:latest", temperature=0, # Context window
    num_predict=512,  # Max tokens to generate
    # Performance settings
    num_thread=4,  # Use multiple CPU threads
    repeat_penalty=1.1,)
intent_llm = llm.with_structured_output(Intentclassify)
evaluation_llm = llm.with_structured_output(EvaluationResult)
AnswerGeneration_llm = llm.with_structured_output(AnswerGeneration)

persist_dir = "/home/kirti/helpdesk_rag_project/data/vector_db"
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-4B", model_kwargs={
            'device': 'cuda',  # or 'cpu' if no GPU
            'trust_remote_code': True
        },
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 16  # Process multiple texts at once
        })

vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

def classify_intent_node(state: PipelineState) -> PipelineState:
    print("\n[DEBUG] ENTER classify_intent_node")
    print("[DEBUG] incoming state =", state.model_dump())

    response = intent_llm.invoke(state.user_query)
    state.intent = response.Intent

    print("[DEBUG] EXIT classify_intent_node")
    print("[DEBUG] returning state =", state.model_dump())
    return state


def rag_retrieval_node(state: PipelineState) -> PipelineState:
    print("\n[DEBUG] ENTER rag_retrieval_node")
    print("[DEBUG] state.intent =", state.intent)

    compressed_docs = compression_retriever.invoke(state.user_query)
    state.compressed_docs = compressed_docs

    print("[DEBUG] EXIT rag_retrieval_node")
    print("[DEBUG] doc_count =", len(compressed_docs))
    return state

def generate_answer_node(state: PipelineState) -> PipelineState:
    print("\n[DEBUG] ENTER generate_answer_node")
    print("[DEBUG] doc_count =", len(state.compressed_docs))

    response = AnswerGeneration_llm.invoke(prompt)
    state.kb_answer = response.answer

    print("[DEBUG] EXIT generate_answer_node")
    print("[DEBUG] kb_answer =", state.kb_answer)
    return state


def evaluate_node(state: PipelineState) -> PipelineState:
    eval_prompt = ChatPromptTemplate.from_template("""
You are an evaluation agent.
Check whether the KB answer fully answers the user's question.

Return ONLY the structured fields required by the schema.

User Question:
{question}

KB Answer:
{kb_answer}
""")
    chain = eval_prompt | evaluation_llm
    result = chain.invoke({"question": state.user_query, "kb_answer": state.kb_answer})
    state.eval_confidence = result.confidence
    state.eval_sufficient = result.sufficient
    state.eval_reason = result.reason
    return state

def post_evaluation_node(state: PipelineState) -> PipelineState:
    response = {}

    if state.eval_sufficient:
        response["answer"] = state.kb_answer
    else:
        response["answer"] = "KB answer insufficient. Escalating to human/HR."

    create_ticket = False
    ticket_summary = None

    if state.intent == "IT_guidelines":
        create_ticket = True
        ticket_summary = f"IT Ticket for user query: {state.user_query}"
    elif state.intent == "HR_Policy":
        if not state.eval_sufficient:
            create_ticket = True
            ticket_summary = f"HR Ticket for user query: {state.user_query}"

    if create_ticket:
        ticket_id = "TICKET-0000"  # placeholder
        response["ticket_id"] = ticket_id
        response["ticket_summary"] = ticket_summary

    if not state.eval_sufficient:
        response["escalation"] = "Human/HR team assigned"
        response["reason"] = state.eval_reason

    state.final_response = response
    return state

def run_helpdesk_pipeline(user_query: str) -> dict:
    state = PipelineState(user_query=user_query)
    state = classify_intent_node(state)
    state = rag_retrieval_node(state)
    state = generate_answer_node(state)
    state = evaluate_node(state)
    state = post_evaluation_node(state)
    return state.final_response

if __name__ == "__main__":
    query = "How can I reset my Outlook password?"
    output = run_helpdesk_pipeline(query)
    print(output)