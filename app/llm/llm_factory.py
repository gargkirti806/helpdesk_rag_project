from langchain_ollama import ChatOllama
from app.original.langraph_pipeline_typed_original import Intentclassify
from app.original.langraph_pipeline_typed_original import EvaluationResult,AnswerGeneration
from langchain.output_parsers import PydanticOutputParser

# Base LLM
llm = ChatOllama(model="qwen:latest", temperature=0)

# Intent classification LLM
intent_llm = llm.with_structured_output(Intentclassify)

# Evaluation LLM
evaluation_llm = llm.with_structured_output(EvaluationResult)
AnswerGeneration_llm = llm.with_structured_output(AnswerGeneration)
# Optional: helper functions to get LLMs
def get_intent_llm():
    return intent_llm

def get_evaluation_llm():
    return evaluation_llm

def get_llm():
    """Return base LLM."""
    return llm

def get_answer_generation_llm():
    return AnswerGeneration_llm    
