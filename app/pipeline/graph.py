from langgraph.graph.state import StateGraph, START,END
from app.original.langraph_pipeline_typed_original import PipelineState
from app.pipeline.nodes.intent_node import classify_intent
from app.pipeline.nodes.retrieve_node import retrieve_docs
from app.pipeline.nodes.generate_node import generate_answer
from app.pipeline.nodes.evaluate_node import evaluate_answer
from app.pipeline.nodes.postprocess_node import postprocess
from app.memory.redis_checkpoint import checkpointer
# from langgraph.checkpoint.memory import MemorySaver

# # Initialize checkpointer
# checkpointer = MemorySaver()

def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("intent", classify_intent)
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("generate", generate_answer)
    graph.add_node("evaluate", evaluate_answer)
    graph.add_node("final", postprocess)

    graph.set_entry_point("intent")

    graph.add_edge("intent", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_edge("evaluate", "final")
    graph.add_edge("final", END)


    compiled = graph.compile(checkpointer=checkpointer)

    # Latest LangGraph: no arguments needed
    workflow = compiled.with_types()

    return workflow
workflow = build_graph()
