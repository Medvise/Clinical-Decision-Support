"""LangGraph wiring for the CDSS query pipeline."""

from langgraph.graph import END, StateGraph

from pipeline.graph_nodes import (
    after_patient_context,
    build_patient_context,
    build_prompt,
    call_model,
    extract_retrieval_tags_node,
    retrieve_general_chunks,
    retrieve_patient_chunks,
    skip_guideline_retrieval,
)
from pipeline.graph_state import CDSSState
from pipeline.routing import (
    build_clarification_result,
    next_after_route,
    route_query,
)


def build_cdss_graph():
    graph = StateGraph(CDSSState)
    graph.add_node("route_query", route_query)
    graph.add_node("build_patient_context", build_patient_context)
    graph.add_node("extract_retrieval_tags", extract_retrieval_tags_node)
    graph.add_node("retrieve_patient_chunks", retrieve_patient_chunks)
    graph.add_node("skip_guideline_retrieval", skip_guideline_retrieval)
    graph.add_node("retrieve_general_chunks", retrieve_general_chunks)
    graph.add_node("build_clarification_result", build_clarification_result)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("call_model", call_model)

    graph.set_entry_point("route_query")
    graph.add_conditional_edges("route_query", next_after_route)
    graph.add_conditional_edges(
        "build_patient_context",
        after_patient_context,
        {
            "extract_retrieval_tags": "extract_retrieval_tags",
            "skip_guideline_retrieval": "skip_guideline_retrieval",
        },
    )
    graph.add_edge("extract_retrieval_tags", "retrieve_patient_chunks")
    graph.add_edge("retrieve_patient_chunks", "build_prompt")
    graph.add_edge("skip_guideline_retrieval", "build_prompt")
    graph.add_edge("retrieve_general_chunks", "build_prompt")
    graph.add_edge("build_clarification_result", END)
    graph.add_edge("build_prompt", "call_model")
    graph.add_edge("call_model", END)
    return graph.compile()


CDSS_GRAPH = build_cdss_graph()
