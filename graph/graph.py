from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEB_SEARCH
from graph.nodes import generate, retrieve, web_search, grade_documents
from graph.state import GraphState
from graph.chains.router import question_router, RouteQuery

load_dotenv()


def decide_to_generate(state):
    print("--- ASSESS GRADED DOCUMENTS ---")
    if state["web_search"]:
        print("--- DECISION: WEB SEARCH(0 RELEVANT DOCUMENTS) ---")
        return WEB_SEARCH
    else:
        print("--- DECISION: GENERATE ---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("--- CHECK HALLUCINATION ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("--- DECISION: GENERATION IS GROUNDED IN DOCUMENTS ---")
        print("--- GRADE GENERATION vs QUESTION ---")
        score = answer_grader.invoke({"question": question, "answer": generation})
        if answer_grade := score.binary_score:
            print("--- DECISION: GENERATION ADDRESSES QUESTION ---")
            return "useful"
        else:
            print("--- DECISION: GENERATION DOES NOT ADDRESS QUESTION ---")
            return "note useful"
    else:
        print("--- DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS ---")
        return "not supported"


def route_question(state: GraphState) -> str:
    print("--- ROUTE QUESTION ---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.data_source == "vector_store":
        print("--- DECISION: ROUTE QUESTION TO VECTOR STORE ---")
        return RETRIEVE
    elif source.data_source == "web_search":
        print("--- DECISION: ROUTE QUESTION TO WEB SEARCH ---")
        return WEB_SEARCH
    else:
        print("--- DECISION: NO DATA SOURCE FOUND ---")
        return END


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

workflow.set_conditional_entry_point(
    route_question,
    path_map={
        WEB_SEARCH: WEB_SEARCH,
        RETRIEVE: RETRIEVE,
        END: END,
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={
        "not_supported": GENERATE,
        "useful": END,
        "note_useful": WEB_SEARCH,
    },
)
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")
