from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
    state (dict): The current graph state

    Returns:
    state (dict): Filtered out irrelevant documents updated web_search state
    """

    print("--- CHECK DOCUMENT RELEVANCE TO QUESTION ---")

    question = state.get("question")
    documents = state.get("documents")

    filtered_docs = []
    web_search = False

    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc})
        grade = score.binary_score

        if grade.lower() == "yes":
            print("--- GRADE: DOCUMENT IS RELEVANT ---")
            filtered_docs.append(doc)

    if not filtered_docs:
        print("--- GRADE: NO DOCUMENTS ARE RELEVANT ---")
        web_search = True

    return {"documents": filtered_docs, "question": question, "web_search": web_search}
