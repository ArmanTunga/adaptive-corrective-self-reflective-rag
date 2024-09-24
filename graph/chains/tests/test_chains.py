from dotenv import load_dotenv

from pprint import pprint

from graph.chains.hallucination_grader import GradeHallucinations, hallucination_grader
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.router import question_router, RouteQuery
from ingestion import get_retriever
from graph.chains.generation import generation_chain

load_dotenv()


def test_retrieval_grader_answer_yes() -> None:
    retriever = get_retriever()
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    assert res.binary_score.lower() == "yes"


def test_retrieval_grader_answer_no() -> None:
    retriever = get_retriever()
    question = "agnet memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "How to make pizza", "document": doc_txt}
    )
    assert res.binary_score.lower() == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    retriever = get_retriever()
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    retriever = get_retriever()
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})

    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    retriever = get_retriever()
    docs = retriever.invoke(question)
    # generation = generation_chain.invoke({"context": docs, "question": question})

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "To make a pizza, you need to mix the dough and bake it in the oven.",
        }
    )
    assert not res.binary_score


def test_router_to_vector_store() -> None:
    question = "agent memory"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.data_source == "vector_store"


def test_router_to_websearch() -> None:
    question = "Object Detection"
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.data_source == "web_search"
