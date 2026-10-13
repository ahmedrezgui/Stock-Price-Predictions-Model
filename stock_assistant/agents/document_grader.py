"""
Relevance grading for retrieved passages.

A similarity search always returns its k nearest passages, however far away they
are: ask about a company nobody wrote about and you still get seven articles back.
Generating an answer from those and only then discovering it is poor costs a full
answer and a retry, so passages are graded first and the weak ones dropped.

One call grades the whole batch. Grading passages one at a time would multiply the
cost of every search by k for no better result.
"""

from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from ..providers import create_chat_model
from pydantic import BaseModel, Field

load_dotenv()

# How much of a passage the grader reads. Relevance is clear from the opening;
# sending the rest would cost tokens on every search to change almost nothing.
PREVIEW_CHARS = 400


class RelevantDocuments(BaseModel):
    """The passages worth keeping out of a retrieved batch."""

    relevant_indices: List[int] = Field(
        description=("The numbers of the passages that could help answer the question. Empty if none of them could.")
    )


system = """You are grading passages retrieved from a Tunisian financial news archive.

Decide which of the numbered passages could help answer the question. Keep a passage \
if it bears on the question at all, even partially: the aim is to drop what is \
plainly about something else, not to keep only the perfect match. Return the numbers \
of the passages worth keeping, and nothing else.

# Output Format
The list of passage numbers to keep. An empty list if none of them relate to the \
question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question}\n\nPassages:\n{passages}"),
    ]
)


def get_document_grader():
    """
    Build the grading chain.

    Built on demand rather than at import so that loading the tools does not
    require API credentials.
    """
    llm = create_chat_model(simple=True)
    return grade_prompt | llm.with_structured_output(RelevantDocuments)


def format_passages(documents: List[dict]) -> str:
    """Number the passages so the grader can refer to them by index."""
    return "\n\n".join(
        f"[{index}] {document.get('title') or 'Untitled'}: {(document.get('content') or '')[:PREVIEW_CHARS]}"
        for index, document in enumerate(documents)
    )


def keep_relevant(question: str, documents: List[dict]) -> List[dict]:
    """
    Drop the passages that do not bear on the question.

    Fails open: if the grader is unreachable or answers with nonsense, every
    passage is kept. A grading outage should cost precision, not the answer.

    Args:
        question: the search query the passages were retrieved for.
        documents: the retrieved passages.

    Returns:
        The passages worth keeping, in their original order.
    """
    if not documents:
        return documents

    try:
        graded = get_document_grader().invoke({"question": question, "passages": format_passages(documents)})
        indices = {int(i) for i in graded.relevant_indices}
    except Exception as e:
        print(f"Document grading failed, keeping every passage: {e}")
        return documents

    return [document for index, document in enumerate(documents) if index in indices]
