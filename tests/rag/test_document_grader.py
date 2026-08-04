from unittest.mock import MagicMock, patch

from src.rag.agents.document_grader import PREVIEW_CHARS, format_passages, keep_relevant

PASSAGES = [
    {"title": "BIAT results", "content": "BIAT reported a profit."},
    {"title": "Football", "content": "The derby ended in a draw."},
    {"title": "Market close", "content": "The index gained 1%."},
]


def grader(indices):
    chain = MagicMock()
    chain.invoke.return_value = MagicMock(relevant_indices=indices)
    return chain


@patch("src.rag.agents.document_grader.get_document_grader")
def test_keeps_only_the_graded_passages_in_order(mock_grader):
    mock_grader.return_value = grader([2, 0])

    kept = keep_relevant("How did BIAT do?", PASSAGES)

    assert [p["title"] for p in kept] == ["BIAT results", "Market close"]


@patch("src.rag.agents.document_grader.get_document_grader")
def test_keeps_nothing_when_nothing_is_relevant(mock_grader):
    mock_grader.return_value = grader([])

    assert keep_relevant("How did BIAT do?", PASSAGES) == []


@patch("src.rag.agents.document_grader.get_document_grader")
def test_ignores_indices_that_do_not_exist(mock_grader):
    mock_grader.return_value = grader([0, 99])

    assert len(keep_relevant("How did BIAT do?", PASSAGES)) == 1


@patch("src.rag.agents.document_grader.get_document_grader")
def test_fails_open_when_the_grader_is_unreachable(mock_grader):
    mock_grader.side_effect = RuntimeError("gemini down")

    # A grading outage should cost precision, never the answer.
    assert keep_relevant("How did BIAT do?", PASSAGES) == PASSAGES


@patch("src.rag.agents.document_grader.get_document_grader")
def test_fails_open_when_the_grader_answers_with_nonsense(mock_grader):
    mock_grader.return_value = grader("not a list of numbers")

    assert keep_relevant("How did BIAT do?", PASSAGES) == PASSAGES


@patch("src.rag.agents.document_grader.get_document_grader")
def test_an_empty_batch_is_not_sent_to_the_grader(mock_grader):
    assert keep_relevant("How did BIAT do?", []) == []
    mock_grader.assert_not_called()


def test_passages_are_numbered_for_the_grader():
    formatted = format_passages(PASSAGES)

    assert formatted.startswith("[0] BIAT results:")
    assert "[1] Football:" in formatted
    assert "[2] Market close:" in formatted


def test_only_the_opening_of_a_passage_is_sent():
    formatted = format_passages([{"title": "Long", "content": "x" * 5000}])

    assert formatted.count("x") == PREVIEW_CHARS
