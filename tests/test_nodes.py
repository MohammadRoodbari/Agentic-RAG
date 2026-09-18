from app.agent.nodes import format_response, validate_input


def _make_state(**overrides):
    base = {
        "question": "What is GDPR?",
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "error": None,
    }
    base.update(overrides)
    return base


class TestValidateInput:
    def test_valid_question(self):
        result = validate_input(_make_state())
        assert "error" not in result or result.get("error") is None

    def test_empty_question(self):
        result = validate_input(_make_state(question=""))
        assert result["error"] == "Question must not be empty"

    def test_whitespace_question(self):
        result = validate_input(_make_state(question="   "))
        assert result["error"] == "Question must not be empty"

    def test_question_too_long(self):
        result = validate_input(_make_state(question="x" * 2001))
        assert "2000" in result["error"]

    def test_question_at_limit(self):
        result = validate_input(_make_state(question="x" * 2000))
        assert "error" not in result or result.get("error") is None


class TestFormatResponse:
    def test_extracts_cited_sources(self):
        state = _make_state(
            answer="According to [1], data must be protected. See also [2].",
            retrieved_chunks=[
                {"text": "Data protection is required under Article 5.",
                 "source": "gdpr.pdf", "page": 1, "chunk_index": 0, "score": 0.9},
                {"text": "Controllers must implement appropriate measures.",
                 "source": "gdpr.pdf", "page": 3, "chunk_index": 2, "score": 0.8},
                {"text": "Unused chunk that was not cited.",
                 "source": "gdpr.pdf", "page": 5, "chunk_index": 4, "score": 0.7},
            ],
        )
        result = format_response(state)
        assert len(result["citations"]) == 2
        assert result["citations"][0]["source"] == "gdpr.pdf"
        assert result["citations"][0]["page"] == 1
        assert result["citations"][1]["page"] == 3

    def test_no_citations_when_none_referenced(self):
        state = _make_state(
            answer="I don't have enough information to answer this question.",
            retrieved_chunks=[
                {"text": "Some chunk", "source": "doc.pdf",
                 "page": 1, "chunk_index": 0, "score": 0.5},
            ],
        )
        result = format_response(state)
        assert result["citations"] == []

    def test_skips_on_error(self):
        result = format_response(_make_state(error="Something failed"))
        assert result == {}

    def test_strips_whitespace_from_answer(self):
        state = _make_state(
            answer="  Some answer with whitespace.  ",
            retrieved_chunks=[],
        )
        result = format_response(state)
        assert result["answer"] == "Some answer with whitespace."

    def test_handles_duplicate_citation_references(self):
        state = _make_state(
            answer="As stated in [1] and reiterated in [1], this is important.",
            retrieved_chunks=[
                {"text": "Important regulation text.",
                 "source": "soc2.pdf", "page": 2, "chunk_index": 0, "score": 0.9},
            ],
        )
        result = format_response(state)
        assert len(result["citations"]) == 1

    def test_citation_text_truncated(self):
        long_text = "x" * 500
        state = _make_state(
            answer="According to [1], this matters.",
            retrieved_chunks=[
                {"text": long_text, "source": "doc.pdf",
                 "page": 1, "chunk_index": 0, "score": 0.9},
            ],
        )
        result = format_response(state)
        assert len(result["citations"][0]["text"]) == 200
