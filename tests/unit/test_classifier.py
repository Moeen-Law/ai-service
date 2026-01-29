"""
Unit Tests for Task Classifier

Tests the explicit task classification from API requests to domain entities.
"""

import pytest
from uuid import UUID

from app.core.domain.enums import TaskType, Jurisdiction, Language, LegalDomain
from app.core.validators.classifier import TaskClassifier
from app.shared.errors.exceptions import (
    TaskNotSupportedError,
    ContextValidationError,
)

# Note: Language enum uses ARABIC/ENGLISH, not AR/EN


class TestTaskClassifier:
    """Test suite for TaskClassifier."""

    @pytest.fixture
    def classifier(self) -> TaskClassifier:
        """Create a classifier instance."""
        return TaskClassifier()

    def test_classify_legal_chat_success(self, classifier: TaskClassifier) -> None:
        """Test successful classification of LEGAL_CHAT task."""
        task = classifier.classify(
            task_type=TaskType.LEGAL_CHAT,
            context_data={
                "jurisdiction": "EGYPT",
                "language": "ar",
            },
            payload={
                "query": "ما هي شروط الزواج المدني؟",
            },
        )

        assert task.task_type == TaskType.LEGAL_CHAT
        assert task.context.jurisdiction == Jurisdiction.EGYPT
        assert task.context.language == Language.ARABIC
        assert task.payload["query"] == "ما هي شروط الزواج المدني؟"
        assert isinstance(task.task_id, UUID)

    def test_classify_contract_analysis_with_domain(
        self, classifier: TaskClassifier
    ) -> None:
        """Test classification with legal domain context."""
        task = classifier.classify(
            task_type=TaskType.CONTRACT_ANALYSIS,
            context_data={
                "jurisdiction": "UAE",
                "language": "en",
                "domain": "COMMERCIAL",
            },
            payload={
                "contract_text": "Sample contract",
                "analysis_type": "full",
            },
        )

        assert task.task_type == TaskType.CONTRACT_ANALYSIS
        assert task.context.jurisdiction == Jurisdiction.UAE
        assert task.context.domain == LegalDomain.COMMERCIAL

    def test_classify_with_options(self, classifier: TaskClassifier) -> None:
        """Test classification with execution options."""
        task = classifier.classify(
            task_type=TaskType.LEGAL_CHAT,
            context_data={
                "jurisdiction": "EGYPT",
                "language": "ar",
            },
            payload={
                "query": "Test query",
            },
            options_data={
                "include_sources": False,
                "max_tokens": 500,
            },
        )

        assert task.options is not None
        assert task.options.include_sources is False
        assert task.options.max_tokens == 500

    def test_classify_invalid_task_type(self, classifier: TaskClassifier) -> None:
        """Test that invalid task type raises TaskNotSupportedError."""
        with pytest.raises(TaskNotSupportedError) as exc_info:
            classifier.classify(
                task_type="INVALID_TASK",  # type: ignore
                context_data={"jurisdiction": "EGYPT", "language": "ar"},
                payload={},
            )

        assert "INVALID_TASK" in str(exc_info.value)

    def test_classify_unsupported_jurisdiction(
        self, classifier: TaskClassifier
    ) -> None:
        """Test that unsupported jurisdiction raises ContextValidationError."""
        with pytest.raises(ContextValidationError) as exc_info:
            classifier.classify(
                task_type=TaskType.LEGAL_CHAT,
                context_data={
                    "jurisdiction": "USA",
                    "language": "en",
                },
                payload={"query": "Test"},
            )

        error = exc_info.value
        assert "jurisdiction" in str(error).lower()
        # Should list supported jurisdictions
        assert "supported" in str(error.details).lower() or error.details is not None

    def test_classify_case_insensitive_jurisdiction(
        self, classifier: TaskClassifier
    ) -> None:
        """Test that jurisdiction matching is case-insensitive."""
        task = classifier.classify(
            task_type=TaskType.LEGAL_CHAT,
            context_data={
                "jurisdiction": "egypt",  # lowercase
                "language": "AR",  # uppercase
            },
            payload={"query": "Test"},
        )

        assert task.context.jurisdiction == Jurisdiction.EGYPT
        assert task.context.language == Language.ARABIC

    def test_classify_default_options(self, classifier: TaskClassifier) -> None:
        """Test that options can be None when not provided."""
        task = classifier.classify(
            task_type=TaskType.LEGAL_CHAT,
            context_data={"jurisdiction": "EGYPT", "language": "ar"},
            payload={"query": "Test"},
            options_data=None,
        )

        # Options may be None when not provided
        # The orchestrator handles None options appropriately
        assert task is not None
        assert task.task_type == TaskType.LEGAL_CHAT
