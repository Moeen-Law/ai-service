"""
Unit Tests for Business Validator

Tests the business rule validation for each task type.
"""

import pytest
from uuid import uuid4

from app.core.domain.entities import Context, ExecutionOptions, Task
from app.core.domain.enums import TaskType, Jurisdiction, Language, LegalDomain
from app.core.validators.business import BusinessValidator
from app.shared.errors.exceptions import PayloadValidationError, ContextValidationError


class TestBusinessValidator:
    """Test suite for BusinessValidator."""

    @pytest.fixture
    def validator(self) -> BusinessValidator:
        """Create a validator instance."""
        return BusinessValidator()

    @pytest.fixture
    def base_context(self) -> Context:
        """Create a base context for tests."""
        return Context(
            jurisdiction=Jurisdiction.EGYPT,
            language=Language.ARABIC,
        )

    @pytest.fixture
    def context_with_domain(self) -> Context:
        """Create a context with domain for tests that require it."""
        return Context(
            jurisdiction=Jurisdiction.EGYPT,
            language=Language.ARABIC,
            domain=LegalDomain.COMMERCIAL,
        )

    def _create_task(
        self,
        task_type: TaskType,
        context: Context,
        payload: dict,
    ) -> Task:
        """Helper to create a Task entity."""
        return Task(
            task_id=uuid4(),
            task_type=task_type,
            context=context,
            payload=payload,
            options=ExecutionOptions(),
        )

    # --- LEGAL_CHAT Tests ---

    def test_legal_chat_valid_payload(
        self, validator: BusinessValidator, base_context: Context
    ) -> None:
        """Test valid LEGAL_CHAT payload passes validation."""
        task = self._create_task(
            TaskType.LEGAL_CHAT,
            base_context,
            {"message": "ما هي شروط الزواج المدني؟"},
        )
        # Should not raise
        validator.validate(task)

    def test_legal_chat_missing_message(
        self, validator: BusinessValidator, base_context: Context
    ) -> None:
        """Test LEGAL_CHAT without message fails validation."""
        task = self._create_task(TaskType.LEGAL_CHAT, base_context, {})

        with pytest.raises(PayloadValidationError) as exc_info:
            validator.validate(task)

        # Check that missing_fields includes "message"
        assert "missing_fields" in exc_info.value.details
        assert "message" in exc_info.value.details["missing_fields"]

    def test_legal_chat_empty_message(
        self, validator: BusinessValidator, base_context: Context
    ) -> None:
        """Test LEGAL_CHAT with empty message fails validation."""
        task = self._create_task(TaskType.LEGAL_CHAT, base_context, {"message": ""})

        with pytest.raises(PayloadValidationError) as exc_info:
            validator.validate(task)

        assert "message" in str(exc_info.value).lower()

    def test_legal_chat_with_files_ids_valid(
        self, validator: BusinessValidator, base_context: Context
    ) -> None:
        """Test LEGAL_CHAT with valid files_ids passes validation."""
        task = self._create_task(
            TaskType.LEGAL_CHAT,
            base_context,
            {
                "message": "حلل الملفات المرفوعة",
                "files_ids": ["file_1", "file_2"],
            },
        )

        validator.validate(task)

    def test_legal_chat_files_ids_must_be_list(
        self, validator: BusinessValidator, base_context: Context
    ) -> None:
        """Test LEGAL_CHAT rejects files_ids when not a list."""
        task = self._create_task(
            TaskType.LEGAL_CHAT,
            base_context,
            {
                "message": "حلل الملف",
                "files_ids": "file_1",
            },
        )

        with pytest.raises(PayloadValidationError) as exc_info:
            validator.validate(task)

        assert "files_ids" in str(exc_info.value).lower()

    def test_legal_chat_files_ids_rejects_blank_values(
        self, validator: BusinessValidator, base_context: Context
    ) -> None:
        """Test LEGAL_CHAT rejects blank file IDs."""
        task = self._create_task(
            TaskType.LEGAL_CHAT,
            base_context,
            {
                "message": "حلل الملف",
                "files_ids": ["file_1", "   "],
            },
        )

        with pytest.raises(PayloadValidationError) as exc_info:
            validator.validate(task)

        assert "non-empty" in str(exc_info.value).lower()

    # --- DOCUMENT_GENERATION Tests ---

    def test_document_generation_valid_payload(
        self, validator: BusinessValidator, base_context: Context
    ) -> None:
        """Test valid DOCUMENT_GENERATION payload passes validation."""
        task = self._create_task(
            TaskType.DOCUMENT_GENERATION,
            base_context,
            {
                "document_type": "contract",
                "parameters": {"party_a": "شركة أ", "party_b": "شركة ب"},
            },
        )
        validator.validate(task)

    def test_document_generation_missing_document_type(
        self, validator: BusinessValidator, base_context: Context
    ) -> None:
        """Test DOCUMENT_GENERATION without document_type fails."""
        task = self._create_task(
            TaskType.DOCUMENT_GENERATION,
            base_context,
            {"parameters": {}},
        )

        with pytest.raises(PayloadValidationError) as exc_info:
            validator.validate(task)

        assert "missing_fields" in exc_info.value.details
        assert "document_type" in exc_info.value.details["missing_fields"]

    # --- CONTRACT_ANALYSIS Tests ---

    def test_contract_analysis_valid_payload(
        self, validator: BusinessValidator, context_with_domain: Context
    ) -> None:
        """Test valid CONTRACT_ANALYSIS payload passes validation."""
        task = self._create_task(
            TaskType.CONTRACT_ANALYSIS,
            context_with_domain,  # Use context with domain
            {
                "contract_text": "نص العقد...",
                "analysis_type": "risk_assessment",
            },
        )
        validator.validate(task)

    def test_contract_analysis_missing_domain(
        self, validator: BusinessValidator, base_context: Context
    ) -> None:
        """Test CONTRACT_ANALYSIS without domain fails validation."""
        # base_context has no domain set
        task = self._create_task(
            TaskType.CONTRACT_ANALYSIS,
            base_context,
            {
                "contract_text": "نص العقد...",
                "analysis_type": "risk_assessment",
            },
        )

        with pytest.raises(ContextValidationError) as exc_info:
            validator.validate(task)

        assert "domain" in str(exc_info.value).lower()

    def test_contract_analysis_missing_text(
        self, validator: BusinessValidator, context_with_domain: Context
    ) -> None:
        """Test CONTRACT_ANALYSIS without contract_text fails."""
        task = self._create_task(
            TaskType.CONTRACT_ANALYSIS,
            context_with_domain,
            {"analysis_type": "risk_assessment"},
        )

        with pytest.raises(PayloadValidationError) as exc_info:
            validator.validate(task)

        assert "missing_fields" in exc_info.value.details
        assert "contract_text" in exc_info.value.details["missing_fields"]

    # --- CONTRACT_REFRAMING Tests ---

    def test_contract_reframing_valid_payload(
        self, validator: BusinessValidator, base_context: Context
    ) -> None:
        """Test valid CONTRACT_REFRAMING payload passes validation."""
        task = self._create_task(
            TaskType.CONTRACT_REFRAMING,
            base_context,
            {
                "clause_text": "البند الأول...",
                "target_perspective": "balanced",
            },
        )
        validator.validate(task)

    def test_contract_reframing_missing_clause(
        self, validator: BusinessValidator, base_context: Context
    ) -> None:
        """Test CONTRACT_REFRAMING without clause_text fails."""
        task = self._create_task(
            TaskType.CONTRACT_REFRAMING,
            base_context,
            {"target_perspective": "balanced"},
        )

        with pytest.raises(PayloadValidationError) as exc_info:
            validator.validate(task)

        assert "missing_fields" in exc_info.value.details
        assert "clause_text" in exc_info.value.details["missing_fields"]

    # --- CASE_EVALUATION Tests ---

    def test_case_evaluation_valid_payload(
        self, validator: BusinessValidator, context_with_domain: Context
    ) -> None:
        """Test valid CASE_EVALUATION payload passes validation."""
        task = self._create_task(
            TaskType.CASE_EVALUATION,
            context_with_domain,  # Case evaluation requires domain
            {"case_description": "وصف القضية..."},
        )
        validator.validate(task)

    def test_case_evaluation_missing_domain(
        self, validator: BusinessValidator, base_context: Context
    ) -> None:
        """Test CASE_EVALUATION without domain fails validation."""
        task = self._create_task(
            TaskType.CASE_EVALUATION,
            base_context,
            {"case_description": "وصف القضية..."},
        )

        with pytest.raises(ContextValidationError) as exc_info:
            validator.validate(task)

        assert "domain" in str(exc_info.value).lower()

    def test_case_evaluation_missing_description(
        self, validator: BusinessValidator, context_with_domain: Context
    ) -> None:
        """Test CASE_EVALUATION without case_description fails."""
        task = self._create_task(
            TaskType.CASE_EVALUATION,
            context_with_domain,
            {},
        )

        with pytest.raises(PayloadValidationError) as exc_info:
            validator.validate(task)

        assert "missing_fields" in exc_info.value.details
        assert "case_description" in exc_info.value.details["missing_fields"]
