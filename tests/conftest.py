"""
Shared test fixtures for pytest.
"""

import pytest
from uuid import uuid4

from app.core.domain.entities import Context, ExecutionOptions, Task
from app.core.domain.enums import (
    TaskType,
    TaskStatus,
    Jurisdiction,
    Language,
    LegalDomain,
)


@pytest.fixture
def egypt_arabic_context() -> Context:
    """Create an Egyptian Arabic context."""
    return Context(
        jurisdiction=Jurisdiction.EGYPT,
        language=Language.ARABIC,
    )


@pytest.fixture
def uae_english_context() -> Context:
    """Create a UAE English context."""
    return Context(
        jurisdiction=Jurisdiction.UAE,
        language=Language.ENGLISH,
    )


@pytest.fixture
def commercial_context() -> Context:
    """Create a context with commercial domain."""
    return Context(
        jurisdiction=Jurisdiction.EGYPT,
        language=Language.ARABIC,
        domain=LegalDomain.COMMERCIAL,
    )


@pytest.fixture
def default_options() -> ExecutionOptions:
    """Create default execution options."""
    return ExecutionOptions()


@pytest.fixture
def custom_options() -> ExecutionOptions:
    """Create custom execution options with specified values."""
    return ExecutionOptions(include_sources=False, max_tokens=1000)


@pytest.fixture
def legal_chat_task(egypt_arabic_context: Context) -> Task:
    """Create a LEGAL_CHAT task."""
    return Task(
        task_id=uuid4(),
        task_type=TaskType.LEGAL_CHAT,
        context=egypt_arabic_context,
        payload={"query": "ما هي شروط الزواج المدني؟"},
        options=ExecutionOptions(),
    )


@pytest.fixture
def contract_analysis_task(commercial_context: Context) -> Task:
    """Create a CONTRACT_ANALYSIS task."""
    return Task(
        task_id=uuid4(),
        task_type=TaskType.CONTRACT_ANALYSIS,
        context=commercial_context,
        payload={
            "contract_text": "نص العقد التجاري...",
            "analysis_type": "full",
        },
        options=ExecutionOptions(),
    )
