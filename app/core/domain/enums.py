"""
Domain Enums

Core enumeration types used throughout the application.
These are pure domain concepts with no infrastructure dependencies.
"""

from enum import Enum


class TaskType(str, Enum):
    """
    Supported AI task types.

    Each task type maps to a specific workflow that handles
    the execution logic. Task types are explicitly declared
    in requests - never inferred.
    """

    LEGAL_CHAT = "LEGAL_CHAT"
    DOCUMENT_GENERATION = "DOCUMENT_GENERATION"
    CONTRACT_ANALYSIS = "CONTRACT_ANALYSIS"
    CONTRACT_REFRAMING = "CONTRACT_REFRAMING"
    CASE_EVALUATION = "CASE_EVALUATION"


class TaskStatus(str, Enum):
    """
    Task execution status.

    Represents the outcome of a task execution.
    """

    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


class Jurisdiction(str, Enum):
    """
    Supported legal jurisdictions.

    Defines the legal system context for task execution.
    """

    EGYPT = "egypt"
    UAE = "uae"
    SAUDI_ARABIA = "saudi_arabia"
    KUWAIT = "kuwait"
    QATAR = "qatar"
    BAHRAIN = "bahrain"
    OMAN = "oman"
    JORDAN = "jordan"


class Language(str, Enum):
    """
    Supported languages for responses.
    """

    ARABIC = "ar"
    ENGLISH = "en"


class LegalDomain(str, Enum):
    """
    Legal domain categories.

    Aligned with the Egyptian law database domains.
    """

    CIVIL = "civil"
    PENAL = "penal"
    COMMERCIAL = "commercial"
    LABOR = "labor"
    CONSTITUTION = "constitution"
    CRIMINAL_PROCEDURE = "criminal_procedure"
    PERSONAL = "personal"
    CHILD = "child"
    CONSUMER = "consumer"
    CYBER = "cyber"
    EDUCATION = "education"
    RENT = "rent"
