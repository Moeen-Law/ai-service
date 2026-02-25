"""
Utility Functions
"""

from app.shared.utils.arabic import (
    DOMAIN_NAMES_AR,
    extract_article_numbers_from_text,
    is_arabic,
    normalize_arabic,
    remove_diacritics,
    standardize_numbers,
)

__all__ = [
    "normalize_arabic",
    "remove_diacritics",
    "standardize_numbers",
    "extract_article_numbers_from_text",
    "is_arabic",
    "DOMAIN_NAMES_AR",
]
