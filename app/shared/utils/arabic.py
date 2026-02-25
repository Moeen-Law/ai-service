"""
Arabic Text Processing Utilities

Shared utilities for normalizing and processing Arabic legal text.
"""

import re
from typing import List


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for fuzzy matching.

    Collapses taa-marbuta / haa, alef variants, yaa / alef-maqsura,
    strips diacritics (tashkeel), and collapses whitespace.
    """
    t = text
    t = re.sub(r"[\u064B-\u065F\u0670]", "", t)  # remove tashkeel
    t = t.replace("ة", "ه")  # taa marbuta → haa
    t = t.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")  # alef variants
    t = t.replace("ى", "ي")  # alef maqsura → yaa
    t = re.sub(r"\s+", " ", t).strip()
    return t


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritics (harakat) from text."""
    return re.compile(r"[\u064B-\u065F\u0670]").sub("", text)


def standardize_numbers(text: str) -> str:
    """Convert Arabic-Indic and Eastern Arabic-Indic digits to Western digits."""
    arabic_indic = "٠١٢٣٤٥٦٧٨٩"
    eastern_arabic = "۰۱۲۳۴۵۶۷۸۹"
    western = "0123456789"
    trans_table = str.maketrans(arabic_indic + eastern_arabic, western * 2)
    return text.translate(trans_table)


def extract_article_numbers_from_text(question: str) -> List[str]:
    """
    Extract numbers that look like article numbers from a question,
    ignoring durations/monetary amounts.

    Rules:
      - Numbers preceded by المادة / مادة are always kept.
      - Bare numbers are kept only if NOT followed by a time/money unit.
    """
    _UNIT_WORDS = (
        r"(?:سنة|سنه|سنين|سنوات|أعوام|اعوام|عام|شهر|شهور|"
        r"أشهر|اشهر|أيام|ايام|يوم|جنيه|جنيهات|ألف|الف|مليون)"
    )

    nums: List[str] = []
    # 1. Numbers explicitly preceded by 'article' word in Arabic
    nums.extend(re.findall(r"الماد[ةه]\s+(\d+)", question))
    nums.extend(re.findall(r"ماد[ةه]\s+(\d+)", question))

    # 2. Bare numbers — only if NOT followed by a time/money unit
    for m in re.finditer(r"(?:^|\s)(\d+)(?=\s|$)", question):
        num = m.group(1)
        after = question[m.end() :].lstrip()
        if re.match(_UNIT_WORDS, after):
            continue
        if num not in nums:
            nums.append(num)

    return list(dict.fromkeys(nums))  # dedupe, preserve order


def is_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return any("\u0600" <= char <= "\u06ff" for char in text)


# Domain name mapping: code → Arabic display name
DOMAIN_NAMES_AR = {
    "penal": "قانون العقوبات",
    "constitution": "الدستور المصري",
    "labor": "قانون العمل",
    "civil": "القانون المدني",
    "commercial": "القانون التجاري",
    "criminal_procedure": "قانون الإجراءات الجنائية",
    "personal": "قانون الأحوال الشخصية",
    "rent": "قانون الإيجارات",
    "child": "قانون الطفل",
    "consumer": "قانون حماية المستهلك",
    "cyber": "قانون مكافحة جرائم تقنية المعلومات",
    "education": "قانون التعليم",
}
