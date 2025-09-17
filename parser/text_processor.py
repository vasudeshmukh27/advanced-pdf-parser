"""
Text Processor Module
======================

Handles all text cleaning, normalization, and advanced preprocessing.
Provides utilities to sanitize, normalize whitespace, and merge hyphenated words.
"""

import logging
import re
from typing import List, Dict, Any

from utils.config import Config
from utils.helpers import sanitize_text


class TextProcessor:
    """
    Advanced text processing utilities for PDF content.
    """
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        """
        Apply sanitization, whitespace normalization, and length trimming.
        """
        cleaned = sanitize_text(text, self.config.max_line_length)
        return cleaned

    def merge_hyphens(self, text: str) -> str:
        """
        Merge hyphenated words that split across lines.
        """
        text = text.replace('-\n', '')
        text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
        return text

    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple punctuation-based splitting.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
