"""
Package Initialization for PDF Parser (Updated)
===============================================

This module initializes the parser package and provides convenient
imports for all processing components including the new specialized processors.
"""

from .document_analyzer import DocumentAnalyzer
from .section_intelligence import SectionIntelligence
from .content_classifier import ContentClassifier
from .table_extractor import TableExtractor
from .text_processor import TextProcessor
from .chart_detector import ChartDetector

__version__ = "1.0.0"
__author__ = "Professional Engineering Team"

__all__ = [
    'DocumentAnalyzer',
    'SectionIntelligence', 
    'ContentClassifier',
    'TableExtractor',
    'TextProcessor',  
    'ChartDetector'
]