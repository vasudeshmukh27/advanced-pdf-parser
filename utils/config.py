"""
Configuration Management for PDF Parser
=======================================

This module centralizes all configuration settings and provides a clean
interface for managing processing parameters. It follows the principle
of configuration over convention, allowing fine-tuned control over
parsing behavior while maintaining sensible defaults.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class Config:
    """
    Configuration container for PDF parsing operations.
    
    This class encapsulates all configurable parameters used throughout
    the parsing pipeline, providing type safety and default values.
    """
    
    # Core processing parameters
    confidence_threshold: float = 0.75
    debug_mode: bool = False
    low_memory_mode: bool = False
    preserve_formatting: bool = False
    extract_images: bool = False
    use_camelot: bool = False  # New parameter for camelot table extraction
    
    # Section detection parameters
    font_size_threshold: float = 1.2  # Multiplier for detecting headers
    bold_weight_threshold: int = 600   # Font weight threshold for bold text
    section_patterns: List[str] = None # Regex patterns for section detection
    
    # Table detection parameters
    table_confidence_threshold: float = 0.7
    min_table_rows: int = 2
    min_table_cols: int = 2
    
    # Chart detection parameters
    image_analysis_enabled: bool = True
    chart_description_length: int = 150
    
    # Text processing parameters
    max_line_length: int = 1000
    cleanup_whitespace: bool = True
    merge_hyphenated_words: bool = True
    
    def __post_init__(self):
        """Initialize default section patterns if not provided."""
        if self.section_patterns is None:
            self.section_patterns = [
                r'^(\d+\.?\d*\.?\d*)\s+[A-Z][^.]*$',  # Numbered sections
                r'^[A-Z][A-Z\s]{2,}$',                # ALL CAPS headers
                r'^(Chapter|Section|Part)\s+\d+',      # Named sections
                r'^(Abstract|Introduction|Conclusion|References)$'  # Common sections
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            'confidence_threshold': self.confidence_threshold,
            'debug_mode': self.debug_mode,
            'low_memory_mode': self.low_memory_mode,
            'preserve_formatting': self.preserve_formatting,
            'extract_images': self.extract_images,
            'use_camelot': self.use_camelot,
            'font_size_threshold': self.font_size_threshold,
            'bold_weight_threshold': self.bold_weight_threshold,
            'table_confidence_threshold': self.table_confidence_threshold,
            'min_table_rows': self.min_table_rows,
            'min_table_cols': self.min_table_cols,
            'image_analysis_enabled': self.image_analysis_enabled,
            'chart_description_length': self.chart_description_length,
            'max_line_length': self.max_line_length,
            'cleanup_whitespace': self.cleanup_whitespace,
            'merge_hyphenated_words': self.merge_hyphenated_words
        }