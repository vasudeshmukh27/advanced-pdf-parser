"""
Utility Helper Functions for PDF Parser
=======================================

This module provides essential utility functions that support the main
parsing pipeline. These functions handle common tasks like logging setup,
file validation, and formatting operations in a reusable manner.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configure the logging system with appropriate formatters and handlers.
    
    This function sets up both console and file logging with different
    formats optimized for their respective output contexts. The console
    output is clean and user-friendly, while file logs include detailed
    debugging information.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
    """
    # Define formatters for different output types
    console_formatter = logging.Formatter(
        '%(levelname)s | %(message)s'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always use DEBUG for file logs
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def validate_file_paths(pdf_path: str, output_path: str) -> Tuple[bool, str]:
    """
    Validate input and output file paths for accessibility and permissions.
    
    This function performs comprehensive validation of file paths to catch
    potential issues early in the processing pipeline, providing clear
    error messages for common problems.
    
    Args:
        pdf_path: Path to the input PDF file
        output_path: Path for the output JSON file
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Raises:
        FileNotFoundError: If the input PDF doesn't exist
        PermissionError: If there are permission issues
        ValueError: If file extensions are incorrect
    """
    # Validate input PDF file
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        raise FileNotFoundError(f"Input PDF file not found: {pdf_path}")
    
    if not pdf_file.is_file():
        raise ValueError(f"Input path is not a file: {pdf_path}")
    
    if pdf_file.suffix.lower() != '.pdf':
        raise ValueError(f"Input file must be a PDF: {pdf_path}")
    
    if not os.access(pdf_file, os.R_OK):
        raise PermissionError(f"No read permission for input file: {pdf_path}")
    
    # Validate output path
    output_file = Path(output_path)
    
    if output_file.suffix.lower() != '.json':
        raise ValueError(f"Output file must have .json extension: {output_path}")
    
    # Check if output directory exists and is writable
    output_dir = output_file.parent
    
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"Cannot create output directory: {output_dir}")
    
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"No write permission for output directory: {output_dir}")
    
    # Check if output file already exists and warn
    if output_file.exists():
        logging.warning(f"Output file already exists and will be overwritten: {output_path}")
    
    return True, ""


def format_processing_time(seconds: float) -> str:
    """
    Format processing time in a human-readable format.
    
    Args:
        seconds: Processing time in seconds
        
    Returns:
        Formatted time string (e.g., "2m 35.2s" or "45.7s")
    """
    if seconds >= 60:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        return f"{seconds:.1f}s"


def calculate_file_size_mb(file_path: str) -> float:
    """
    Calculate file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB rounded to 2 decimal places
    """
    try:
        size_bytes = Path(file_path).stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
    except (OSError, FileNotFoundError):
        return 0.0


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Clean and sanitize text content for JSON serialization.
    
    This function handles common text processing issues including:
    - Removing or replacing non-printable characters
    - Normalizing whitespace
    - Truncating overly long strings
    - Handling encoding issues
    
    Args:
        text: Raw text content
        max_length: Optional maximum length for truncation
        
    Returns:
        Cleaned and sanitized text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove non-printable characters except common whitespace
    cleaned_text = ''.join(char for char in text if char.isprintable() or char in '\n\t\r ')
    
    # Normalize whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    
    # Truncate if necessary
    if max_length and len(cleaned_text) > max_length:
        cleaned_text = cleaned_text[:max_length-3] + '...'
    
    return cleaned_text


def generate_block_id(page_num: int, block_num: int) -> str:
    """
    Generate a unique identifier for content blocks.
    
    Args:
        page_num: Page number (1-indexed)
        block_num: Block number within the page (1-indexed)
        
    Returns:
        Unique block identifier (e.g., "p1_b3")
    """
    return f"p{page_num}_b{block_num}"


def estimate_confidence_score(metrics: dict) -> float:
    """
    Calculate an estimated confidence score based on various quality metrics.
    
    This function combines multiple quality indicators to produce a single
    confidence score that reflects the reliability of the extracted content.
    
    Args:
        metrics: Dictionary containing quality metrics
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    base_score = 0.5
    
    # Adjust based on text clarity
    if metrics.get('text_length', 0) > 10:
        base_score += 0.2
    
    # Adjust based on formatting consistency
    if metrics.get('consistent_formatting', False):
        base_score += 0.15
    
    # Adjust based on structural indicators
    if metrics.get('has_clear_boundaries', False):
        base_score += 0.15
    
    # Ensure score is within valid range
    return min(1.0, max(0.0, base_score))