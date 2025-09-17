"""
Document Analyzer - Core PDF Processing Engine (Updated)
========================================================

This module serves as the central orchestrator for PDF document analysis.
It coordinates with specialized processors (table_extractor, text_processor,
chart_detector) to provide comprehensive document parsing capabilities.

Updated to use modular processor architecture for better separation of concerns.
"""

import logging
import fitz  # PyMuPDF
import pdfplumber
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from utils.config import Config
from utils.helpers import sanitize_text, generate_block_id, calculate_file_size_mb


class DocumentAnalyzer:
    """
    Core document processing engine for PDF analysis with modular processors.
    
    This class implements the primary document processing logic, handling
    PDF loading, page extraction, and coordination of specialized processors.
    Updated to work with separate table_extractor, text_processor, and chart_detector modules.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the document analyzer with configuration.
        
        Args:
            config: Configuration object containing processing parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize processing statistics
        self.stats = {
            'pages_processed': 0,
            'blocks_extracted': 0,
            'tables_found': 0,
            'charts_detected': 0,
            'processing_errors': 0
        }
    
    def load_and_analyze_with_processors(self, pdf_path: str, table_extractor, text_processor, chart_detector) -> Dict[str, Any]:
        """
        Load PDF document and perform initial structural analysis with specialized processors.
        
        This method opens the PDF using multiple libraries and delegates specific
        processing tasks to specialized processor modules.
        
        Args:
            pdf_path: Path to the PDF file
            table_extractor: TableExtractor instance for table processing
            text_processor: TextProcessor instance for text cleaning
            chart_detector: ChartDetector instance for chart detection
            
        Returns:
            Dictionary containing document structure and metadata
        """
        self.logger.info(f"Loading PDF document with specialized processors: {pdf_path}")
        
        try:
            # Initialize document containers
            document_data = {
                'document_metadata': {
                    'source_file': pdf_path,
                    'file_size_mb': calculate_file_size_mb(pdf_path),
                    'total_pages': 0,
                    'confidence_scores': {}
                },
                'pages': []
            }
            
            # Open document with PyMuPDF for fast processing
            with fitz.open(pdf_path) as fitz_doc:
                document_data['document_metadata']['total_pages'] = len(fitz_doc)
                
                # Also open with pdfplumber for detailed analysis
                with pdfplumber.open(pdf_path) as plumber_doc:
                    
                    # Process each page with specialized processors
                    for page_num in range(len(fitz_doc)):
                        self.logger.debug(f"Processing page {page_num + 1}/{len(fitz_doc)} with modular processors")
                        
                        page_data = self._process_page_with_processors(
                            page_num + 1,
                            fitz_doc[page_num],
                            plumber_doc.pages[page_num],
                            table_extractor,
                            text_processor,
                            chart_detector
                        )
                        
                        document_data['pages'].append(page_data)
                        self.stats['pages_processed'] += 1
            
            # Add processing statistics to metadata
            document_data['document_metadata']['processing_stats'] = self.stats.copy()
            
            self.logger.info(f"Successfully loaded document with {document_data['document_metadata']['total_pages']} pages using modular processors")
            return document_data
            
        except Exception as e:
            self.logger.error(f"Failed to load PDF document: {str(e)}")
            raise
    
    def _process_page_with_processors(self, page_num: int, fitz_page, plumber_page, 
                                    table_extractor, text_processor, chart_detector) -> Dict[str, Any]:
        """
        Process a single page using specialized processor modules.
        
        This method coordinates between different processors to extract and
        classify content blocks on each page.
        
        Args:
            page_num: Page number (1-indexed)
            fitz_page: PyMuPDF page object
            plumber_page: pdfplumber page object
            table_extractor: TableExtractor instance
            text_processor: TextProcessor instance
            chart_detector: ChartDetector instance
            
        Returns:
            Dictionary containing page structure and content
        """
        page_data = {
            'page_number': page_num,
            'page_dimensions': {
                'width': float(fitz_page.rect.width),
                'height': float(fitz_page.rect.height)
            },
            'content_blocks': []
        }
        
        try:
            # Extract text blocks with position and font information
            text_blocks = self._extract_text_blocks_with_processor(fitz_page, text_processor)
            
            # Extract tables using the specialized table extractor
            table_blocks = self._extract_table_blocks_with_processor(plumber_page, table_extractor)
            
            # Extract images and potential charts using chart detector
            image_blocks = []
            if self.config.extract_images:
                image_blocks = self._extract_image_blocks_with_processor(fitz_page, chart_detector)
            
            # Combine all blocks and sort by reading order (top to bottom, left to right)
            all_blocks = text_blocks + table_blocks + image_blocks
            all_blocks.sort(key=lambda block: (block['position']['top'], block['position']['left']))
            
            # Assign reading order and block IDs
            for idx, block in enumerate(all_blocks, 1):
                block['block_id'] = generate_block_id(page_num, idx)
                block['reading_order'] = idx
            
            page_data['content_blocks'] = all_blocks
            self.stats['blocks_extracted'] += len(all_blocks)
            
        except Exception as e:
            self.logger.error(f"Error processing page {page_num}: {str(e)}")
            self.stats['processing_errors'] += 1
            
            # Create minimal page structure on error
            page_data['content_blocks'] = [{
                'block_id': generate_block_id(page_num, 1),
                'type': 'error',
                'text': f'Error processing page: {str(e)}',
                'confidence': 0.0,
                'reading_order': 1,
                'position': {'top': 0, 'left': 0, 'width': 0, 'height': 0}
            }]
        
        return page_data
    
    def _extract_text_blocks_with_processor(self, fitz_page, text_processor) -> List[Dict[str, Any]]:
        """
        Extract text blocks using PyMuPDF and clean them with TextProcessor.
        
        Args:
            fitz_page: PyMuPDF page object
            text_processor: TextProcessor instance for text cleaning
            
        Returns:
            List of text block dictionaries
        """
        text_blocks = []
        
        try:
            # Get text blocks with detailed information
            blocks = fitz_page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" not in block:  # Skip image blocks
                    continue
                
                # Extract text and formatting information
                block_text = ""
                font_sizes = []
                font_flags = []
                
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"]
                        font_sizes.append(span["size"])
                        font_flags.append(span["flags"])
                
                if block_text.strip():  # Only process non-empty blocks
                    # Clean text using TextProcessor
                    cleaned_text = text_processor.clean_text(block_text.strip())
                    
                    # Apply hyphen merging if configured
                    if self.config.merge_hyphenated_words:
                        cleaned_text = text_processor.merge_hyphens(cleaned_text)
                    
                    # Calculate average font metrics
                    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                    is_bold = any(flag & 2**4 for flag in font_flags)  # Bold flag
                    is_italic = any(flag & 2**1 for flag in font_flags)  # Italic flag
                    
                    text_block = {
                        'type': 'text',  # Will be refined by content classifier
                        'text': cleaned_text,
                        'position': {
                            'top': float(block['bbox'][1]),
                            'left': float(block['bbox'][0]),
                            'width': float(block['bbox'][2] - block['bbox'][0]),
                            'height': float(block['bbox'][3] - block['bbox'][1])
                        },
                        'formatting': {
                            'font_size': avg_font_size,
                            'is_bold': is_bold,
                            'is_italic': is_italic
                        },
                        'confidence': 0.9  # High confidence for text extraction
                    }
                    
                    text_blocks.append(text_block)
                    
        except Exception as e:
            self.logger.error(f"Error extracting text blocks: {str(e)}")
        
        return text_blocks
    
    def _extract_table_blocks_with_processor(self, plumber_page, table_extractor) -> List[Dict[str, Any]]:
        """
        Extract table structures using the specialized TableExtractor.
        
        Args:
            plumber_page: pdfplumber page object
            table_extractor: TableExtractor instance
            
        Returns:
            List of table block dictionaries
        """
        table_blocks = []
        
        try:
            # Use the specialized table extractor
            extracted_tables = table_extractor.extract_tables(plumber_page)
            
            for table_info in extracted_tables:
                table_data = table_info['table_data']
                
                # Filter out None rows and ensure minimum size
                clean_data = [row for row in table_data if row and any(cell for cell in row if cell)]
                
                if (len(clean_data) >= self.config.min_table_rows and 
                    len(clean_data[0]) >= self.config.min_table_cols):
                    
                    # Calculate table confidence based on data consistency
                    confidence = self._calculate_table_confidence(clean_data)
                    
                    if confidence >= self.config.table_confidence_threshold:
                        bbox = table_info['bbox']
                        
                        table_block = {
                            'type': 'table',
                            'table_data': clean_data,
                            'description': self._generate_table_description(clean_data),
                            'position': {
                                'top': float(bbox[1]),
                                'left': float(bbox[0]),
                                'width': float(bbox[2] - bbox[0]),
                                'height': float(bbox[3] - bbox[1])
                            },
                            'confidence': confidence,
                            'extraction_method': table_info['source']
                        }
                        
                        table_blocks.append(table_block)
                        self.stats['tables_found'] += 1
                        
        except Exception as e:
            self.logger.error(f"Error extracting tables: {str(e)}")
        
        return table_blocks
    
    def _extract_image_blocks_with_processor(self, fitz_page, chart_detector) -> List[Dict[str, Any]]:
        """
        Extract image blocks using the specialized ChartDetector.
        
        Args:
            fitz_page: PyMuPDF page object
            chart_detector: ChartDetector instance
            
        Returns:
            List of image/chart block dictionaries
        """
        image_blocks = []
        
        try:
            # Use the specialized chart detector
            detected_charts = chart_detector.detect_charts(fitz_page)
            
            for chart_info in detected_charts:
                image_block = {
                    'type': chart_info['type'],
                    'description': chart_info['description'],
                    'position': chart_info['bbox'],
                    'confidence': chart_info['confidence']
                }
                
                image_blocks.append(image_block)
                self.stats['charts_detected'] += 1
                        
        except Exception as e:
            self.logger.error(f"Error extracting images: {str(e)}")
        
        return image_blocks
    
    def _calculate_table_confidence(self, table_data: List[List[str]]) -> float:
        """
        Calculate confidence score for extracted table data.
        
        This method analyzes table structure and content to estimate
        the reliability of the extraction.
        
        Args:
            table_data: 2D list representing table content
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not table_data:
            return 0.0
        
        # Check for consistent column count
        col_counts = [len(row) for row in table_data]
        consistent_cols = len(set(col_counts)) == 1
        
        # Check for reasonable cell content
        non_empty_cells = sum(
            1 for row in table_data 
            for cell in row 
            if cell and str(cell).strip()
        )
        total_cells = sum(len(row) for row in table_data)
        fill_ratio = non_empty_cells / total_cells if total_cells > 0 else 0
        
        # Calculate base confidence
        confidence = 0.5
        
        if consistent_cols:
            confidence += 0.3
        
        if fill_ratio > 0.5:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _generate_table_description(self, table_data: List[List[str]]) -> str:
        """
        Generate a descriptive summary of table content.
        
        Args:
            table_data: 2D list representing table content
            
        Returns:
            Human-readable description of the table
        """
        if not table_data:
            return "Empty table"
        
        rows = len(table_data)
        cols = len(table_data[0]) if table_data else 0
        
        # Try to identify the nature of the table from headers
        if rows > 0 and table_data[0]:
            headers = [str(cell).lower() for cell in table_data[0] if cell]
            
            if any(term in ' '.join(headers) for term in ['financial', 'revenue', 'profit', 'amount']):
                return f"Financial data table with {rows} rows and {cols} columns"
            elif any(term in ' '.join(headers) for term in ['performance', 'metric', 'score']):
                return f"Performance metrics table with {rows} rows and {cols} columns"
            elif any(term in ' '.join(headers) for term in ['date', 'time', 'year', 'month']):
                return f"Time-series data table with {rows} rows and {cols} columns"
        
        return f"Data table with {rows} rows and {cols} columns"
    
    def apply_post_processing_with_processors(self, document_data: Dict[str, Any], text_processor) -> Dict[str, Any]:
        """
        Apply final post-processing enhancements using specialized processors.
        
        This method performs cleanup, validation, and enhancement operations
        on the fully processed document structure.
        
        Args:
            document_data: Processed document structure
            text_processor: TextProcessor instance for final text processing
            
        Returns:
            Enhanced document structure
        """
        self.logger.info("Applying post-processing enhancements with specialized processors...")
        
        try:
            # Calculate overall confidence scores
            document_data['document_metadata']['confidence_scores'] = self._calculate_overall_confidence(document_data)
            
            # Apply final text processing if requested
            if self.config.merge_hyphenated_words:
                self._apply_final_text_processing(document_data, text_processor)
            
            # Validate and clean the final structure
            self._validate_document_structure(document_data)
            
            self.logger.info("Post-processing completed successfully with specialized processors")
            
        except Exception as e:
            self.logger.error(f"Error during post-processing: {str(e)}")
        
        return document_data
    
    def _apply_final_text_processing(self, document_data: Dict[str, Any], text_processor) -> None:
        """
        Apply final text processing across all content blocks.
        
        Args:
            document_data: Document structure to process
            text_processor: TextProcessor instance
        """
        for page in document_data.get('pages', []):
            for block in page.get('content_blocks', []):
                if block.get('type') == 'paragraph' and 'text' in block:
                    # Apply final text processing
                    processed_text = text_processor.merge_hyphens(block['text'])
                    block['text'] = processed_text
    
    def _calculate_overall_confidence(self, document_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate overall confidence scores for the document processing.
        
        Args:
            document_data: Complete document structure
            
        Returns:
            Dictionary of confidence scores
        """
        all_confidences = []
        
        for page in document_data.get('pages', []):
            for block in page.get('content_blocks', []):
                if 'confidence' in block:
                    all_confidences.append(block['confidence'])
        
        if not all_confidences:
            return {'overall_structure': 0.5}
        
        overall_confidence = sum(all_confidences) / len(all_confidences)
        
        return {
            'overall_structure': round(overall_confidence, 3),
            'content_classification': round(overall_confidence * 0.95, 3),  # Slightly conservative
            'section_detection': round(overall_confidence * 0.90, 3)  # More conservative for sections
        }
    
    def _validate_document_structure(self, document_data: Dict[str, Any]) -> None:
        """
        Validate the final document structure for consistency and completeness.
        
        Args:
            document_data: Document structure to validate
        """
        # Ensure all required fields are present
        required_fields = ['document_metadata', 'pages']
        for field in required_fields:
            if field not in document_data:
                self.logger.warning(f"Missing required field: {field}")
                document_data[field] = {} if field == 'document_metadata' else []
        
        # Validate page structures
        for page_idx, page in enumerate(document_data.get('pages', [])):
            if 'page_number' not in page:
                page['page_number'] = page_idx + 1
            
            if 'content_blocks' not in page:
                page['content_blocks'] = []
        
        self.logger.debug("Document structure validation completed")