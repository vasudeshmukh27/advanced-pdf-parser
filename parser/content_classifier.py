"""
Content Classifier - Intelligent Content Type Detection
=======================================================

This module implements advanced algorithms for classifying content blocks
into appropriate types (paragraph, table, chart, list, etc.). It uses
multiple analysis techniques including structural analysis, content
pattern recognition, and contextual clues.

The classifier provides confidence scores for each classification to
support quality assessment and downstream processing decisions.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

from utils.config import Config
from utils.helpers import estimate_confidence_score


class ContentClassifier:
    """
    Intelligent content type classification system.
    
    This class analyzes content blocks and assigns appropriate type
    classifications with confidence scores. It uses multiple classification
    strategies to handle various content types accurately.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the content classifier.
        
        Args:
            config: Configuration object with classification parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Classification patterns and rules
        self.list_patterns = [
            re.compile(r'^\s*[\u2022\u2023\u25E6\u2043\u2219]\s+'),  # Bullet points
            re.compile(r'^\s*\d+[\.\)]\s+'),  # Numbered lists
            re.compile(r'^\s*[a-zA-Z][\.\)]\s+'),  # Lettered lists
            re.compile(r'^\s*[ivxlcdm]+[\.\)]\s+', re.IGNORECASE),  # Roman numerals
        ]
        
        # Chart/diagram keywords
        self.chart_keywords = {
            'figure', 'chart', 'graph', 'diagram', 'plot', 'visualization',
            'image', 'picture', 'illustration', 'map', 'timeline'
        }
        
        # Processing statistics
        self.classification_stats = {
            'paragraphs': 0,
            'tables': 0,
            'charts': 0,
            'lists': 0,
            'headers': 0,
            'other': 0
        }
    
    def classify_content_blocks(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify all content blocks in the document.
        
        This is the main entry point for content classification. It processes
        all content blocks and assigns appropriate type classifications with
        confidence scores.
        
        Args:
            document_data: Document structure with initial content blocks
            
        Returns:
            Document with enhanced content type classifications
        """
        self.logger.info("Beginning content type classification...")
        
        try:
            # Process each page and its content blocks
            for page in document_data.get('pages', []):
                self._classify_page_content(page)
            
            # Apply post-classification refinements
            self._apply_classification_refinements(document_data)
            
            # Update document metadata with classification statistics
            document_data['document_metadata']['classification_stats'] = self.classification_stats.copy()
            
            self.logger.info(f"Classification completed: {self.classification_stats}")
            return document_data
            
        except Exception as e:
            self.logger.error(f"Error in content classification: {str(e)}")
            return document_data
    
    def _classify_page_content(self, page: Dict[str, Any]) -> None:
        """
        Classify content blocks within a single page.
        
        Args:
            page: Page dictionary containing content blocks
        """
        content_blocks = page.get('content_blocks', [])
        
        for block in content_blocks:
            # Skip blocks that already have specific types (e.g., tables from extraction)
            if block.get('type') in ['table', 'chart', 'image']:
                self._update_classification_stats(block['type'])
                continue
            
            # Classify text-based content
            if block.get('type') == 'text' or 'text' in block:
                self._classify_text_block(block)
    
    def _classify_text_block(self, block: Dict[str, Any]) -> None:
        """
        Classify a text-based content block.
        
        This method applies various classification strategies to determine
        the most appropriate content type for a text block.
        
        Args:
            block: Text block to classify
        """
        text = block.get('text', '').strip()
        if not text:
            block['type'] = 'empty'
            block['confidence'] = 0.1
            return
        
        # Apply classification strategies in order of specificity
        classification_results = []
        
        # Strategy 1: Header detection
        header_result = self._classify_as_header(block, text)
        if header_result['confidence'] > 0.5:
            classification_results.append(header_result)
        
        # Strategy 2: List detection
        list_result = self._classify_as_list(block, text)
        if list_result['confidence'] > 0.3:
            classification_results.append(list_result)
        
        # Strategy 3: Chart/diagram reference detection
        chart_result = self._classify_as_chart_reference(block, text)
        if chart_result['confidence'] > 0.3:
            classification_results.append(chart_result)
        
        # Strategy 4: Paragraph classification (default)
        paragraph_result = self._classify_as_paragraph(block, text)
        classification_results.append(paragraph_result)
        
        # Select the best classification
        best_classification = max(classification_results, key=lambda x: x['confidence'])
        
        # Apply classification to block
        block['type'] = best_classification['type']
        block['confidence'] = best_classification['confidence']
        
        # Add additional metadata
        if 'metadata' in best_classification:
            block.update(best_classification['metadata'])
        
        self._update_classification_stats(block['type'])
    
    def _classify_as_header(self, block: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Evaluate if the block should be classified as a header.
        
        Args:
            block: Content block to evaluate
            text: Text content of the block
            
        Returns:
            Classification result with confidence score
        """
        confidence = 0.0
        
        # Check formatting indicators
        formatting = block.get('formatting', {})
        
        # Font size analysis
        font_size = formatting.get('font_size', 12.0)
        if font_size > 14:
            confidence += 0.3
        
        # Bold text often indicates headers
        if formatting.get('is_bold', False):
            confidence += 0.2
        
        # Text characteristics
        word_count = len(text.split())
        
        # Headers are typically concise
        if 2 <= word_count <= 10:
            confidence += 0.2
        elif word_count > 20:
            confidence -= 0.2
        
        # Capitalization patterns
        if text.isupper() and word_count <= 6:
            confidence += 0.3
        elif text.istitle():
            confidence += 0.1
        
        # Position indicators (headers often at top)
        position = block.get('position', {})
        if position.get('top', 0) < 100:
            confidence += 0.1
        
        # Reading order (early blocks more likely to be headers)
        reading_order = block.get('reading_order', 999)
        if reading_order <= 2:
            confidence += 0.1
        
        return {
            'type': 'header',
            'confidence': min(1.0, max(0.0, confidence))
        }
    
    def _classify_as_list(self, block: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Evaluate if the block should be classified as a list.
        
        Args:
            block: Content block to evaluate
            text: Text content of the block
            
        Returns:
            Classification result with confidence score
        """
        confidence = 0.0
        list_type = 'unordered'
        
        # Check for list patterns
        lines = text.split('\n')
        matching_lines = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check each list pattern
            for pattern in self.list_patterns:
                if pattern.match(line):
                    matching_lines += 1
                    
                    # Determine list type
                    if re.match(r'^\s*\d+[\.\)]', line):
                        list_type = 'ordered'
                    elif re.match(r'^\s*[a-zA-Z][\.\)]', line):
                        list_type = 'alphabetic'
                    elif re.match(r'^\s*[ivxlcdm]+[\.\)]', line, re.IGNORECASE):
                        list_type = 'roman'
                    break
        
        # Calculate confidence based on pattern matching
        total_lines = len([line for line in lines if line.strip()])
        if total_lines > 0:
            match_ratio = matching_lines / total_lines
            
            if match_ratio >= 0.8:
                confidence = 0.9
            elif match_ratio >= 0.6:
                confidence = 0.7
            elif match_ratio >= 0.4:
                confidence = 0.5
            elif match_ratio >= 0.2:
                confidence = 0.3
        
        return {
            'type': 'list',
            'confidence': confidence,
            'metadata': {
                'list_type': list_type,
                'item_count': matching_lines
            }
        }
    
    def _classify_as_chart_reference(self, block: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Evaluate if the block references a chart or diagram.
        
        Args:
            block: Content block to evaluate
            text: Text content of the block
            
        Returns:
            Classification result with confidence score
        """
        confidence = 0.0
        
        # Convert text to lowercase for keyword matching
        text_lower = text.lower()
        
        # Check for chart-related keywords
        keyword_matches = sum(1 for keyword in self.chart_keywords if keyword in text_lower)
        
        if keyword_matches > 0:
            confidence += 0.4 * min(1.0, keyword_matches / 2)
        
        # Check for figure/table references
        figure_patterns = [
            r'figure\s+\d+',
            r'fig\.\s*\d+',
            r'chart\s+\d+',
            r'graph\s+\d+',
            r'diagram\s+\d+'
        ]
        
        for pattern in figure_patterns:
            if re.search(pattern, text_lower):
                confidence += 0.3
                break
        
        # Short text with chart keywords likely refers to a chart
        word_count = len(text.split())
        if word_count < 10 and keyword_matches > 0:
            confidence += 0.2
        
        return {
            'type': 'chart_reference',
            'confidence': confidence
        }
    
    def _classify_as_paragraph(self, block: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Evaluate the block as a regular paragraph.
        
        This is the default classification for text content that doesn't
        match other specific patterns.
        
        Args:
            block: Content block to evaluate
            text: Text content of the block
            
        Returns:
            Classification result with confidence score
        """
        confidence = 0.5  # Base confidence for paragraph classification
        
        # Text characteristics that support paragraph classification
        word_count = len(text.split())
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Longer text with multiple sentences is likely a paragraph
        if word_count > 20 and sentence_count > 1:
            confidence += 0.3
        elif word_count > 10:
            confidence += 0.2
        
        # Check for paragraph-like formatting
        formatting = block.get('formatting', {})
        
        # Regular font size suggests paragraph content
        font_size = formatting.get('font_size', 12.0)
        if 10 <= font_size <= 14:
            confidence += 0.1
        
        # Non-bold text is typical for paragraphs
        if not formatting.get('is_bold', False):
            confidence += 0.1
        
        return {
            'type': 'paragraph',
            'confidence': min(1.0, confidence)
        }
    
    def _apply_classification_refinements(self, document_data: Dict[str, Any]) -> None:
        """
        Apply post-classification refinements to improve accuracy.
        
        This method analyzes the overall classification results and applies
        corrections based on document-wide patterns and context.
        
        Args:
            document_data: Document with initial classifications
        """
        self.logger.debug("Applying classification refinements...")
        
        # Collect classification statistics for analysis
        type_distribution = Counter()
        
        for page in document_data.get('pages', []):
            for block in page.get('content_blocks', []):
                block_type = block.get('type', 'unknown')
                type_distribution[block_type] += 1
        
        # Apply context-based refinements
        self._refine_header_classifications(document_data, type_distribution)
        self._refine_list_classifications(document_data)
        self._merge_related_blocks(document_data)
    
    def _refine_header_classifications(self, document_data: Dict[str, Any], 
                                     type_distribution: Counter) -> None:
        """
        Refine header classifications based on document context.
        
        Args:
            document_data: Document structure
            type_distribution: Distribution of content types
        """
        # If too many headers detected, increase threshold
        total_blocks = sum(type_distribution.values())
        header_ratio = type_distribution.get('header', 0) / max(total_blocks, 1)
        
        if header_ratio > 0.3:  # More than 30% headers seems excessive
            self.logger.debug("High header ratio detected, applying stricter criteria")
            
            for page in document_data.get('pages', []):
                for block in page.get('content_blocks', []):
                    if (block.get('type') == 'header' and 
                        block.get('confidence', 0) < 0.7):
                        # Reclassify as paragraph
                        block['type'] = 'paragraph'
                        block['confidence'] = 0.6
    
    def _refine_list_classifications(self, document_data: Dict[str, Any]) -> None:
        """
        Refine list classifications by merging adjacent list blocks.
        
        Args:
            document_data: Document structure
        """
        for page in document_data.get('pages', []):
            blocks = page.get('content_blocks', [])
            
            # Find consecutive list blocks that should be merged
            i = 0
            while i < len(blocks) - 1:
                current_block = blocks[i]
                next_block = blocks[i + 1]
                
                if (current_block.get('type') == 'list' and 
                    next_block.get('type') == 'list' and
                    self._should_merge_list_blocks(current_block, next_block)):
                    
                    # Merge the blocks
                    merged_block = self._merge_list_blocks(current_block, next_block)
                    
                    # Replace the two blocks with the merged one
                    blocks[i] = merged_block
                    blocks.pop(i + 1)
                    
                    # Don't increment i to check for further merges
                else:
                    i += 1
    
    def _should_merge_list_blocks(self, block1: Dict[str, Any], block2: Dict[str, Any]) -> bool:
        """
        Determine if two list blocks should be merged.
        
        Args:
            block1: First list block
            block2: Second list block
            
        Returns:
            True if blocks should be merged
        """
        # Check if list types match
        list_type1 = block1.get('list_type', 'unordered')
        list_type2 = block2.get('list_type', 'unordered')
        
        if list_type1 != list_type2:
            return False
        
        # Check if blocks are adjacent (within reasonable distance)
        pos1 = block1.get('position', {})
        pos2 = block2.get('position', {})
        
        vertical_gap = pos2.get('top', 0) - (pos1.get('top', 0) + pos1.get('height', 0))
        
        # Merge if gap is small (less than 20 pixels)
        return vertical_gap < 20
    
    def _merge_list_blocks(self, block1: Dict[str, Any], block2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two list blocks into a single block.
        
        Args:
            block1: First list block
            block2: Second list block
            
        Returns:
            Merged list block
        """
        merged_text = block1.get('text', '') + '\n' + block2.get('text', '')
        
        # Calculate combined position
        pos1 = block1.get('position', {})
        pos2 = block2.get('position', {})
        
        merged_position = {
            'top': pos1.get('top', 0),
            'left': min(pos1.get('left', 0), pos2.get('left', 0)),
            'width': max(pos1.get('width', 0), pos2.get('width', 0)),
            'height': (pos2.get('top', 0) + pos2.get('height', 0)) - pos1.get('top', 0)
        }
        
        # Combine item counts
        item_count1 = block1.get('item_count', 0)
        item_count2 = block2.get('item_count', 0)
        
        return {
            'type': 'list',
            'text': merged_text,
            'position': merged_position,
            'confidence': min(block1.get('confidence', 0.5), block2.get('confidence', 0.5)),
            'list_type': block1.get('list_type', 'unordered'),
            'item_count': item_count1 + item_count2,
            'block_id': block1.get('block_id', ''),  # Keep first block's ID
            'reading_order': block1.get('reading_order', 0)
        }
    
    def _merge_related_blocks(self, document_data: Dict[str, Any]) -> None:
        """
        Merge related content blocks that should be combined.
        
        This method identifies blocks that are logically related and
        should be presented as a single unit.
        
        Args:
            document_data: Document structure
        """
        # This is a placeholder for more sophisticated block merging logic
        # In a production system, this could include:
        # - Merging split paragraphs
        # - Combining caption and chart blocks
        # - Merging broken table segments
        pass
    
    def _update_classification_stats(self, content_type: str) -> None:
        """
        Update classification statistics.
        
        Args:
            content_type: Type of content classified
        """
        if content_type in self.classification_stats:
            self.classification_stats[content_type] += 1
        else:
            self.classification_stats['other'] += 1