"""
Section Intelligence - Advanced Hierarchical Structure Detection
================================================================

This module implements sophisticated algorithms for detecting document
sections, subsections, and hierarchical structure. It uses multiple
heuristics including font analysis, pattern matching, and contextual
clues to accurately identify document organization.

The intelligence system goes beyond simple font-size detection to provide
robust section identification even in complex document layouts.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict

from utils.config import Config
from utils.helpers import sanitize_text


class SectionIntelligence:
    """
    Advanced section detection and hierarchical structure analysis.
    
    This class implements multi-strategy section detection that combines
    font analysis, pattern matching, and contextual understanding to
    accurately identify document structure even in challenging layouts.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the section intelligence system.
        
        Args:
            config: Configuration object with detection parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Compile regex patterns for performance
        self.section_patterns = [re.compile(pattern) for pattern in config.section_patterns]
        
        # Track section hierarchy state
        self.current_section = None
        self.current_subsection = None
        self.section_stack = []
        
        # Font analysis cache
        self.font_statistics = {
            'sizes': defaultdict(int),
            'avg_size': 12.0,
            'header_threshold': 14.0
        }
    
    def detect_hierarchical_structure(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze and enhance document structure with section hierarchy.
        
        This is the main entry point for section detection. It processes
        the entire document to identify sections and subsections, then
        assigns appropriate labels to all content blocks.
        
        Args:
            document_data: Document structure from initial processing
            
        Returns:
            Enhanced document with section hierarchy information
        """
        self.logger.info("Beginning hierarchical structure detection...")
        
        try:
            # Phase 1: Analyze font characteristics across the document
            self._analyze_document_fonts(document_data)
            
            # Phase 2: Identify potential section headers
            header_candidates = self._identify_header_candidates(document_data)
            
            # Phase 3: Build section hierarchy
            section_hierarchy = self._build_section_hierarchy(header_candidates)
            
            # Phase 4: Assign sections to all content blocks
            enhanced_document = self._assign_sections_to_blocks(document_data, section_hierarchy)
            
            # Phase 5: Validate and refine the hierarchy
            final_document = self._refine_section_assignments(enhanced_document)
            
            self.logger.info(f"Section detection completed. Found {len(section_hierarchy)} main sections.")
            return final_document
            
        except Exception as e:
            self.logger.error(f"Error in section detection: {str(e)}")
            # Return original document with minimal section info
            return self._apply_fallback_sections(document_data)
    
    def _analyze_document_fonts(self, document_data: Dict[str, Any]) -> None:
        """
        Analyze font characteristics across the entire document.
        
        This method builds statistical profiles of font usage to inform
        header detection. It identifies common font sizes and establishes
        thresholds for header identification.
        
        Args:
            document_data: Document structure to analyze
        """
        self.logger.debug("Analyzing document font characteristics...")
        
        font_sizes = []
        
        for page in document_data.get('pages', []):
            for block in page.get('content_blocks', []):
                if block.get('type') == 'text' and 'formatting' in block:
                    font_size = block['formatting'].get('font_size', 12.0)
                    font_sizes.append(font_size)
                    self.font_statistics['sizes'][font_size] += 1
        
        if font_sizes:
            self.font_statistics['avg_size'] = sum(font_sizes) / len(font_sizes)
            
            # Calculate header threshold as a multiplier of average size
            self.font_statistics['header_threshold'] = (
                self.font_statistics['avg_size'] * self.config.font_size_threshold
            )
            
            self.logger.debug(f"Font analysis: avg={self.font_statistics['avg_size']:.1f}, "
                            f"header_threshold={self.font_statistics['header_threshold']:.1f}")
    
    def _identify_header_candidates(self, document_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify potential section headers using multiple detection strategies.
        
        This method applies various heuristics to identify text blocks that
        are likely to be section headers, including font analysis, pattern
        matching, and positional analysis.
        
        Args:
            document_data: Document structure to analyze
            
        Returns:
            List of header candidate dictionaries with confidence scores
        """
        self.logger.debug("Identifying section header candidates...")
        
        header_candidates = []
        
        for page in document_data.get('pages', []):
            for block in page.get('content_blocks', []):
                if block.get('type') != 'text':
                    continue
                
                text = block.get('text', '').strip()
                if not text:
                    continue
                
                # Calculate header probability using multiple factors
                header_score = self._calculate_header_probability(block, text)
                
                if header_score > 0.5:  # Threshold for header candidacy
                    candidate = {
                        'block': block,
                        'text': text,
                        'score': header_score,
                        'page_number': page['page_number'],
                        'header_level': self._estimate_header_level(block, header_score)
                    }
                    header_candidates.append(candidate)
        
        # Sort candidates by page and position
        header_candidates.sort(key=lambda x: (x['page_number'], x['block']['position']['top']))
        
        self.logger.debug(f"Found {len(header_candidates)} header candidates")
        return header_candidates
    
    def _calculate_header_probability(self, block: Dict[str, Any], text: str) -> float:
        """
        Calculate the probability that a text block is a section header.
        
        This method combines multiple signals to estimate header likelihood:
        - Font size relative to document average
        - Bold/italic formatting
        - Pattern matching against common header formats
        - Text length and characteristics
        - Position on page
        
        Args:
            block: Text block to analyze
            text: Text content of the block
            
        Returns:
            Probability score between 0.0 and 1.0
        """
        score = 0.0
        
        # Font size analysis
        if 'formatting' in block:
            font_size = block['formatting'].get('font_size', 12.0)
            if font_size >= self.font_statistics['header_threshold']:
                score += 0.3
            
            # Bold text often indicates headers
            if block['formatting'].get('is_bold', False):
                score += 0.2
        
        # Pattern matching against known header formats
        for pattern in self.section_patterns:
            if pattern.match(text):
                score += 0.3
                break
        
        # Text characteristics analysis
        word_count = len(text.split())
        
        # Headers are typically short and concise
        if 2 <= word_count <= 8:
            score += 0.15
        elif word_count > 20:
            score -= 0.1  # Very long text unlikely to be header
        
        # Capitalization patterns
        if text.isupper() and word_count <= 6:
            score += 0.2
        elif text.istitle():
            score += 0.1
        
        # Position analysis - headers often appear at top of content
        position = block.get('position', {})
        if position.get('top', 0) < 100:  # Near top of page
            score += 0.1
        
        # Reading order analysis - early blocks more likely to be headers
        reading_order = block.get('reading_order', 999)
        if reading_order <= 3:
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def _estimate_header_level(self, block: Dict[str, Any], score: float) -> int:
        """
        Estimate the hierarchical level of a header (1=main, 2=sub, etc.).
        
        Args:
            block: Header block to analyze
            score: Header probability score
            
        Returns:
            Estimated header level (1-4)
        """
        if 'formatting' not in block:
            return 2  # Default to subsection level
        
        font_size = block['formatting'].get('font_size', 12.0)
        is_bold = block['formatting'].get('is_bold', False)
        
        # Larger fonts and bold text suggest higher-level headers
        if font_size >= self.font_statistics['avg_size'] * 1.5 and is_bold:
            return 1  # Main section
        elif font_size >= self.font_statistics['avg_size'] * 1.3:
            return 2  # Subsection
        elif is_bold or score > 0.8:
            return 3  # Sub-subsection
        else:
            return 4  # Minor heading
    
    def _build_section_hierarchy(self, header_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build a hierarchical structure from identified header candidates.
        
        This method organizes header candidates into a tree-like structure
        that represents the document's logical organization.
        
        Args:
            header_candidates: List of potential headers with scores
            
        Returns:
            Hierarchical section structure
        """
        self.logger.debug("Building section hierarchy...")
        
        if not header_candidates:
            return []
        
        hierarchy = []
        section_stack = []
        
        for candidate in header_candidates:
            level = candidate['header_level']
            
            # Clean up the stack based on current header level
            while section_stack and section_stack[-1]['level'] >= level:
                section_stack.pop()
            
            # Create section entry
            section = {
                'title': candidate['text'],
                'level': level,
                'page_number': candidate['page_number'],
                'block_position': candidate['block']['position'],
                'confidence': candidate['score'],
                'subsections': []
            }
            
            # Add to hierarchy
            if level == 1 or not section_stack:
                hierarchy.append(section)
            else:
                # Find appropriate parent
                parent = section_stack[-1]
                parent['subsections'].append(section)
            
            section_stack.append(section)
        
        self.logger.debug(f"Built hierarchy with {len(hierarchy)} top-level sections")
        return hierarchy
    
    def _assign_sections_to_blocks(self, document_data: Dict[str, Any], 
                                  section_hierarchy: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assign section and subsection labels to all content blocks.
        
        This method maps every content block to its appropriate section
        based on the detected hierarchy and block positions.
        
        Args:
            document_data: Original document structure
            section_hierarchy: Detected section hierarchy
            
        Returns:
            Document with section assignments
        """
        self.logger.debug("Assigning sections to content blocks...")
        
        # Create a flattened list of sections with their ranges
        section_ranges = self._create_section_ranges(section_hierarchy)
        
        # Process each page and block
        for page in document_data.get('pages', []):
            page_num = page['page_number']
            
            for block in page.get('content_blocks', []):
                # Find the appropriate section for this block
                section_info = self._find_block_section(
                    block, page_num, section_ranges
                )
                
                # Assign section information
                block['section'] = section_info['section']
                block['sub_section'] = section_info['sub_section']
        
        return document_data
    
    def _create_section_ranges(self, hierarchy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create a flattened list of sections with their positional ranges.
        
        Args:
            hierarchy: Hierarchical section structure
            
        Returns:
            List of section ranges for block assignment
        """
        ranges = []
        
        def process_sections(sections: List[Dict[str, Any]], parent_title: str = None):
            for i, section in enumerate(sections):
                range_info = {
                    'title': section['title'],
                    'parent_title': parent_title,
                    'level': section['level'],
                    'start_page': section['page_number'],
                    'start_position': section['block_position']['top'],
                    'end_page': None,
                    'end_position': None,
                    'confidence': section['confidence']
                }
                
                # Determine end position (start of next section at same or higher level)
                if i + 1 < len(sections):
                    next_section = sections[i + 1]
                    range_info['end_page'] = next_section['page_number']
                    range_info['end_position'] = next_section['block_position']['top']
                
                ranges.append(range_info)
                
                # Process subsections
                if section['subsections']:
                    process_sections(section['subsections'], section['title'])
        
        process_sections(hierarchy)
        return ranges
    
    def _find_block_section(self, block: Dict[str, Any], page_num: int, 
                           section_ranges: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Find the appropriate section assignment for a content block.
        
        Args:
            block: Content block to assign
            page_num: Page number of the block
            section_ranges: List of section ranges
            
        Returns:
            Dictionary with section and sub_section assignments
        """
        block_position = block.get('position', {}).get('top', 0)
        
        # Find the most specific section that contains this block
        best_match = {'section': None, 'sub_section': None}
        best_specificity = 0
        
        for section_range in section_ranges:
            # Check if block falls within this section's range
            if self._block_in_section_range(block, page_num, section_range):
                # More specific (higher level) sections take precedence
                if section_range['level'] > best_specificity:
                    if section_range['level'] == 1:
                        best_match['section'] = section_range['title']
                        best_match['sub_section'] = None
                    else:
                        best_match['section'] = section_range['parent_title'] or section_range['title']
                        best_match['sub_section'] = section_range['title']
                    
                    best_specificity = section_range['level']
        
        return best_match
    
    def _block_in_section_range(self, block: Dict[str, Any], page_num: int, 
                              section_range: Dict[str, Any]) -> bool:
        """
        Check if a block falls within a section's range.
        
        Args:
            block: Content block to check
            page_num: Page number of the block
            section_range: Section range definition
            
        Returns:
            True if block is within the section range
        """
        block_position = block.get('position', {}).get('top', 0)
        
        # Check page boundaries
        if page_num < section_range['start_page']:
            return False
        
        if section_range['end_page'] and page_num > section_range['end_page']:
            return False
        
        # Check position boundaries
        if page_num == section_range['start_page']:
            if block_position < section_range['start_position']:
                return False
        
        if (section_range['end_page'] and 
            page_num == section_range['end_page'] and
            section_range['end_position'] and
            block_position >= section_range['end_position']):
            return False
        
        return True
    
    def _refine_section_assignments(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine and validate section assignments using contextual analysis.
        
        This method performs post-processing to improve section assignments
        by analyzing context and correcting obvious errors.
        
        Args:
            document_data: Document with initial section assignments
            
        Returns:
            Document with refined section assignments
        """
        self.logger.debug("Refining section assignments...")
        
        # Track section usage statistics for validation
        section_stats = defaultdict(int)
        
        for page in document_data.get('pages', []):
            for block in page.get('content_blocks', []):
                section = block.get('section')
                if section:
                    section_stats[section] += 1
        
        # Apply refinements based on statistical analysis
        self._apply_statistical_refinements(document_data, section_stats)
        
        return document_data
    
    def _apply_statistical_refinements(self, document_data: Dict[str, Any], 
                                     section_stats: Dict[str, int]) -> None:
        """
        Apply statistical refinements to improve section assignments.
        
        Args:
            document_data: Document structure to refine
            section_stats: Statistics about section usage
        """
        # Identify sections that appear too infrequently (likely errors)
        min_section_size = 2
        problematic_sections = {
            section for section, count in section_stats.items() 
            if count < min_section_size
        }
        
        if problematic_sections:
            self.logger.debug(f"Refining {len(problematic_sections)} problematic sections")
            
            # Reassign blocks from problematic sections
            for page in document_data.get('pages', []):
                for block in page.get('content_blocks', []):
                    if block.get('section') in problematic_sections:
                        # Use fallback section assignment
                        block['section'] = self._get_fallback_section(block, page)
    
    def _get_fallback_section(self, block: Dict[str, Any], page: Dict[str, Any]) -> str:
        """
        Get a fallback section assignment for blocks without clear sections.
        
        Args:
            block: Content block needing section assignment
            page: Page containing the block
            
        Returns:
            Fallback section name
        """
        page_num = page['page_number']
        
        # Use page-based fallback sections
        if page_num == 1:
            return "Document Header"
        elif page_num <= 3:
            return "Introduction"
        else:
            return "Main Content"
    
    def _apply_fallback_sections(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply minimal fallback section assignments when detection fails.
        
        Args:
            document_data: Document structure
            
        Returns:
            Document with basic section assignments
        """
        self.logger.warning("Applying fallback section assignments due to detection failure")
        
        for page in document_data.get('pages', []):
            for block in page.get('content_blocks', []):
                block['section'] = self._get_fallback_section(block, page)
                block['sub_section'] = None
        
        return document_data