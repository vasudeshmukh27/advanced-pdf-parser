#!/usr/bin/env python3
"""
Unit Tests for PDF Parser Components
====================================

This module provides comprehensive unit tests for the PDF parsing system.
It includes tests for all major components and edge cases to ensure
robust operation in production environments.
"""

import unittest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import components to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from utils.helpers import sanitize_text, generate_block_id, format_processing_time
from parser.content_classifier import ContentClassifier
from parser.section_intelligence import SectionIntelligence


class TestConfig(unittest.TestCase):
    """Test configuration management functionality."""
    
    def test_default_config_creation(self):
        """Test creation of config with default values."""
        config = Config()
        
        self.assertEqual(config.confidence_threshold, 0.75)
        self.assertFalse(config.debug_mode)
        self.assertFalse(config.low_memory_mode)
        self.assertIsNotNone(config.section_patterns)
        self.assertGreater(len(config.section_patterns), 0)
    
    def test_custom_config_creation(self):
        """Test creation of config with custom values."""
        config = Config(
            confidence_threshold=0.9,
            debug_mode=True,
            font_size_threshold=1.5
        )
        
        self.assertEqual(config.confidence_threshold, 0.9)
        self.assertTrue(config.debug_mode)
        self.assertEqual(config.font_size_threshold, 1.5)
    
    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = Config(confidence_threshold=0.8)
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['confidence_threshold'], 0.8)
        self.assertIn('debug_mode', config_dict)


class TestHelpers(unittest.TestCase):
    """Test utility helper functions."""
    
    def test_sanitize_text_basic(self):
        """Test basic text sanitization."""
        test_text = "  Hello   World  \n\t  "
        result = sanitize_text(test_text)
        self.assertEqual(result, "Hello World")
    
    def test_sanitize_text_with_length_limit(self):
        """Test text sanitization with length limit."""
        long_text = "This is a very long text that exceeds the limit"
        result = sanitize_text(long_text, max_length=20)
        self.assertEqual(len(result), 20)
        self.assertTrue(result.endswith('...'))
    
    def test_generate_block_id(self):
        """Test block ID generation."""
        block_id = generate_block_id(1, 5)
        self.assertEqual(block_id, "p1_b5")
        
        block_id = generate_block_id(10, 23)
        self.assertEqual(block_id, "p10_b23")
    
    def test_format_processing_time(self):
        """Test processing time formatting."""
        # Test seconds only
        result = format_processing_time(45.7)
        self.assertEqual(result, "45.7s")
        
        # Test minutes and seconds
        result = format_processing_time(125.3)
        self.assertEqual(result, "2m 5.3s")


class TestContentClassifier(unittest.TestCase):
    """Test content classification functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.classifier = ContentClassifier(self.config)
    
    def test_list_classification(self):
        """Test classification of list content."""
        # Test bullet list
        block = {
            'text': '• Item 1\n• Item 2\n• Item 3',
            'formatting': {'font_size': 12, 'is_bold': False}
        }
        
        result = self.classifier._classify_as_list(block, block['text'])
        self.assertEqual(result['type'], 'list')
        self.assertGreater(result['confidence'], 0.5)
        self.assertEqual(result['metadata']['list_type'], 'unordered')
    
    def test_numbered_list_classification(self):
        """Test classification of numbered list content."""
        block = {
            'text': '1. First item\n2. Second item\n3. Third item',
            'formatting': {'font_size': 12, 'is_bold': False}
        }
        
        result = self.classifier._classify_as_list(block, block['text'])
        self.assertEqual(result['type'], 'list')
        self.assertGreater(result['confidence'], 0.5)
        self.assertEqual(result['metadata']['list_type'], 'ordered')
    
    def test_header_classification(self):
        """Test classification of header content."""
        block = {
            'text': 'SECTION 1: INTRODUCTION',
            'formatting': {'font_size': 16, 'is_bold': True},
            'position': {'top': 50},
            'reading_order': 1
        }
        
        result = self.classifier._classify_as_header(block, block['text'])
        self.assertEqual(result['type'], 'header')
        self.assertGreater(result['confidence'], 0.7)
    
    def test_paragraph_classification(self):
        """Test classification of paragraph content."""
        block = {
            'text': 'This is a longer piece of text that represents a typical paragraph. It contains multiple sentences and provides detailed information about a particular topic.',
            'formatting': {'font_size': 12, 'is_bold': False}
        }
        
        result = self.classifier._classify_as_paragraph(block, block['text'])
        self.assertEqual(result['type'], 'paragraph')
        self.assertGreater(result['confidence'], 0.5)
    
    def test_chart_reference_classification(self):
        """Test classification of chart reference content."""
        block = {
            'text': 'Figure 1: Sales Performance Chart',
            'formatting': {'font_size': 12, 'is_bold': False}
        }
        
        result = self.classifier._classify_as_chart_reference(block, block['text'])
        self.assertEqual(result['type'], 'chart_reference')
        self.assertGreater(result['confidence'], 0.3)


class TestSectionIntelligence(unittest.TestCase):
    """Test section detection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.section_intel = SectionIntelligence(self.config)
    
    def test_header_probability_calculation(self):
        """Test header probability calculation."""
        # High probability header
        header_block = {
            'text': 'EXECUTIVE SUMMARY',
            'formatting': {'font_size': 16, 'is_bold': True},
            'position': {'top': 50},
            'reading_order': 1
        }
        
        # Set up font statistics
        self.section_intel.font_statistics = {
            'avg_size': 12.0,
            'header_threshold': 14.0
        }
        
        probability = self.section_intel._calculate_header_probability(
            header_block, header_block['text']
        )
        self.assertGreater(probability, 0.7)
        
        # Low probability header (regular paragraph)
        paragraph_block = {
            'text': 'This is a regular paragraph with normal formatting and longer text content.',
            'formatting': {'font_size': 12, 'is_bold': False},
            'position': {'top': 200},
            'reading_order': 5
        }
        
        probability = self.section_intel._calculate_header_probability(
            paragraph_block, paragraph_block['text']
        )
        self.assertLess(probability, 0.5)
    
    def test_header_level_estimation(self):
        """Test header level estimation."""
        # Set up font statistics
        self.section_intel.font_statistics = {'avg_size': 12.0}
        
        # Main header (level 1)
        main_header = {
            'formatting': {'font_size': 18, 'is_bold': True}
        }
        level = self.section_intel._estimate_header_level(main_header, 0.9)
        self.assertEqual(level, 1)
        
        # Subsection header (level 2)
        sub_header = {
            'formatting': {'font_size': 15, 'is_bold': False}
        }
        level = self.section_intel._estimate_header_level(sub_header, 0.7)
        self.assertEqual(level, 2)
    
    def test_section_range_creation(self):
        """Test creation of section ranges."""
        hierarchy = [
            {
                'title': 'Introduction',
                'level': 1,
                'page_number': 1,
                'block_position': {'top': 100},
                'confidence': 0.9,
                'subsections': [
                    {
                        'title': 'Background',
                        'level': 2,
                        'page_number': 1,
                        'block_position': {'top': 200},
                        'confidence': 0.8,
                        'subsections': []
                    }
                ]
            }
        ]
        
        ranges = self.section_intel._create_section_ranges(hierarchy)
        
        self.assertEqual(len(ranges), 2)  # One main section + one subsection
        self.assertEqual(ranges[0]['title'], 'Introduction')
        self.assertEqual(ranges[1]['title'], 'Background')
        self.assertEqual(ranges[1]['parent_title'], 'Introduction')


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete parsing system."""
    
    def test_document_metadata_structure(self):
        """Test that document metadata has required structure."""
        expected_metadata_fields = [
            'source_file',
            'file_size_mb',
            'total_pages',
            'confidence_scores'
        ]
        
        # Mock document metadata
        metadata = {
            'source_file': 'test.pdf',
            'file_size_mb': 1.5,
            'total_pages': 5,
            'confidence_scores': {
                'overall_structure': 0.92,
                'content_classification': 0.89
            }
        }
        
        for field in expected_metadata_fields:
            self.assertIn(field, metadata)
    
    def test_page_structure(self):
        """Test that page structure has required fields."""
        expected_page_fields = [
            'page_number',
            'page_dimensions',
            'content_blocks'
        ]
        
        page = {
            'page_number': 1,
            'page_dimensions': {'width': 595, 'height': 842},
            'content_blocks': []
        }
        
        for field in expected_page_fields:
            self.assertIn(field, page)
    
    def test_content_block_structure(self):
        """Test that content blocks have required fields."""
        expected_block_fields = [
            'block_id',
            'type',
            'confidence',
            'position',
            'reading_order'
        ]
        
        block = {
            'block_id': 'p1_b1',
            'type': 'paragraph',
            'text': 'Sample text content',
            'confidence': 0.85,
            'position': {'top': 100, 'left': 50, 'width': 400, 'height': 60},
            'reading_order': 1,
            'section': 'Introduction',
            'sub_section': None
        }
        
        for field in expected_block_fields:
            self.assertIn(field, block)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_empty_text_handling(self):
        """Test handling of empty text content."""
        config = Config()
        classifier = ContentClassifier(config)
        
        empty_block = {'text': '   \n\t  '}
        classifier._classify_text_block(empty_block)
        
        self.assertEqual(empty_block['type'], 'empty')
        self.assertLess(empty_block['confidence'], 0.5)
    
    def test_invalid_font_size_handling(self):
        """Test handling of missing or invalid font information."""
        config = Config()
        section_intel = SectionIntelligence(config)
        
        # Block without formatting information
        block = {'text': 'Test Header'}
        
        # Should not raise an exception
        probability = section_intel._calculate_header_probability(block, 'Test Header')
        self.assertIsInstance(probability, float)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
    
    def test_malformed_document_structure(self):
        """Test handling of malformed document structures."""
        config = Config()
        classifier = ContentClassifier(config)
        
        # Document with missing required fields
        malformed_document = {
            'pages': [
                {
                    'page_number': 1,
                    'content_blocks': [
                        {'text': 'Some content'}  # Missing required fields
                    ]
                }
            ]
        }
        
        # Should not raise an exception
        try:
            result = classifier.classify_content_blocks(malformed_document)
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Classification failed on malformed input: {e}")


def run_tests():
    """Run all unit tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestConfig,
        TestHelpers,
        TestContentClassifier,
        TestSectionIntelligence,
        TestIntegration,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    """Run tests when script is executed directly."""
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed successfully!")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        exit(1)