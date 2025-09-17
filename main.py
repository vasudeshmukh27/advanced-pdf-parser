import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Import our custom parsing modules
from parser.document_analyzer import DocumentAnalyzer
from parser.section_intelligence import SectionIntelligence
from parser.content_classifier import ContentClassifier
from parser.table_extractor import TableExtractor
from parser.text_processor import TextProcessor
from parser.chart_detector import ChartDetector
from utils.config import Config
from utils.helpers import setup_logging, validate_file_paths, format_processing_time


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Configure command-line argument parsing with comprehensive options.
    
    This goes beyond basic file I/O to provide professional-grade control
    over the parsing process, allowing users to fine-tune behavior based
    on their specific document characteristics.
    """
    parser = argparse.ArgumentParser(
        description="Advanced PDF Structure Parser - Extract hierarchical content with intelligent classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pdf document.pdf --output result.json
  %(prog)s --pdf report.pdf --output structured.json --debug --confidence-threshold 0.85
  %(prog)s --pdf complex_doc.pdf --output data.json --low-memory --preserve-formatting
        """
    )
    
    # Core functionality arguments
    parser.add_argument(
        '--pdf', 
        type=str, 
        required=True,
        help='Path to the input PDF file to be processed'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='Path for the output JSON file containing structured data'
    )
    
    # Advanced processing options
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.75,
        help='Minimum confidence score for content classification (0.0-1.0, default: 0.75)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable detailed logging and intermediate processing outputs'
    )
    
    parser.add_argument(
        '--low-memory',
        action='store_true',
        help='Optimize for memory efficiency when processing large documents'
    )
    
    parser.add_argument(
        '--preserve-formatting',
        action='store_true',
        help='Maintain original text formatting including whitespace and line breaks'
    )
    
    parser.add_argument(
        '--extract-images',
        action='store_true',
        help='Attempt to extract and analyze embedded images and charts'
    )
    
    parser.add_argument(
        '--use-camelot',
        action='store_true',
        help='Enable camelot library for enhanced table detection (requires ghostscript)'
    )
    
    return parser


def initialize_processing_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Initialize all processing components with user-specified configuration.
    
    This function sets up the entire processing pipeline with proper
    dependency injection and configuration management, following
    enterprise software engineering practices.
    """
    # Create configuration object from command-line arguments
    config = Config(
        confidence_threshold=args.confidence_threshold,
        debug_mode=args.debug,
        low_memory_mode=args.low_memory,
        preserve_formatting=args.preserve_formatting,
        extract_images=args.extract_images,
        use_camelot=args.use_camelot if hasattr(args, 'use_camelot') else False
    )
    
    # Initialize processing components
    document_analyzer = DocumentAnalyzer(config)
    section_intelligence = SectionIntelligence(config)
    content_classifier = ContentClassifier(config)
    
    # Initialize specialized processors
    table_extractor = TableExtractor(config)
    text_processor = TextProcessor(config)
    chart_detector = ChartDetector(config)
    
    return {
        'config': config,
        'document_analyzer': document_analyzer,
        'section_intelligence': section_intelligence,
        'content_classifier': content_classifier,
        'table_extractor': table_extractor,
        'text_processor': text_processor,
        'chart_detector': chart_detector
    }


def process_document(pdf_path: str, pipeline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the complete document processing workflow.
    
    This orchestrates the multi-stage parsing process:
    1. Document loading and initial analysis
    2. Page-by-page content extraction with specialized processors
    3. Section boundary detection
    4. Content type classification
    5. Structure validation and enhancement
    
    Args:
        pdf_path: Path to the input PDF file
        pipeline: Dictionary containing initialized processing components
        
    Returns:
        Structured document representation as dictionary
    """
    analyzer = pipeline['document_analyzer']
    section_intel = pipeline['section_intelligence']
    classifier = pipeline['content_classifier']
    table_extractor = pipeline['table_extractor']
    text_processor = pipeline['text_processor']
    chart_detector = pipeline['chart_detector']
    config = pipeline['config']
    
    logging.info(f"Starting document analysis for: {pdf_path}")
    start_time = time.time()
    
    try:
        # Stage 1: Load and analyze document structure with specialized processors
        logging.info("Stage 1: Loading document and extracting basic structure...")
        document_data = analyzer.load_and_analyze_with_processors(
            pdf_path, 
            table_extractor, 
            text_processor, 
            chart_detector
        )
        
        # Stage 2: Apply intelligent section detection
        logging.info("Stage 2: Applying advanced section detection algorithms...")
        enhanced_document = section_intel.detect_hierarchical_structure(document_data)
        
        # Stage 3: Classify content types with confidence scoring
        logging.info("Stage 3: Classifying content types and generating confidence scores...")
        classified_document = classifier.classify_content_blocks(enhanced_document)
        
        # Stage 4: Post-processing and quality enhancement
        logging.info("Stage 4: Applying post-processing and quality enhancements...")
        final_document = analyzer.apply_post_processing_with_processors(
            classified_document, 
            text_processor
        )
        
        processing_time = time.time() - start_time
        
        # Add processing metadata
        final_document['document_metadata']['processing_time_seconds'] = round(processing_time, 2)
        final_document['document_metadata']['processing_timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
        final_document['document_metadata']['parser_version'] = '1.0.0'
        final_document['document_metadata']['processors_used'] = {
            'table_extractor': True,
            'text_processor': True,
            'chart_detector': config.extract_images,
            'camelot_enabled': getattr(config, 'use_camelot', False)
        }
        
        logging.info(f"Document processing completed successfully in {format_processing_time(processing_time)}")
        return final_document
        
    except Exception as e:
        logging.error(f"Critical error during document processing: {str(e)}")
        raise


def save_structured_output(data: Dict[str, Any], output_path: str, debug_mode: bool = False) -> None:
    """
    Save the structured document data to JSON with proper formatting.
    
    This function handles JSON serialization with human-readable formatting
    and includes validation to ensure data integrity.
    """
    try:
        # Configure JSON output formatting based on debug mode
        indent = 2 if debug_mode else None
        separators = (',', ': ') if debug_mode else (',', ':')
        
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(
                data, 
                output_file, 
                indent=indent,
                separators=separators,
                ensure_ascii=False,
                sort_keys=False  # Preserve logical ordering of fields
            )
        
        # Calculate and log output statistics
        file_size = Path(output_path).stat().st_size
        total_blocks = sum(len(page.get('content_blocks', [])) for page in data.get('pages', []))
        
        logging.info(f"Successfully saved structured output to: {output_path}")
        logging.info(f"Output file size: {file_size:,} bytes")
        logging.info(f"Total content blocks processed: {total_blocks}")
        
    except Exception as e:
        logging.error(f"Failed to save output file: {str(e)}")
        raise


def main() -> int:
    """
    Main execution function with comprehensive error handling.
    
    This function coordinates the entire processing workflow while
    providing robust error handling and user feedback.
    
    Returns:
        Exit code (0 for success, non-zero for error conditions)
    """
    # Parse command-line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Initialize logging system
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)
    
    try:
        # Validate input and output file paths
        logging.info("Validating file paths and permissions...")
        validate_file_paths(args.pdf, args.output)
        
        # Initialize processing pipeline with specialized processors
        logging.info("Initializing processing pipeline with specialized processors...")
        pipeline = initialize_processing_pipeline(args)
        
        # Process the document
        logging.info("Beginning document processing with modular architecture...")
        structured_data = process_document(args.pdf, pipeline)
        
        # Save the results
        logging.info("Saving structured output...")
        save_structured_output(structured_data, args.output, args.debug)
        
        # Report final statistics
        pages_processed = len(structured_data.get('pages', []))
        overall_confidence = structured_data.get('document_metadata', {}).get('confidence_scores', {}).get('overall_structure', 0)
        processors_used = structured_data.get('document_metadata', {}).get('processors_used', {})
        
        print(f"\nProcessing completed successfully!")
        print(f"Pages processed: {pages_processed}")
        print(f"Overall confidence: {overall_confidence:.1%}")
        print(f"Processors used: {', '.join([k for k, v in processors_used.items() if v])}")
        print(f"Output saved to: {args.output}")
        
        return 0
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
        print("Error: The specified PDF file could not be found.")
        return 1
        
    except PermissionError as e:
        logging.error(f"Permission denied: {str(e)}")
        print("Error: Insufficient permissions to read input file or write output file.")
        return 1
        
    except ValueError as e:
        logging.error(f"Invalid parameter: {str(e)}")
        print(f"Error: Invalid parameter - {str(e)}")
        return 1
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"Unexpected error occurred: {str(e)}")
        print("Please check the log output for detailed error information.")
        return 1


if __name__ == "__main__":
    """
    Entry point for command-line execution.
    
    This ensures the script can be run directly from the command line
    while also being importable as a module for testing or integration.
    """
    exit_code = main()
    sys.exit(exit_code)