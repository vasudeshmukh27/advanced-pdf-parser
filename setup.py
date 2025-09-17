"""
Project Setup and Installation Script
=====================================

This script sets up the PDF Parser project with all necessary dependencies
and directory structure. It also performs basic validation to ensure
the environment is properly configured.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Ensure Python 3.7+ is being used."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("Error: Python 3.7 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def create_directory_structure():
    """Create the required project directory structure."""
    print("Creating directory structure...")
    
    directories = [
        "parser",
        "utils", 
        "tests",
        "tests/sample_outputs",
        "logs",
        "output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   Created: {directory}/")
    
    # Create __init__.py files
    init_files = [
        "parser/__init__.py",
        "utils/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"   Created: {init_file}")


def install_system_dependencies():
    """Install system-level dependencies based on the operating system."""
    system = platform.system().lower()
    
    print(f"🔧 Installing system dependencies for {system}...")
    
    if system == "linux":
        # Ubuntu/Debian
        try:
            subprocess.run([
                "sudo", "apt-get", "update"
            ], check=True, capture_output=True)
            
            subprocess.run([
                "sudo", "apt-get", "install", "-y", 
                "ghostscript", "python3-tk", "poppler-utils"
            ], check=True, capture_output=True)
            
            print("Linux system dependencies installed")
            
        except subprocess.CalledProcessError:
            print("Could not install system dependencies automatically")
            print("Please install manually: sudo apt-get install ghostscript python3-tk poppler-utils")
            
    elif system == "darwin":
        # macOS
        try:
            subprocess.run([
                "brew", "install", "ghostscript", "poppler"
            ], check=True, capture_output=True)
            
            print("macOS system dependencies installed")
            
        except subprocess.CalledProcessError:
            print("Could not install system dependencies automatically")
            
    elif system == "windows":
        print("Windows users: Please install ghostscript manually")
    
    else:
        print(f"Unknown system: {system}")
        print("Please install ghostscript and poppler manually")


def install_python_dependencies():
    """Install Python package dependencies."""
    print("Installing Python dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("Python dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False


def validate_installation():
    """Validate that all dependencies are properly installed."""
    print("Validating installation...")
    
    required_packages = [
        "fitz",  # PyMuPDF
        "pdfplumber",
        "camelot",
        "cv2",  # opencv-python
        "pandas",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package}")
        except ImportError:
            print(f"{package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n Missing packages: {', '.join(missing_packages)}")
        print("Please install them manually or check the error messages above.")
        return False
    
    print("All dependencies validated successfully!")
    return True


def create_sample_files():
    """Create sample configuration and test files."""
    print("Creating sample files...")
    
    # Create sample JSON output for reference
    sample_output = {
        "document_metadata": {
            "source_file": "sample.pdf",
            "file_size_mb": 1.2,
            "total_pages": 3,
            "processing_timestamp": "2025-09-18T00:25:00Z",
            "parser_version": "1.0.0",
            "confidence_scores": {
                "overall_structure": 0.92,
                "section_detection": 0.89,
                "content_classification": 0.94
            }
        },
        "pages": [
            {
                "page_number": 1,
                "page_dimensions": {"width": 595, "height": 842},
                "content_blocks": [
                    {
                        "block_id": "p1_b1",
                        "type": "header",
                        "section": "Executive Summary", 
                        "sub_section": None,
                        "text": "QUARTERLY PERFORMANCE REPORT",
                        "confidence": 0.95,
                        "position": {"top": 100, "left": 50, "width": 500, "height": 30},
                        "reading_order": 1
                    },
                    {
                        "block_id": "p1_b2",
                        "type": "paragraph",
                        "section": "Executive Summary",
                        "sub_section": "Overview",
                        "text": "This report presents the quarterly performance metrics...",
                        "confidence": 0.88,
                        "position": {"top": 150, "left": 50, "width": 500, "height": 80},
                        "reading_order": 2
                    }
                ]
            }
        ]
    }
    
    with open("tests/sample_outputs/expected_structure.json", "w") as f:
        import json
        json.dump(sample_output, f, indent=2)
    
    print("   Created: tests/sample_outputs/expected_structure.json")
    
    # Create run script
    run_script = """#!/bin/bash

# PDF Parser Run Script
# =====================

echo "Starting PDF Parser..."

# Check if required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input.pdf> <output.json>"
    echo "Example: $0 documents/report.pdf output/structured_data.json"
    exit 1
fi

INPUT_PDF="$1"
OUTPUT_JSON="$2"

# Validate input file exists
if [ ! -f "$INPUT_PDF" ]; then
    echo "Error: Input PDF file not found: $INPUT_PDF"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_JSON")
mkdir -p "$OUTPUT_DIR"

# Run the parser
python main.py --pdf "$INPUT_PDF" --output "$OUTPUT_JSON" --debug

# Check if parsing was successful
if [ $? -eq 0 ]; then
    echo "Parsing completed successfully!"
    echo "Input:  $INPUT_PDF"
    echo "Output: $OUTPUT_JSON"
    
    # Show file size
    if command -v ls &> /dev/null; then
        echo "Output size: $(ls -lh "$OUTPUT_JSON" | awk '{print $5}')"
    fi
else
    echo "Parsing failed. Check the logs for details."
    exit 1
fi
"""
    
    with open("run_parser.sh", "w") as f:
        f.write(run_script)
    
    os.chmod("run_parser.sh", 0o755)  # Make executable
    print("   Created: run_parser.sh (executable)")


def run_tests():
    """Run the test suite to validate everything works."""
    print("🧪 Running test suite...")
    
    try:
        result = subprocess.run([
            sys.executable, "tests/test_parser.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("All tests passed!")
            return True
        else:
            print("Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Could not run tests: {e}")
        return False


def main():
    """Main setup function."""
    print("PDF Parser Project Setup")
    print("=" * 40)
    
    # Step 1: Check Python version
    if not check_python_version():
        return False
    
    # Step 2: Create directory structure
    create_directory_structure()
    
    # Step 3: Install system dependencies
    install_system_dependencies()
    
    # Step 4: Install Python dependencies
    if not install_python_dependencies():
        return False
    
    # Step 5: Validate installation
    if not validate_installation():
        return False
    
    # Step 6: Create sample files
    create_sample_files()
    
    # Step 7: Run tests (optional, might fail if modules aren't in place yet)
    print("\n Setup completed successfully!")
    print("\nNext steps:")
    print("1. Place your PDF files in the project directory")
    print("2. Run: python main.py --pdf your_document.pdf --output result.json")
    print("3. Or use: ./run_parser.sh your_document.pdf result.json")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)