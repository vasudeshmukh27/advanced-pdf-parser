"""
Table Extractor Module
======================

Encapsulates all table detection and extraction logic using pdfplumber and camelot.
Provides a clean API to detect tables on a page and extract structured data.
"""

import logging
from typing import List, Dict, Any
import pdfplumber
import camelot

from utils.config import Config


class TableExtractor:
    """
    Table extraction helper using pdfplumber and camelot.
    """
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def extract_tables(self, pdf_page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
        """
        Detect and extract tables from a pdfplumber page.

        Args:
            pdf_page: pdfplumber page object

        Returns:
            List of table data dictionaries
        """
        tables = []
        # First attempt pdfplumber extraction
        try:
            for table in pdf_page.find_tables():
                data = table.extract()
                if len(data) >= self.config.min_table_rows:
                    tables.append({
                        'table_data': data,
                        'source': 'pdfplumber',
                        'bbox': table.bbox
                    })
        except Exception as e:
            self.logger.debug(f"pdfplumber table extract error: {e}")

        # Fallback to camelot if configured
        if self.config.use_camelot:
            try:
                camelot_tables = camelot.read_pdf(
                    pdf_page.pdf.stream.name,
                    pages=str(pdf_page.page_number),
                    flavor='stream'
                )
                for ct in camelot_tables:
                    tables.append({
                        'table_data': ct.df.values.tolist(),
                        'source': 'camelot',
                        'bbox': ct._bbox
                    })
            except Exception as e:
                self.logger.debug(f"Camelot table extract error: {e}")

        return tables
