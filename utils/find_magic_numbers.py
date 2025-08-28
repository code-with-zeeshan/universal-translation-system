# utils/find_magic_numbers.py
"""
Find magic numbers in the codebase.
This script analyzes the codebase for magic numbers that should be replaced with constants.
"""

import os
import re
import ast
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

logger = logging.getLogger(__name__)


class MagicNumberFinder:
    """
    Finder for magic numbers in the codebase.
    """
    
    def __init__(self, root_dir: str = '.'):
        """
        Initialize the finder.
        
        Args:
            root_dir: Root directory to search
        """
        self.root_dir = Path(root_dir)
        self.magic_numbers = {}
        self.excluded_numbers = {0, 1, -1, 2, 10, 100}  # Common non-magic numbers
        
    def find(self) -> Dict[str, List[Tuple[int, str, int]]]:
        """
        Find magic numbers in the codebase.
        
        Returns:
            Dictionary mapping file paths to lists of (line number, context, number)
        """
        # Find Python files
        python_files = list(self.root_dir.glob('**/*.py'))
        
        # Analyze files
        for file_path in python_files:
            self._analyze_file(file_path)
            
        return self.magic_numbers
        
    def _analyze_file(self, file_path: Path) -> None:
        """
        Analyze a Python file for magic numbers.
        
        Args:
            file_path: Path to Python file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            # Find magic numbers
            for node in ast.walk(tree):
                if isinstance(node, ast.Num) and node.n not in self.excluded_numbers:
                    # Get line number
                    line_no = getattr(node, 'lineno', 0)
                    
                    # Get context
                    context = self._get_context(content, line_no)
                    
                    # Add to results
                    if str(file_path) not in self.magic_numbers:
                        self.magic_numbers[str(file_path)] = []
                        
                    self.magic_numbers[str(file_path)].append((line_no, context, node.n))
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            
    def _get_context(self, content: str, line_no: int) -> str:
        """
        Get context for a line number.
        
        Args:
            content: File content
            line_no: Line number
            
        Returns:
            Context string
        """
        lines = content.split('\n')
        if 0 <= line_no - 1 < len(lines):
            return lines[line_no - 1].strip()
        return ""


def main():
    """Main function."""
    import json
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create finder
    finder = MagicNumberFinder()
    
    # Find magic numbers
    magic_numbers = finder.find()
    
    # Print results
    for file_path, numbers in magic_numbers.items():
        print(f"\n{file_path}:")
        for line_no, context, number in numbers:
            print(f"  Line {line_no}: {context} (Number: {number})")
            
    # Save results
    with open('magic_numbers_report.json', 'w') as f:
        json.dump(magic_numbers, f, indent=2)
        
    print(f"\nMagic numbers report saved to magic_numbers_report.json")
    
    # Suggest constants
    print("\nSuggested constants:")
    all_numbers = {}
    for file_path, numbers in magic_numbers.items():
        for line_no, context, number in numbers:
            if number not in all_numbers:
                all_numbers[number] = []
            all_numbers[number].append((file_path, line_no, context))
            
    for number, occurrences in all_numbers.items():
        if len(occurrences) > 1:
            print(f"\n{number} ({len(occurrences)} occurrences):")
            for file_path, line_no, context in occurrences:
                print(f"  {file_path}:{line_no}: {context}")
            
            # Suggest constant name
            if number > 0:
                if number % 1024 == 0:
                    unit = "KB" if number < 1024 * 1024 else "MB" if number < 1024 * 1024 * 1024 else "GB"
                    value = number / 1024 if unit == "KB" else number / (1024 * 1024) if unit == "MB" else number / (1024 * 1024 * 1024)
                    print(f"  Suggested constant: MAX_{unit}_{int(value)}")
                elif number % 60 == 0:
                    minutes = number // 60
                    if minutes % 60 == 0:
                        hours = minutes // 60
                        if hours % 24 == 0:
                            days = hours // 24
                            print(f"  Suggested constant: DAYS_{days}")
                        else:
                            print(f"  Suggested constant: HOURS_{hours}")
                    else:
                        print(f"  Suggested constant: MINUTES_{minutes}")
                elif number % 1000 == 0:
                    print(f"  Suggested constant: MAX_{number // 1000}K")
                else:
                    print(f"  Suggested constant: MAX_{number}")


if __name__ == '__main__':
    main()
