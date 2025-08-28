# utils/thread_safety_audit.py
"""
Thread safety audit for the Universal Translation System.
This script analyzes the codebase for thread safety issues.
"""

import os
import re
import ast
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from utils.thread_safety import generate_thread_safety_report

logger = logging.getLogger(__name__)


class ThreadSafetyAuditor:
    """
    Auditor for thread safety issues.
    """
    
    def __init__(self, root_dir: str = '.'):
        """
        Initialize the auditor.
        
        Args:
            root_dir: Root directory to audit
        """
        self.root_dir = Path(root_dir)
        self.shared_resources = set()
        self.thread_safe_classes = set()
        self.unsafe_classes = set()
        self.lock_usage = {}
        
    def audit(self) -> Dict[str, Any]:
        """
        Perform thread safety audit.
        
        Returns:
            Audit report
        """
        # Get documented thread safety
        documented = generate_thread_safety_report()
        
        # Find Python files
        python_files = list(self.root_dir.glob('**/*.py'))
        
        # Analyze files
        for file_path in python_files:
            self._analyze_file(file_path)
            
        # Generate report
        return {
            "documented_thread_safety": documented,
            "shared_resources": list(self.shared_resources),
            "thread_safe_classes": list(self.thread_safe_classes),
            "unsafe_classes": list(self.unsafe_classes),
            "lock_usage": self.lock_usage
        }
        
    def _analyze_file(self, file_path: Path) -> None:
        """
        Analyze a Python file for thread safety issues.
        
        Args:
            file_path: Path to Python file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            # Find classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._analyze_class(node, file_path)
                    
            # Find shared resources
            self._find_shared_resources(content, file_path)
            
            # Find lock usage
            self._find_lock_usage(content, file_path)
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            
    def _analyze_class(self, node: ast.ClassDef, file_path: Path) -> None:
        """
        Analyze a class for thread safety issues.
        
        Args:
            node: AST node for class
            file_path: Path to Python file
        """
        class_name = node.name
        has_lock = False
        has_thread_safe_decorator = False
        
        # Check for lock attribute
        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id.endswith('_lock'):
                        has_lock = True
                        
            # Check for thread_safe decorator
            if isinstance(child, ast.FunctionDef):
                for decorator in child.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'thread_safe':
                        has_thread_safe_decorator = True
                        
        # Check class docstring for thread safety
        docstring = ast.get_docstring(node)
        has_thread_safety_doc = docstring and 'thread safe' in docstring.lower()
        
        # Determine thread safety
        if has_lock or has_thread_safe_decorator or has_thread_safety_doc:
            self.thread_safe_classes.add(class_name)
        else:
            self.unsafe_classes.add(class_name)
            
    def _find_shared_resources(self, content: str, file_path: Path) -> None:
        """
        Find shared resources in a file.
        
        Args:
            content: File content
            file_path: Path to Python file
        """
        # Look for class attributes
        class_attr_pattern = r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        matches = re.findall(class_attr_pattern, content)
        
        # Filter out private attributes and known safe types
        for match in matches:
            if not match.startswith('_') and match not in ('logger', 'config'):
                self.shared_resources.add(match)
                
    def _find_lock_usage(self, content: str, file_path: Path) -> None:
        """
        Find lock usage in a file.
        
        Args:
            content: File content
            file_path: Path to Python file
        """
        # Look for lock usage
        lock_pattern = r'with\s+([a-zA-Z_][a-zA-Z0-9_\.]*lock[a-zA-Z0-9_\.]*)\s*:'
        matches = re.findall(lock_pattern, content)
        
        if matches:
            self.lock_usage[str(file_path)] = matches


def main():
    """Main function."""
    import json
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create auditor
    auditor = ThreadSafetyAuditor()
    
    # Perform audit
    report = auditor.audit()
    
    # Print report
    print(json.dumps(report, indent=2))
    
    # Save report
    with open('thread_safety_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Thread safety report saved to thread_safety_report.json")


if __name__ == '__main__':
    main()
