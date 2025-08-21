#!/bin/bash
# setup.sh - Quick setup script for Python documentation skeleton

set -e

echo "🚀 Setting up Python Documentation Skeleton"
echo "=========================================="
echo ""

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p .github/workflows
mkdir -p src
mkdir -p scripts
mkdir -p docs/api/uml

# Create .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv/
*.egg-info/
dist/
build/

# IDEs
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Documentation build files
docs/_build/
*.log
EOF

# Create source package init
echo "📦 Creating Python package structure..."
cat > src/__init__.py << 'EOF'
"""
Example Package
===============

This is an example package for demonstrating documentation generation.
"""

__version__ = "0.1.0"
EOF

# Create example module
cat > src/example.py << 'EOF'
"""
Example Module
==============

This module demonstrates various Python constructs for documentation.
"""

from typing import List, Optional, Dict, Any


def hello(name: str = "World") -> str:
    """
    Generate a greeting message.
    
    Args:
        name: The name to greet (default: "World")
        
    Returns:
        A formatted greeting string
        
    Example:
        >>> hello("Alice")
        'Hello, Alice!'
    """
    return f"Hello, {name}!"


def calculate_sum(numbers: List[float]) -> float:
    """
    Calculate the sum of a list of numbers.
    
    Args:
        numbers: List of numbers to sum
        
    Returns:
        The sum of all numbers
        
    Raises:
        ValueError: If the list is empty
    """
    if not numbers:
        raise ValueError("Cannot sum an empty list")
    return sum(numbers)


class Calculator:
    """
    A simple calculator class for basic operations.
    
    This class provides methods for basic arithmetic operations
    and maintains a history of calculations.
    
    Attributes:
        history: List of calculation results
        precision: Number of decimal places for results
    """
    
    def __init__(self, precision: int = 2):
        """
        Initialize the calculator.
        
        Args:
            precision: Number of decimal places for results (default: 2)
        """
        self.precision = precision
        self.history: List[float] = []
    
    def add(self, a: float, b: float) -> float:
        """
        Add two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            The sum of a and b
        """
        result = round(a + b, self.precision)
        self.history.append(result)
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """
        Subtract b from a.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            The difference (a - b)
        """
        result = round(a - b, self.precision)
        self.history.append(result)
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """
        Multiply two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            The product of a and b
        """
        result = round(a * b, self.precision)
        self.history.append(result)
        return result
    
    def divide(self, a: float, b: float) -> float:
        """
        Divide a by b.
        
        Args:
            a: Dividend
            b: Divisor
            
        Returns:
            The quotient (a / b)
            
        Raises:
            ZeroDivisionError: If b is zero
        """
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        result = round(a / b, self.precision)
        self.history.append(result)
        return result
    
    def get_history(self) -> List[float]:
        """
        Get the calculation history.
        
        Returns:
            List of all calculation results
        """
        return self.history.copy()
    
    def clear_history(self) -> None:
        """Clear the calculation history."""
        self.history.clear()
    
    @property
    def last_result(self) -> Optional[float]:
        """Get the last calculation result."""
        return self.history[-1] if self.history else None
    
    @staticmethod
    def is_even(n: int) -> bool:
        """
        Check if a number is even.
        
        Args:
            n: The number to check
            
        Returns:
            True if n is even, False otherwise
        """
        return n % 2 == 0


class DataContainer:
    """
    A container for storing key-value data.
    
    This class provides a simple interface for storing
    and retrieving data with optional metadata.
    """
    
    def __init__(self):
        """Initialize an empty container."""
        self._data: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def set(self, key: str, value: Any, **metadata) -> None:
        """
        Store a value with optional metadata.
        
        Args:
            key: The key to store the value under
            value: The value to store
            **metadata: Optional metadata as keyword arguments
        """
        self._data[key] = value
        if metadata:
            self._metadata[key] = metadata
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value by key.
        
        Args:
            key: The key to look up
            default: Default value if key not found
            
        Returns:
            The stored value or default
        """
        return self._data.get(key, default)
    
    def get_metadata(self, key: str) -> Dict[str, Any]:
        """
        Get metadata for a key.
        
        Args:
            key: The key to get metadata for
            
        Returns:
            Dictionary of metadata or empty dict if none
        """
        return self._metadata.get(key, {})
    
    def keys(self) -> List[str]:
        """Get all stored keys."""
        return list(self._data.keys())
    
    def clear(self) -> None:
        """Clear all data and metadata."""
        self._data.clear()
        self._metadata.clear()
    
    def __len__(self) -> int:
        """Get the number of stored items."""
        return len(self._data)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self._data
EOF

# Create a simple requirements.txt
echo "📋 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Documentation generation
pylint>=2.15.0  # For pyreverse UML generation

# Add your project dependencies below
# requests>=2.28.0
# numpy>=1.24.0
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Next steps:"
echo "1. Copy simple_doc_generator.py to scripts/ directory"
echo "2. Copy the GitHub Actions workflow to .github/workflows/generate-docs.yml"
echo "3. Test locally: python scripts/simple_doc_generator.py src docs/api"
echo "4. Commit and push to GitHub"
echo ""
echo "📁 Created structure:"
tree -L 2 2>/dev/null || {
    echo "your-project/"
    echo "├── .github/"
    echo "│   └── workflows/"
    echo "├── src/"
    echo "│   ├── __init__.py"
    echo "│   └── example.py"
    echo "├── scripts/"
    echo "├── docs/"
    echo "│   └── api/"
    echo "├── .gitignore"
    echo "└── requirements.txt"
}
echo ""
echo "🎉 Ready to generate documentation!"
