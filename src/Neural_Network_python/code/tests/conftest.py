"""Pytest configuration: add code/ directory to sys.path so that
``from modules.xxx import yyy`` works regardless of where pytest is
invoked from.
"""

import sys
from pathlib import Path

# Insert the code/ directory (parent of this tests/ package) so that
# ``import modules.data_analytic`` etc. resolve correctly.
sys.path.insert(0, str(Path(__file__).parent.parent))
