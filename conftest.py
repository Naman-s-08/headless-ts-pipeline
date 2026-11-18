import sys
from pathlib import Path

# Add the project root to Python path so 'app' can be imported during tests
sys.path.insert(0, str(Path(__file__).parent))
