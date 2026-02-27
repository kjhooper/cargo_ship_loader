import sys
from pathlib import Path

# Make the repo root importable so tests can do `from models import ...`
sys.path.insert(0, str(Path(__file__).parent))
