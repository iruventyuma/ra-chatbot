"""
Run this once before starting the app:
    python scripts/build_indices.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import GUIDELINES_DIR
from src.image_index import build_image_index
from src.text_index import build_text_index

if __name__ == "__main__":
    guideline_files = list(GUIDELINES_DIR.glob("*.txt"))
    if not guideline_files:
        print(
            f"WARNING: no .txt files found in {GUIDELINES_DIR} -- skipping the "
            "text/guideline index. The chatbot will still run, but hybrid_search() "
            "will return nothing until you add guideline text and re-run this "
            "script. See data/guidelines/README.md for what to add."
        )
    else:
        print(f"Building text (guideline) index from {len(guideline_files)} file(s)...")
        build_text_index()

    print("Building image (RAM-H1200) index...")
    build_image_index()

    print("Done. Run: streamlit run app.py")
