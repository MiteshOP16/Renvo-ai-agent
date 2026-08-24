"""
Stand-alone script to (re)build the Qdrant tool-description index.

Run this:
  - Once after first deploying this feature (the app also does this
    automatically on startup if the index is empty -- see main.py -- so
    this is mainly useful for CI or a fresh environment where you want it
    done explicitly).
  - Any time you add, remove, or edit a tool's name/description in
    app/tools/definitions.py -- pass --force to re-embed everything,
    since an incremental diff isn't worth the complexity at this tool count.

Usage:
    cd backend
    python scripts/build_tool_index.py            # only builds if empty
    python scripts/build_tool_index.py --force     # full rebuild
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.tool_router import build_tool_index  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build/rebuild the Qdrant tool-description index.")
    parser.add_argument("--force", action="store_true", help="Re-embed and re-upsert all tools even if the index is already populated.")
    args = parser.parse_args()

    n = build_tool_index(force_rebuild=args.force)
    if n:
        print(f"Indexed {n} tool(s).")
    else:
        print("Index already populated -- nothing to do (use --force to rebuild).")


if __name__ == "__main__":
    main()