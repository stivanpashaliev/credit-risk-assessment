"""
Credit Risk Assessment — interactive CLI application.
Run this file in PyCharm to assess a new customer's credit risk.
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

from src.utils import load_model, get_feature_names


def main():
    print("=" * 55)
    print("   Credit Risk Assessment System")
    print("=" * 55)
    print()

    # TODO: This will be expanded in Commit 11
    # For now, verify the project structure is working
    features = get_feature_names()
    print(f"Model expects {len(features)} features:")
    for f in features:
        print(f"  - {f}")
    print()
    print("Project structure verified. Ready for development.")
    print("=" * 55)


if __name__ == "__main__":
    main()