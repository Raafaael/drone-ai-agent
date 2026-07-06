"""Wrapper para executar o agente a partir da raiz do repositorio."""

import os
import sys


ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from main import main  # noqa: E402


if __name__ == "__main__":
    main()
