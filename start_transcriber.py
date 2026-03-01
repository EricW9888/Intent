from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if os.name == "nt":
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
else:
    venv_python = ROOT / ".venv" / "bin" / "python"

python_executable = str(venv_python if venv_python.exists() else Path(sys.executable))
raise SystemExit(subprocess.call([python_executable, str(ROOT / "main.py"), *sys.argv[1:]], cwd=ROOT))
