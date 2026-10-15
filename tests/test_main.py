import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_help_lists_subcommands_without_crashing():
    r = subprocess.run([sys.executable, "src/main.py", "--help"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "sample" in r.stdout
    assert "download" in r.stdout
