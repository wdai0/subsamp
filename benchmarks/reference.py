"""Load the unmodified pinned Python source solely for paired audit runs."""

import hashlib
from pathlib import Path
import subprocess
import types

BASE = "e0366d3bcf917b875b982214a2ef342f8d463e42"
SOURCE_SHA256 = "545101723a0b7505ec6d3a02425e456200c70ec46c5c02e2f625bf6fcee7f412"


def load_legacy():
    # Only this historical audit adapter requires mpi4py for the old import.
    import mpi4py
    mpi4py.rc.initialize = False
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(["git", "show", f"{BASE}:src/subsampwinner/subsamp.py"], cwd=root)
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("pinned legacy source checksum mismatch")
    legacy = types.ModuleType("pinned_legacy_subsamp")
    exec(compile(source, f"{BASE}/subsamp.py", "exec"), legacy.__dict__)
    return legacy.subsamp
