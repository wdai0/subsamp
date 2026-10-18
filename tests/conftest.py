"""Use checkout sources by default, or explicitly verify an installed wheel."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Benchmark helpers live beside the tests.
if os.environ.get("SUBSAMP_TEST_INSTALLED") == "1":
    import subsampwinner
    package = Path(subsampwinner.__file__).resolve()
    if "site-packages" not in package.parts or ROOT / "src" in package.parents:
        raise RuntimeError(f"Expected an installed wheel, but imported {package}")
else:
    sys.path.insert(0, str(ROOT / "src"))
