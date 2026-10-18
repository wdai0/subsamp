# Packaging and release checks

Maintain one codebase containing production code, tests, essential reference fixtures, benchmark scripts/configurations, and documentation.
Publish packages built from a tested version tag.
No separate branch with tests removed is needed.
The [audit archive](AUDIT_ARCHIVE.md) preserves the verbose generated evidence removed during this cleanup.

## Distribution policy

The wheel contains only `subsampwinner/` and package metadata.
The source distribution includes `src/`, `tests/` (including fixed datasets and optional R/MPI drivers), benchmark helpers/configurations, and concise documentation.
It excludes generated `output/` files and local build/cache state.
`MANIFEST.in` declares those source-distribution contents; package discovery remains confined to `src/`.
`pyproject.toml` explicitly selects the existing setuptools backend; runtime metadata stays in `setup.py`.

`tests/check_distributions.py` checks both artifacts against the source tree, including byte equality for runtime modules and all required fixtures/test helpers.
It fails if generated outputs enter the source archive or files outside the runtime package and metadata enter the wheel.
Tests do not require an audit archive or Git history.
The optional historical benchmark/capture commands do require the original Git revisions, as before.

## Local verification

Use an isolated environment containing the chosen runtime/test dependencies and build tools:

```bash
python -m pip install -r tests/requirements.txt
python -m pip install build setuptools wheel twine
python -m build --no-isolation
python tests/check_distributions.py dist
python -m twine check --strict dist/*
python -m pip install --force-reinstall --no-deps dist/*.whl
python -m pip check
python -I tests/installed_smoke.py
```

Use an empty build destination for each verification to avoid mixing versions.
The default build creates a source archive and builds the wheel from that archive.
Verify that the source archive is self-contained and the installed wheel passes the complete serial suite:

```bash
mkdir -p /tmp/subsamp-package-source
tar -xzf dist/*.tar.gz -C /tmp/subsamp-package-source
cd /tmp/subsamp-package-source/subsampwinner-0.1.0
SUBSAMP_TEST_INSTALLED=1 python -m pytest -q
```

`SUBSAMP_TEST_INSTALLED=1` makes test startup require a `site-packages` import and prevents source-path injection, including in subprocess tests.
Without it, normal checkout testing continues to use `src/`.
R and actual MPI checks retain their explicit opt-in requirements.

## Continuous validation and publication

PR/push CI builds both distributions, checks their contents and metadata, installs the wheel, and runs the full serial suite from the extracted source distribution in four environments.
Python 3.10 uses the complete reference lock; Python 3.9 uses the same numerical versions with compatible test-tool dependencies because parts of the Python 3.10 tooling lock require Python 3.10.
Python 3.12 tests both fresh current dependencies and the minimum statsmodels backend with pandas 3.
The publication workflow checks out the published release's tag, builds and verifies the artifacts, requires the tag to match the installed package version, runs the same installed-wheel suite with the frozen reference stack, and uploads those exact artifacts only after the checks pass.
It retains the repository's existing PyPI credential configuration.
Publishing a release is the upload trigger.
Before each release, select an unused version, update its metadata, require successful PR checks and review, verify the tagged code, and publish that version's release.

Check GitHub's results for the exact revision being released; local results alone do not establish that the hosted workflow passes.
Required branch checks are configured separately through repository settings.
