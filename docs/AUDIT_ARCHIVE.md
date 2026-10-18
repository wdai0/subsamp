# Preserved audit checkpoint

The complete pre-cleanup snapshot is preserved at commit `18f982bd1d07e35d9a1dc46616593c88905c2d42`, with local tag `audit/refactor-18f982b`.
The versioned archive is `subsampwinner-audit-18f982b.zip`, saved in the local Downloads directory.
It contains all 155 committed files, including source code, test fixtures, documentation, and the generated evidence.
Every file in the ZIP was checked byte-for-byte against the commit before generated files were removed from the active tree.

Archive SHA-256:

```text
93326f41d041b5c4325aeb04dea73bc4a9626a20563c9d7cc492a8ad6e644e21
```

The [archive manifest](audit-archive.json) records the full commit, archive name, size, checksum, and recovery command.
The ZIP and tag are local; no release asset or remote tag has been published.
Git history still contains the preserved evidence, so this cleanup reduces the current tree and review diff rather than rewriting repository history.

## Recover the snapshot

Use the ZIP in Downloads, or recreate the snapshot from the retained commit:

```bash
git archive --format=zip --prefix=subsampwinner-audit-18f982b/ \
  --output=/tmp/subsampwinner-audit-18f982b.zip \
  18f982bd1d07e35d9a1dc46616593c88905c2d42
unzip /tmp/subsampwinner-audit-18f982b.zip -d /tmp
```

The extracted snapshot is `/tmp/subsampwinner-audit-18f982b/`.
Its original documentation links resolve inside that complete snapshot.

## Original audit

| Path inside the extracted snapshot | Evidence |
| --- | --- |
| `output/refactor-audit/final-classic/` | Original 25-case comparison and timings |
| `output/refactor-audit/large-smoke/` | Original one-dataset smoke run |
| `output/pdf/subsampwinner-refactoring-proposal.pdf` | Previously delivered proposal |

## Revision audit

| Path inside the extracted snapshot | Evidence |
| --- | --- |
| `output/revision-audit/production-performance.json` | Five timing repeats, allocation measurements, and source hashes |
| `output/revision-audit/environment-*.json` | Three dependency-environment checks and freezes |
| `output/revision-audit/combined-tests.json` | 123-pass result including R/MPI |
| `output/revision-audit/corrected-historical-summary.json` | Historical metrics separated by method configuration |
| `output/revision-audit/fixed-compatibility/` | All 25 strict compatibility results |
| `output/revision-audit/scaled-score-smoke/` | Intentional scoring-difference smoke check |
| `output/revision-audit/paper-family-smoke/` | Explicit p- versus q-family smoke check and dataset |
| `output/revision-audit/verified-files.json` | Source and test file hashes at the revision audit |

## Ongoing storage policy

Keep production code, correctness tests, essential frozen fixtures, benchmark scripts/configurations, concise reports, and CI in the maintained branch.
The 25 classic datasets, ordered subsets, expected outputs, and their checksum manifest remain under `tests/reference/`.
Normal tests require no external audit archive.
The reporting regression reconstructs its three small input records from the existing frozen manifest.

New generated results go under the ignored `output/` directory.
For evidence worth retaining, save a versioned archive outside the source tree and record its commit, configuration, environment, and checksum in a concise manifest/report.
Any later upload as a release asset or other durable remote archive is a separate publication step.
