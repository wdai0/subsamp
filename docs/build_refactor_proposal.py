"""Build the review proposal with ReportLab; does not modify package code.

Run with a Python environment containing reportlab.
The editable proposal text lives here; final output is under output/pdf/.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/subsampwinner-refactoring-proposal.pdf"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
for name, filename in [
    ("Body", "Arial.ttf"), ("Body-Bold", "Arial Bold.ttf"),
    ("Body-Italic", "Arial Italic.ttf"), ("Mono", "Courier New.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic")

NAVY = colors.HexColor("#16394D")
TEAL = colors.HexColor("#087F80")
INK = colors.HexColor("#243340")
MUTED = colors.HexColor("#5D6C78")
LINE = colors.HexColor("#D4DFE6")
PALE = colors.HexColor("#EFF5F7")
WIDTH = letter[0] - 100

styles = getSampleStyleSheet()
for name, spec in {
    "Text": dict(fontName="Body", fontSize=10.3, leading=14.1, textColor=INK, spaceAfter=7.5),
    "TitleCustom": dict(fontName="Body-Bold", fontSize=25, leading=29, textColor=NAVY, spaceAfter=14),
    "SectionCustom": dict(fontName="Body-Bold", fontSize=20, leading=24, textColor=NAVY, spaceAfter=13),
    "Subhead": dict(fontName="Body-Bold", fontSize=11.5, leading=15, textColor=NAVY, spaceBefore=8, spaceAfter=6, keepWithNext=True),
    "Eyebrow": dict(fontName="Body-Bold", fontSize=8.5, leading=11, textColor=TEAL, spaceAfter=7),
    "SmallCustom": dict(fontName="Body", fontSize=9, leading=12, textColor=MUTED, spaceAfter=6),
    "Cell": dict(fontName="Body", fontSize=9.3, leading=12.3, textColor=INK, spaceAfter=0),
    "CellHead": dict(fontName="Body-Bold", fontSize=9.1, leading=12, textColor=colors.white),
    "CodeCustom": dict(fontName="Mono", fontSize=8.8, leading=12.5, textColor=INK, spaceAfter=5),
}.items():
    styles.add(ParagraphStyle(name, alignment=TA_LEFT, **spec))

story = []


def p(text, style="Text"):
    story.append(Paragraph(text, styles[style]))


def h(text):
    p(text, "Subhead")


def table(headers, rows, widths):
    data = [[Paragraph(value, styles["CellHead"]) for value in headers]]
    data += [[Paragraph(value, styles["Cell"]) for value in row] for row in rows]
    item = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    item.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, -1), (-1, -1), .5, LINE),
    ]))
    story.extend([item, Spacer(1, 9)])


def callout(title, text):
    item = Table([[Paragraph(f"<b>{title}</b><br/>{text}", styles["Text"])]], colWidths=[WIDTH])
    item.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), .6, LINE),
        ("LINEBEFORE", (0, 0), (0, -1), 3, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([item, Spacer(1, 10)])


def section(number, title):
    if story:
        story.append(PageBreak())
    p(f"{number:02d} / PROPOSAL FOR REVIEW", "Eyebrow")
    p(title, "SectionCustom")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(.5)
    canvas.line(50, 751, 562, 751)
    canvas.setFont("Body-Bold", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(50, 761, "SUBSAMPWINNER  /  IMPLEMENTATION REFACTORING")
    canvas.line(50, 40, 562, 40)
    canvas.setFont("Body", 8)
    canvas.drawString(50, 27, "October 17, 2026  |  Review draft  |  Reference-based refactoring")
    canvas.drawRightString(562, 27, str(doc.page))
    canvas.restoreState()


# Page 1: purpose and how the supplied review is incorporated.
p("01 / OBJECTIVE AND RECOMMENDATION", "Eyebrow")
p("SubsampWinner<br/>Implementation Refactoring Proposal", "TitleCustom")
p("For review  •  October 17, 2026  •  Python baseline e0366d3", "SmallCustom")
callout("Primary objective", "Make the existing scientific implementation readable, maintainable, testable, and reliable, while preserving its core algorithm and valid numerical behavior.<br/>Create a clear foundation for future algorithm improvements and features.")
p("Use an incremental refactor of the existing classes and calculations, with small, reviewable changes protected by reference tests.\nA wholesale rewrite would introduce avoidable numerical and compatibility risk.\nAI assistance supports this engineering work; it does not change the package's scientific purpose.")
h("Current foundation")
p("The package uses NumPy, statsmodels, and SciPy; the inspected source has no Torch dependency.\nKeep that stack and statsmodels' OLS/WLS fits in this phase.\nThe main work is clearer responsibilities, explicit contracts, isolated randomness, and less duplicated state and storage.")
p("Stage 1 has already established saved data and feature-subset schedules, independent OLS/WLS checks, defect reproductions, and a comparison with archived CRAN code.\nThe recorded local run passed 15 tests, including the R comparison [4].\nThose results provide a baseline, not a claim that all current behavior is correct.")
h("Treatment of the supplied review [5]")
table(["Recommendation", "Treatment in this proposal"], [
    ("Reference behavior first", "Adopted and already substantially completed; close the remaining targeted coverage gaps before extracting the core."),
    ("Clear stages, compact statistics, optional MPI", "Adopt as implementation improvements, with equivalence checks at each stage."),
    ("Union-refit-select assurance, new selection rules", "Defer as method changes. Preserve the existing adaptive procedure and document its limitations."),
    ("Simulation study and release work", "Separate later work. This phase uses small local checks and benchmarks; it includes no publication or release."),
], [154, WIDTH - 154])
p("This document proposes the next refactoring work for evaluation.\nIt does not implement that work or change the package's defaults.", "SmallCustom")


# Page 2: the numerical and statistical contract.
section(2, "Behavior to preserve")
p("The preservation target is the current Python algorithm on its valid, documented input domain.\nThe paper and R code explain the method and expose discrepancies; they do not automatically replace existing Python choices [1-4].")
table(["Part of the workflow", "Refactoring invariant"], [
    ("Feature sampling", "Draw s distinct columns per repetition, use all observations, and perform m repetitions. Repeated subsets across repetitions remain possible."),
    ("Base regression", "Preserve OLS or user-weighted WLS with an intercept. Preserve conventional model statistics and original feature identities."),
    ("Scoring", "Average abs(t) / sqrt(RSS) over retained nonzero t statistics. Preserve zero scores when a feature has no nonzero contribution."),
    ("Retention and ranking", "Keep use_top_s=False as the existing default; True retains the best min(s,m) submodels. Keep qnum=None meaning s."),
    ("Final filtering", "Preserve Bonferroni by default, BH when requested, and an explicit unadjusted option. The adjustment family remains the q semifinalists."),
    ("Result interpretation", "final_model is the joint semifinalist model; finalists are a significance-filtered index list. Do not refit automatically on finalists."),
    ("Adaptive assurance", "Preserve its range formula, intersection/union score, largest-set update, and cumulative union on valid nonempty cases. No union-model refit is added."),
], [137, WIDTH - 137])
h("What the reference comparison established")
p("The accessible preprint's equation (3) uses 1/sqrt(RSS).\nThe Python computation agrees; its 1/RSS docstring is incorrect.\nThe R scorer uses residual standard error, sqrt(RSS/d), so its scores equal sqrt(d) times Python's top-s scores when every OLS submodel has the same d = n - s - 1 [1,2].")
p("The saved example gave a maximum difference of 4.44e-16 after this scale adjustment.\nR's final stepwise AIC selection differs from Python's significance filter, so agreement in finalists is not an acceptance criterion.\nThe final journal equation is still unverified because full-text access failed [3,4].")
h("Changes that require a separate method decision")
p("Changing the all-model default, the RSS exponent, the nonzero-statistic count, the multiplicity family, the WLS scoring criterion, or the assurance selection rule is outside the refactor.\nWLS currently scores raw, unweighted RSS; preserve that adaptation while documenting it.\nDo not add an RSS floor, silently resample failed fits, or silently drop predictors.")


# Page 3: maintainable implementation without a new framework.
section(3, "A smaller, clearer implementation")
p("Keep the existing import paths and public class names.\nMake subsamp a thin coordinator over small typed helpers; keep SubsampDoubleAssurance as composition over subsamp.\nOrganize by statistical responsibility and extract private modules only when reuse or clarity warrants it.")
table(["Responsibility", "Proposed boundary and purpose"], [
    ("Validate inputs", "Check dimensions, finite values, counts, weights, and option names before fitting. Share these checks between entry points."),
    ("Schedule subsets", "Produce ordered feature subsets from an explicit local random stream. Accept a saved schedule internally for numerical comparisons."),
    ("Analyze one subset", "Fit one specified design with statsmodels; return feature IDs, t statistics, raw RSS, and residual degrees of freedom."),
    ("Aggregate scores", "Choose retained repetitions, accumulate scores and nonzero counts, then rank features. Keep sampling and fitting out of this helper."),
    ("Fit and filter", "Fit semifinalists once, align p values with feature IDs, and apply the chosen adjustment. Reuse the same base fitting helper."),
    ("Evaluate assurance", "Calculate set union/intersection and the current update criterion separately from iteration control and printing."),
], [132, WIDTH - 132])
h("Class and result discipline")
p("Compute intermediate values locally and assign fitted attributes together after success, so an exception cannot leave a mixture of old and new results.\nRetain feature_weights, semifinalists, finalists, p_adjusted, final_model, X, and y with explicit meanings.\nPreserve the historical subsample_anlysis spelling through a compatibility wrapper if an internal corrected name is introduced.")
p("Use at most one small private record for submodel statistics if it clarifies the data flow.\nKeep compatibility adapters for methods that currently return dictionaries or dense vectors.\nAvoid a plugin framework, estimator hierarchy, or new branded public API.")
h("Optimize storage after equivalence is established")
p("Internally store each submodel's s feature IDs and s statistics, rather than a length-p vector.\nThis changes submodel storage from O(mp) to O(ms), plus O(p) score accumulators; it is a storage analysis, not a measured speedup.\nPreserve repetition order during summation and retain explicit zero statistics for the denominator rule.")
p("Keep the regression solver unchanged.\nProfile small local cases before attempting solver batching, alternate linear algebra, or additional parallel backends.\nTorch, GPU support, and new numerical dependencies are not needed for this refactor.", "SmallCustom")


# Page 4: decisions for the user's later evaluation.
section(4, "Discussion and compatibility choices")
p("These are proposed policies for review, not decisions already implemented.\nThey repair undefined or misleading behavior while leaving the valid numerical workflow intact.")
table(["Issue", "Recommended treatment and trade-off"], [
    ("Invalid inputs and qnum=0", "Reject invalid counts, shapes, nonfinite values, and unknown method names with actionable errors. Treat qnum=0 as invalid; empty finalists remain a valid outcome after filtering."),
    ("Constants and singular fits", "Reject supplied constant predictors and nonidentifiable fitted designs; construct the intercept explicitly. Check rank on each fitted subset and the semifinalist fit, not on the entire high-dimensional X. Do not silently remove or resample columns."),
    ("Saturated or perfect fits", "Require positive residual degrees of freedom and finite model statistics. Fail explicitly for zero RSS. Any near-zero tolerance must be specified and tested rather than becoming a hidden score adjustment."),
    ("Random-state compatibility", "First isolate the legacy sampler in a local RandomState seeded per fit, preserving historical serial sequences where verified. A later Generator migration is a separate reproducibility choice. seed=None should use local entropy, not the caller's global NumPy stream."),
    ("fit() return value", "Preserve None for compatibility in this phase and correct the documentation. Returning self is reasonable later, but is an API decision rather than a prerequisite for maintainable internals."),
    ("Assurance boundary cases", "Return a documented empty result when all selections are empty, without calling it perfect stability. Validate candidate sizes and reject infeasible triplets; clipping or resampling would need a separate rule. Report the actual last evaluated range."),
    ("Simulation covariance", "Keep simulation utilities separate from algorithm verification. Add validity checks now; choose a PSD-generating recipe in a separately reviewed simulator repair because it changes the data-generating process."),
], [127, WIDTH - 127])
h("Parallel result contract")
p("For a future MPI repair in this phase, recommend one schedule created on root, jobs identified by repetition, canonical aggregation order, and the complete fitted result on every rank.\nThis is easiest to reason about, but broadcasts add memory and communication cost.\nRoot-only results are an alternative that requires a clearly documented worker return contract.")
p("A rank-local failure must be reported collectively before a coordinated failure; otherwise other ranks can hang.\nActual multi-process tests are required before describing MPI support as repaired.", "SmallCustom")


# Page 5: implementation and acceptance, not statistical expansion.
section(5, "Execution sequence and acceptance")
table(["Change set", "Deliverable and acceptance condition"], [
    ("1. Complete the guardrails", "Add the supplied hand-calculated retention fixture, an analytical orthogonal OLS example with nonzero intercept, and focused repair tests. Keep defect observations separate from preservation fixtures."),
    ("2. Extract the serial core", "Introduce the helper boundaries and simplify class state, keeping the solver and defaults unchanged. Replay every saved OLS/WLS schedule; compare fits, scores, rankings, adjusted p values, and finalists."),
    ("3. Repair contracts", "Implement the reviewed validation, intercept, RNG, and empty-result policies. Each changed behavior gets a focused test, rationale, and compatibility note. Preserve valid historical outputs."),
    ("4. Reduce duplication", "Introduce compact internal statistics and shared fitting logic. Recheck arithmetic and ordering, then measure runtime and peak memory on bounded local workloads."),
    ("5. Isolate parallel execution", "Make MPI lazy and optional. Add schedule-based worker execution and collective error handling only after the serial core passes. Validate multiple ranks, uneven allocation, and idle ranks."),
    ("6. Review the result", "Deliver the local diff, updated behavior documentation, validation commands/environment, and measured limitations. Limit metadata edits to optional MPI and an honest supported Python range."),
], [125, WIDTH - 125])
h("Required evidence")
p("Use four layers: historical characterization, independent arithmetic, fixed-schedule R/Python comparison, and contract/failure tests.\nCompare R/Python only where their procedures match; correct the known score scale.\nKeep current reference tolerances unless conditioning or changed evaluation order justifies a documented adjustment.")
p("Check nonzero-count semantics, feature-ID alignment, valid empty finalists, global RNG isolation, and predictor-rescaling/response-shift invariance with fixed schedules.\nA serial import and fit must work with MPI absent.\nMPI equivalence should compare all ranks against the same serial schedule, not unrelated seeds.")
h("Bounded optimization and delivery")
p("Use representative local cases such as (n,p,s,m) = (80,100,10,100) and (200,1000,20,200); record hardware, library versions, runtime, and peak memory.\nThese are proposed benchmarks, not completed measurements.\nNo speedup target is asserted before profiling.")
p("The existing 15-test run is a recorded stage-1 result, not evidence that these proposed repairs already pass.\nAdd local test execution now; a repository test-CI change can be reviewed later without modifying the publishing workflow or remote settings.\nNo HPC runs or broad simulation study are needed to accept the refactor.", "SmallCustom")


# Page 6: evaluation criteria and compact provenance.
section(6, "Evaluation and evidence")
h("Questions for review")
table(["Decision", "Recommended answer"], [
    ("What is being preserved?", "The valid current Python workflow, including its documented adaptations and defaults. The reference sources explain differences, rather than silently overriding them."),
    ("What changes now?", "Internal structure, data flow, documentation, numerical contracts, randomness ownership, and avoidable storage. Fixes to invalid cases are explicit."),
    ("What makes it ready?", "A readable orchestration path, independently checked helpers, equivalent valid outputs, clear failure behavior, and reproducible local evidence."),
    ("What remains separate?", "New scoring or inference rules, union-refit-select assurance, new public APIs, Torch/GPU work, new R implementation, rebranding, release, and publication."),
], [135, WIDTH - 135])
callout("Recommended next authorization", "Proceed with the serial refactor and explicit contract repairs after reviewing the policies on page 4.\nTreat compact storage and MPI as subsequent, separately reviewable changes within the same implementation objective.")
h("Sources and provenance")
p('[1] Fan, Y. and Sun, J. (2020). <i>Subsampling Winner Algorithm for Feature Selection in Large Regression Data.</i> arXiv:2002.02903v1. Algorithm 1, equation (3), Algorithm 3. <link href="https://arxiv.org/abs/2002.02903v1" color="#087F80">Primary preprint</link>.', "SmallCustom")
p('[2] CRAN <i>subsamp</i> 0.1.0 (2017), regression implementation. Pinned mirror e56d150; the archived source was inspected and executed locally. <link href="https://github.com/cran/subsamp/blob/e56d150a9cd566067dec47197546f70548d906ac/R/subsamp.R" color="#087F80">R source</link> | <link href="https://cran.r-project.org/src/contrib/Archive/subsamp/subsamp_0.1.0.tar.gz" color="#087F80">CRAN archive</link>.', "SmallCustom")
p('[3] Fan, Y. and Sun, J. (2021). <i>Subsampling from features in large regression to find winning features.</i> Statistical Analysis and Data Mining 14(2), 168-184. <link href="https://doi.org/10.1002/sam.11499" color="#087F80">doi:10.1002/sam.11499</link>. Bibliographic record verified; final full-text equation not verified.', "SmallCustom")
p('[4] Local evidence: <font name="Mono">docs/REFERENCE_BEHAVIOR.md</font>, <font name="Mono">docs/REFERENCE_HANDOFF.md</font>, and <font name="Mono">tests/reference/</font>. Python baseline <link href="https://github.com/wdai0/subsamp/tree/e0366d3bcf917b875b982214a2ef342f8d463e42" color="#087F80">e0366d3</link>. Recorded environment: Python 3.10.7, NumPy 1.26.4, SciPy 1.13.0, statsmodels 0.14.2, mpi4py 4.1.1, pytest 8.4.2; R 4.3.2. MPI initialization disabled for serial tests.', "SmallCustom")
p('[5] User-supplied review memo, beginning "I recommend making the first milestone establish and verify reference behavior," supplied October 17, 2026. Its prior source-only assessment is incorporated and updated with the completed local reference work. The published-PDF check identified in that memo remains outstanding.', "SmallCustom")
h("Scientific limits")
p("Implementation equivalence does not establish general FDR control, post-selection validity, heteroskedastic robustness, or large-scale performance.\nThe full journal audit and actual MPI execution remain open.\nFuture algorithm extensions should use this refactored core as their comparison baseline and receive their own statistical evaluation.", "SmallCustom")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(OUTPUT), pagesize=letter, rightMargin=50, leftMargin=50,
    topMargin=57, bottomMargin=53, title="SubsampWinner: Implementation Refactoring Proposal",
    author="SubsampWinner project", subject="Behavior-preserving refactoring proposal for review",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUTPUT)
