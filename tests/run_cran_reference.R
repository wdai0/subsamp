# Execute the separately downloaded, unmodified CRAN reference with saved subsets.
# No R package installation, source copying, or production implementation changes.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 5)
options(digits = 17)
X <- as.matrix(read.csv(args[2], header = FALSE))
y <- scan(args[3], quiet = TRUE, sep = ",")
schedule <- as.matrix(read.csv(args[4], header = FALSE)) + 1L
colnames(X) <- paste0("X", seq_len(ncol(X)))
s <- ncol(schedule)
m <- nrow(schedule)

scope <- new.env(parent = globalenv())
sys.source(args[1], envir = scope)
draw <- 0L
scope$sample <- function(x, size, replace = FALSE, ...) {
    stopifnot(identical(x, seq_len(ncol(X))), size == s, !replace)
    draw <<- draw + 1L
    stopifnot(draw <= m)
    schedule[draw, ]
}

# The first expression defines the package's nested scoring function.
# The Python caller verifies the exact source SHA256 before invoking this script.
definition <- body(scope$subsamp)[[2]]
stopifnot(as.character(definition[[1]]) == "<-",
          as.character(definition[[2]]) == "sub.weight")
eval(definition, envir = scope)
scores <- scope$sub.weight(X, y, s = s, m = m)
stopifnot(draw == m)
write.table(scores, file.path(args[5], "scores.csv"), sep = ",",
            row.names = FALSE, col.names = c("feature_one_based", "score"))

draw <- 0L
selected <- scope$subsamp(X, y, s = s, m = m, qnum = 3, wplot = FALSE)
stopifnot(draw == m)
write.csv(selected$coefficients, file.path(args[5], "final_coefficients.csv"))
writeLines(R.version.string, file.path(args[5], "r_version.txt"))

# Independently expose base R submodel statistics to check cross-language fits.
statistics <- t(vapply(seq_len(m), function(i) {
    fit <- lm(y ~ ., data.frame(y, X[, schedule[i, ]]))
    c(rss = sum(resid(fit)^2), sigma = summary(fit)$sigma,
      df = df.residual(fit), unname(summary(fit)$coefficients[-1, 3]))
}, numeric(3 + s)))
write.table(statistics, file.path(args[5], "submodels.csv"), sep = ",",
            row.names = FALSE, col.names = FALSE)
