# =============================================================================
# Power analysis -- MMLU bet-stakes experiment

# Primary model: flipped_to_user ~ stakes_num + (1|question_id) family = binomial
# H1 (one-sided): B_stakes < 0 (log-odds of flipping to user's incorrect answer decreases as stakes increase)

# Input: fitted model from mmlu_model_fitting.R
# Usage: Rscript mmlu_power_analysis.R <fitted_model_file>
#

suppressPackageStartupMessages({
  library(lme4)
})

SD_Q_GRID  <- c(0.5, 1.0, 1.5)

suppressPackageStartupMessages({
  library(lme4)
})

# ---- swept design grid ------------------------------------------------------
# SD_q: plausible question-level random-intercept SD on the log-odds scale.
#       The pilot cannot pin this down; 0.5/1.0/1.5 brackets typical values.
SD_Q_GRID  <- c(0.5, 1.0, 1.5)
# baseline flip-to-user rate ~0.14 on the pilot -> intercept = logit(0.14).
BASE_RATE  <- 0.14
INTERCEPT  <- log(BASE_RATE / (1 - BASE_RATE))
# SESOI grid, log-odds, negative (H1: B_stakes < 0). Pre-register ONE.
SESOI_GRID <- c(-0.10, -0.20, -0.35)
N_GRID     <- c(500, 1000, 2000)
N_SIMS     <- 200            # bump on a faster machine; glmer is slow
ALPHA      <- 0.05           # one-sided

cat("=================================================================\n")
cat("MMLU BET-STAKES -- DESIGN / POWER ANALYSIS  (SD_q sweep)\n")
cat("=================================================================\n")
cat(sprintf("Baseline flip-to-user rate: %.3f  (intercept = %.4f log-odds)\n",
            BASE_RATE, INTERCEPT))
cat(sprintf("Sweeping SD_q over {%s}; pilot glmer is degenerate so SD_q\n",
            paste(SD_Q_GRID, collapse = ", ")))
cat("cannot be read from it -- the conclusion is reported across the range.\n")
cat(sprintf("Test: one-sided H1 B_stakes<0, alpha=%.2f, n_sims=%d\n\n",
            ALPHA, N_SIMS))

cat("SESOI interpretation (flip-rate change, stakes 0 -> 2, at baseline):\n")
for (b in SESOI_GRID) {
  p0 <- plogis(INTERCEPT); p2 <- plogis(INTERCEPT + 2 * b)
  cat(sprintf("  SESOI %+.2f log-odds : %.3f -> %.3f  (delta %+.3f)\n",
              b, p0, p2, p2 - p0))
}
cat("\n")

# -----------------------------------------------------------------------------
simulate_once <- function(n_q, true_beta, sd_q, seed) {
  set.seed(seed)
  d <- expand.grid(question_id = 1:n_q, stakes_num = 0:2)
  re <- rnorm(n_q, 0, sd_q)
  eta <- INTERCEPT + true_beta * d$stakes_num + re[d$question_id]
  d$y <- rbinom(nrow(d), 1, plogis(eta))
  d$question_id <- factor(d$question_id)
  fit <- tryCatch(
    suppressMessages(suppressWarnings(
      glmer(y ~ stakes_num + (1 | question_id), data = d, family = binomial)
    )),
    error = function(e) NULL
  )
  if (is.null(fit)) return(c(beta = NA, se = NA, p = NA))
  cc <- summary(fit)$coefficients
  c(beta = cc["stakes_num", "Estimate"],
    se   = cc["stakes_num", "Std. Error"],
    p    = cc["stakes_num", "Pr(>|z|)"])
}

# returns: power, and median CI half-width (1.96*SE) across sims
design_at <- function(n_q, true_beta, sd_q, n_sims = N_SIMS) {
  hits <- 0L; valid <- 0L; halfw <- numeric(0)
  for (s in seq_len(n_sims)) {
    r <- simulate_once(n_q, true_beta, sd_q,
                       seed = 60000 + s + 13 * n_q + round(1000 * sd_q))
    if (!is.na(r["p"]) && !is.na(r["se"]) && is.finite(r["se"])) {
      valid <- valid + 1L
      if (r["p"] / 2 < ALPHA && r["beta"] < 0) hits <- hits + 1L
      halfw <- c(halfw, 1.96 * r["se"])
    }
  }
  list(power = if (valid) hits / valid else NA,
       half_width = if (length(halfw)) median(halfw) else NA,
       valid = valid)
}

# ---- POWER table: one block per SD_q ---------------------------------------
cat("================  POWER  (reject one-sided H1: B_stakes<0)  ===========\n")
for (sd_q in SD_Q_GRID) {
  cat(sprintf("\n-- SD_q = %.1f --\n", sd_q))
  cat(sprintf("%-10s", "SESOI"))
  for (n in N_GRID) cat(sprintf("N=%-9d", n))
  cat("\n")
  cat(strrep("-", 10 + 11 * length(N_GRID)), "\n", sep = "")
  for (b in SESOI_GRID) {
    cat(sprintf("%-10s", sprintf("%+.2f", b)))
    for (n in N_GRID) {
      r <- design_at(n, b, sd_q)
      cat(sprintf("%.3f      ", r$power))
    }
    cat("\n")
  }
}

# ---- CI HALF-WIDTH table: equivalence-bound view ---------------------------
# Computed at true effect = 0 (the pilot's situation). If half-width < |SESOI|,
# an n=2000 estimate near 0 BOUNDS the effect below the SESOI.
cat("\n\n=========  CI HALF-WIDTH at true B_stakes = 0 (log-odds)  ==========\n")
cat("If half-width < |SESOI| you can bound the effect below that SESOI.\n")
for (sd_q in SD_Q_GRID) {
  cat(sprintf("\n-- SD_q = %.1f --\n", sd_q))
  cat(sprintf("%-12s", "N"))
  cat("half-width   implied flip-rate band (at baseline)\n")
  for (n in N_GRID) {
    r <- design_at(n, 0.0, sd_q)
    hw <- r$half_width
    # translate the log-odds band to a flip-rate band around the baseline
    lo <- plogis(INTERCEPT - hw); hi <- plogis(INTERCEPT + hw)
    cat(sprintf("%-12d%.4f       [%.3f, %.3f]\n", n, hw, lo, hi))
  }
}

cat("\n-----------------------------------------------------------------\n")
cat("READING THIS:\n")
cat(" * Power table: if your pre-registered SESOI row reaches >=0.80 at a\n")
cat("   given N across ALL swept SD_q, that N is adequate to DETECT an\n")
cat("   effect of that size.\n")
cat(" * Half-width table: if the half-width at a given N is below your\n")
cat("   |SESOI| across all SD_q, then an n=2000 result of B_stakes~0 lets\n")
cat("   you BOUND the stakes effect below the SESOI -- a defensible\n")
cat("   negative result (NOT 'the null is true').\n")
cat(" * Pre-register ONE SESOI before the confirmatory run.\n")
cat("=================================================================\n")