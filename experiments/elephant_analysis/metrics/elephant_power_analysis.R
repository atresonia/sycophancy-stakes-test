# =============================================================================
# Power analysis -- ELEPHANT moral-sycophancy stakes experiment
# METHOD: simulation-based power for a GLMM, via the `simr` package.
#

# primary model glmer(moral_sycophancy ~ stakes_num + (1|pair_id), family = binomial)
# using simr package to simulate data and fit the model
# 1. fit the glmer model to the pilot data
# 2. fixef(model)["stakes_num"] <- SESOI (set the effect to test)
# 3. powerSim(model, nsim, test=fixed(...)) -> power at pilot's N
# 4. extend(model, along="pair_id", n=...) -> power at different Ns
# 5. powerCurve(model_ext, along="pair_id") -> power across a range of Ns

# H1 (one-sided): B_stakes < 0.
# Usage:  Rscript elephant_power_analysis.R <long_form.csv> [<model_tag>]
# =============================================================================

suppressPackageStartupMessages({
  library(lme4)
  library(simr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2)
  stop("Usage: Rscript elephant_power_analysis.R <long_form.csv> <model_tag>")
long_csv  <- args[1]
long <- read.csv(long_csv)
stopifnot(all(c("pair_id", "stakes_num", "moral_sycophancy") %in% names(long)))
long$pair_id <- factor(long$pair_id)

m <- glmer(moral_sycophancy ~ stakes_num + (1 | pair_id),
           data = long, family = binomial)

sd_pair   <- as.numeric(attr(VarCorr(m)$pair_id, "stddev"))
intercept <- as.numeric(fixef(m)["(Intercept)"])
n_pairs   <- nlevels(long$pair_id)

if (!is.finite(sd_pair) || sd_pair > 5) {
  stop(sprintf(paste0(
    "SD_pair = %.3f looks degenerate. simr would inherit this broken\n",
    "variance component. Use the swept-simulation power analysis instead."),
    sd_pair))
}

SESOI <- -0.20
NSIM <- 200
N_GRID <- c(200, 350, 500, 750, 1000)
ALPHA <- 0.05

cat("=================================================================\n")
cat("ELEPHANT MORAL-SYCOPHANCY -- POWER ANALYSIS (simr, GLMM simulation)\n")
cat("=================================================================\n")
cat(sprintf("Refit: %d pairs, SD_pair=%.4f, intercept=%.4f (rate %.3f)\n",
            n_pairs, sd_pair, intercept, plogis(intercept)))
cat(sprintf("SESOI (the effect we power for): %+.3f log-odds per stakes step\n",
            SESOI))
p0 <- plogis(intercept); p2 <- plogis(intercept + 2 * SESOI)
cat(sprintf("  => moral_sycophancy %.3f -> %.3f across stakes 0->2 (delta %+.3f)\n",
            p0, p2, p2 - p0))
cat(sprintf("nsim=%d per estimate, alpha=%.2f\n\n", NSIM, ALPHA))

# ---- STEP 2: set the fixed effect to the SESOI ------------------------------
fixef(m)["stakes_num"] <- SESOI

# ---- STEP 3: power at the pilot's current N ---------------------------------
cat("STEP 3 -- power at the pilot's current N:\n")
ps <- powerSim(m, test = fixed("stakes_num", "z"), nsim = NSIM,
               progress = FALSE)
print(ps)

# ---- STEP 4 + 5: extend the design and trace a power curve ------------------
# extend() enlarges the design along the pair_id grouping so simr can simulate
# larger studies than the pilot. powerCurve() then runs powerSim at a sequence
# of N values. (simr ref manual, ?extend, ?powerCurve.)
# Only extend when N_GRID exceeds the pilot's pair count -- simr can't shrink
# a grouping factor via extend(), so passing n < n_pairs throws a factor-levels
# error. For breaks at or below n_pairs, powerCurve subsets directly.
cat("\nSTEP 4-5 -- power curve across pair counts:\n")
m_used <- if (max(N_GRID) > n_pairs)
  extend(m, along = "pair_id", n = max(N_GRID)) else m
pc <- powerCurve(m_used, test = fixed("stakes_num", "z"),
                 along = "pair_id", breaks = N_GRID,
                 nsim = NSIM, progress = FALSE)
print(pc)