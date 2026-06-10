# =============================================================================
# Mixed-effects model fit -- essay grading stakes experiment
#
# Model: grade_inflation ~ stakes_num + (1|essay_id) 
# NOTE: a random slope can't be added (1 + stakes_num | essay_id) 
# because we just run a single call per essay x stakes level
# input: long-form CSV from essay_metrics.build_long_form_csv()
# columns: essay_id, stakes_num, grade_inflation
# Pre-registration hypothesis: B_stakes < 0 (the cond-kb shift becomes more negative as stakes increase)

# workflow:
# 1. refit lmer on long-form CSV
# 2. fixef(model)["stakes_num"] <- SESOI (set the effect to test)
# 3. powerSim(model, nsim, test=fixed(...)) -> power at pilot's N
# 4. extend(model, along="essay_id", n=...) -> power at different Ns
# 5. powerCurve(model_ext, along="essay_id") -> power across a range of Ns

# H1 (one-sided): B_stakes < 0 (the cond-kb shift becomes more negative as stakes increase)

# Usage:  Rscript essay_power_analysis.R <csv_file>
# Example:  Rscript essay_power_analysis.R experiments/essay_grading_analysis/metrics/long_form.csv
# =============================================================================

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(simr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript essay_power_analysis.R <csv_file>")
long_csv <- args[1]

long <- read.csv(long_csv)
stopifnot(all(c("essay_id", "stakes_num", "grade_inflation") %in% names(long)))
long$essay_id <- factor(long$essay_id)

# STEP 1: refit the primary model (so simr has the data frame).
m = lmer(grade_inflation ~ stakes_num + (1 | essay_id), data = long, REML = FALSE)
sd_essay  <- as.numeric(attr(VarCorr(m)$essay_id, "stddev"))
sd_resid  <- sigma(m)
intercept <- as.numeric(fixef(m)["(Intercept)"])
n_essays  <- nlevels(long$essay_id)


# ---- analysis settings ------------------------------------------------------
# SESOI grid, grade-units per stakes step, negative (H1a: B_stakes < 0).
SESOI_GRID   <- c(-0.05, -0.10, -0.15, -0.25)
PREREG_SESOI <- -0.10        # the pre-registered SESOI -- powerCurve uses this
N_GRID       <- c(100, 200, 350, 500)   # essay counts for the power curve
NSIM         <- 200          # >=500 for a reported figure (Arend & Schaefer 2019)
ALPHA        <- 0.05

cat("=================================================================\n")
cat("ESSAY GRADING STAKES -- POWER ANALYSIS (simr, LMM simulation)\n")
cat("=================================================================\n")
cat(sprintf("Refit: %d essays  SD_essay=%.4f  SD_resid=%.4f  intercept=%.4f\n",
            n_essays, sd_essay, sd_resid, intercept))
cat(sprintf("nsim=%d per estimate, alpha=%.2f (powerSim test is two-sided)\n\n",
            NSIM, ALPHA))

# ---- STEPS 2-3: power at the pilot's N, across the SESOI grid ---------------
# For each SESOI we overwrite the stakes fixed effect and run powerSim. This
# powers for a CHOSEN effect, not the pilot's observed one.
cat("POWER at the pilot's N (=", n_essays, " essays), by SESOI:\n", sep = "")
cat(sprintf("%-10s %-22s\n", "SESOI", "power (95% CI)"))
cat(strrep("-", 34), "\n", sep = "")
for (b in SESOI_GRID) {
  m_b <- m
  fixef(m_b)["stakes_num"] <- b          # set the effect to this SESOI
  ps <- powerSim(m_b, test = fixed("stakes_num", "t"),
                 nsim = NSIM, progress = FALSE)
  s <- summary(ps)                       # power + CI as a data frame
  cat(sprintf("%-10s %.1f%% (%.1f, %.1f)\n",
              sprintf("%+.2f", b),
              100 * s$mean, 100 * s$lower, 100 * s$upper))
}

# ---- STEPS 4-5: power curve at the PRE-REGISTERED SESOI ---------------------
# extend() enlarges the design along essay_id; powerCurve() runs powerSim at a
# sequence of essay counts. Run once, at the pre-registered SESOI.
cat(sprintf("\nPOWER CURVE at the pre-registered SESOI = %+.2f:\n", PREREG_SESOI))
m_prereg <- m
fixef(m_prereg)["stakes_num"] <- PREREG_SESOI
m_ext <- extend(m_prereg, along = "essay_id", n = max(N_GRID))
pc <- powerCurve(m_ext, test = fixed("stakes_num", "t"),
                 along = "essay_id", breaks = N_GRID,
                 nsim = NSIM, progress = FALSE)
print(pc)