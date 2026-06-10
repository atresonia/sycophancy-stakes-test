# =============================================================================
# Degeneracy fallback fit -- binary stakes experiments (MMLU & ELEPHANT)
#
# Use when the pre-registered glmer logistic mixed model DEGENERATES (random-
# intercept SD in the double digits, B_stakes pinned / huge SE) or fails to
# converge. Causes: rare-event + small within-cluster variation -> the per-
# cluster random intercept runs to the boundary (cluster separation).
#
# Fallback (pre-registered): drop the random intercept, fit plain logistic
# regression with CLUSTER-ROBUST standard errors clustered on the same
# grouping variable (sandwich::vcovCL; Cameron & Miller 2015). The random
# intercept modelled within-cluster correlation; the cluster-robust SE
# accounts for it instead. No random effect -> no cluster separation -> the
# fit is identifiable.
#
# The coefficient is the MARGINAL (population-averaged) stakes effect, not the
# cluster-conditional effect a GLMM gives -- note this in the prereg amendment.
#
# H1 (one-sided): B_stakes < 0.
#
# Writes: appends one row to <model>_fit_results.csv with the SAME schema the
# glmer scripts use, plus method="glm_clustered", degenerate=FALSE.
#
# Usage:
#   Rscript glm_fallback_fitting.R <long_form.csv> <experiment> <model> <outcome_col> <cluster_col>
# e.g.
#   Rscript glm_fallback_fitting.R mmlu_gpt_long.csv     mmlu     gpt-oss-120b flipped_to_user  question_id
#   Rscript glm_fallback_fitting.R elephant_gem_long.csv elephant gemini-3.1   moral_sycophancy pair_id
# =============================================================================

suppressPackageStartupMessages({
  library(sandwich)
  library(lmtest)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5)
  stop(paste("Usage: Rscript glm_fallback_fitting.R <long_form.csv>",
             "<experiment> <model> <outcome_col> <cluster_col>"))
long_csv    <- args[1]
experiment  <- args[2]
model       <- args[3]
outcome_col <- args[4]
cluster_col <- args[5]

long <- read.csv(long_csv)
for (c in c(outcome_col, cluster_col, "stakes_num"))
  if (!c %in% names(long))
    stop(sprintf("column '%s' not in %s", c, long_csv))

long$.y       <- long[[outcome_col]]
long$.cluster <- factor(long[[cluster_col]])

cat("=================================================================\n")
cat(sprintf("%s -- DEGENERACY FALLBACK FIT (glm + cluster-robust SEs)\n",
            toupper(experiment)))
cat("=================================================================\n")
cat(sprintf("Rows: %d  |  Clusters: %d  |  outcome: %s  |  cluster: %s\n",
            nrow(long), nlevels(long$.cluster), outcome_col, cluster_col))
cat("\noutcome rate by stakes_num:\n")
print(do.call(rbind, lapply(split(long$.y, long$stakes_num),
              function(x) c(rate = mean(x), n = length(x)))))

cat("\n--- glm(", outcome_col, "~ stakes_num, binomial) ---\n", sep = "")
m <- glm(.y ~ stakes_num, data = long, family = binomial)

V  <- vcovCL(m, cluster = long$.cluster, cadjust = TRUE)
ct <- coeftest(m, vcov. = V)
print(round(ct, 5))

beta  <- ct["stakes_num", "Estimate"]
se    <- ct["stakes_num", "Std. Error"]
p2    <- ct["stakes_num", "Pr(>|z|)"]
p1    <- if (beta < 0) p2 / 2 else 1 - p2 / 2
ci_lo <- beta - 1.96 * se
ci_hi <- beta + 1.96 * se

cat("\n--- Stakes coefficient (B_stakes, log-odds per stakes step) ---\n")
cat(sprintf("  estimate          : %+.5f  (odds ratio %.4f)\n", beta, exp(beta)))
cat(sprintf("  cluster-robust SE : %.5f\n", se))
cat(sprintf("  two-sided p       : %.5f\n", p2))
cat(sprintf("  one-sided p       : %.5f   (H1: B_stakes < 0)\n", p1))
cat(sprintf("  95%% CI (cluster-robust Wald): [%+.5f, %+.5f]\n", ci_lo, ci_hi))
cat("  NOTE: marginal (population-averaged) stakes effect.\n")

fit_row <- data.frame(
  experiment  = experiment,
  model       = model,
  beta        = beta,
  se          = se,
  p_one_sided = p1,
  ci_lo       = ci_lo,
  ci_hi       = ci_hi,
  n           = nlevels(long$.cluster),
  sd_random   = NA,
  degenerate  = FALSE,
  method      = "glm_clustered"
)

args_all    <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir  <- if (length(script_path)) dirname(normalizePath(script_path)) else getwd()
fr_path <- file.path(script_dir, paste0(model, "_fit_results.csv"))

# if an existing fit_results.csv (written by the glmer scripts) lacks the
# `method` column, add it as "glmer" so the append stays schema-consistent.
if (file.exists(fr_path)) {
  existing <- read.csv(fr_path)
  if (!"method" %in% names(existing)) {
    existing$method <- "glmer"
    write.csv(existing, fr_path, row.names = FALSE)
  }
}
write.table(fit_row, fr_path, sep = ",", row.names = FALSE,
            col.names = !file.exists(fr_path), append = file.exists(fr_path))
cat(sprintf("\nAppended fallback fit row to %s\n", fr_path))
cat(sprintf("  (experiment=%s, model=%s, method=glm_clustered)\n",
            experiment, model))
cat("=================================================================\n")