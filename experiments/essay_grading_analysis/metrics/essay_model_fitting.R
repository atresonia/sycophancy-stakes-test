# =============================================================================
# Mixed-effects model fit -- essay grading stakes experiment
#
# Model: grade_inflation ~ stakes_num + (1|essay_id) 
# NOTE: a random slope can't be added (1 + stakes_num | essay_id) 
# because we just run a single call per essay x stakes level
# input: long-form CSV from essay_metrics.build_long_form_csv()
# columns: essay_id, stakes_num, grade_inflation
# Pre-registration hypothesis: B_stakes < 0 (the cond-kb shift becomes more negative as stakes increase)

# Usage:  Rscript essay_model_fitting.R <long_form_csv> <model_name>
# Example:  Rscript essay_model_fitting.R experiments/essay_grading_analysis/metrics/long_form.csv gemini
# =============================================================================

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript essay_model_fit.R <long_form_csv> <model_name>")
long_csv <- args[1]
model_name <- args[2]

long <- read.csv(long_csv)
stopifnot(all(c("essay_id", "stakes_num", "grade_inflation") %in% names(long)))
long$essay_id <- factor(long$essay_id)

cat(sprintf("Rows: %d  |  Essays: %d  |  stakes levels: %s\n",
            nrow(long), nlevels(long$essay_id),
            paste(sort(unique(long$stakes_num)), collapse = ",")))

cat("\nGrade inflation (cond - kb) mean by stakes_num:\n")
agg <- aggregate(grade_inflation ~ stakes_num, data = long, FUN = function(x) c(mean=mean(x), sd=sd(x), n=length(x)))

print(agg)

# REML (Restricted Maximum Likelihood) = FALSE uses ML (Maximum Likelihood) estimation.
# REML = TRUE estimates the variance components 
cat("\n--- Model: grade_inflation ~ stakes_num + (1 | essay_id) ---\n")
m <- lmer(grade_inflation ~ stakes_num + (1 | essay_id), data = long, REML = FALSE)

print(round(summary(m)$coefficients, 5))
cat("\nVariance components:\n")
print(VarCorr(m))
cat(sprintf("Residual SD: %.4f\n", sigma(m)))

co <- summary(m)$coefficients
beta <- co["stakes_num", "Estimate"]
se   <- co["stakes_num", "Std. Error"]
p2   <- co["stakes_num", "Pr(>|t|)"]
# one-sided p for H1a: B_stakes < 0
p1   <- if (beta < 0) p2 / 2 else 1 - p2 / 2

cat("\n--- Stakes coefficient (B_stakes) ---\n")
cat(sprintf("  estimate    : %+.5f\n", beta))
cat(sprintf("  std. error  : %.5f\n", se))
cat(sprintf("  two-sided p : %e\n", p2))
cat(sprintf("  one-sided p : %e   (H1a: B_stakes < 0)\n", p1))
# use Wald for CI because we are assuming a normal distribution
ci <- tryCatch(confint(m, parm = "stakes_num", method = "Wald"),
               error = function(e) NULL)
ci_lo <- if (!is.null(ci)) ci[1, 1] else NA
ci_hi <- if (!is.null(ci)) ci[1, 2] else NA
if (!is.null(ci)) {
  cat(sprintf("  95%% CI: [%+.5f, %+.5f]\n", ci[1, 1], ci[1, 2]))
}

# Use REML for variant components
m_reml <- lmer(grade_inflation ~ stakes_num + (1 | essay_id),
               data = long, REML = TRUE)
sd_essay <- as.numeric(attr(VarCorr(m_reml)$essay_id, "stddev"))
sd_resid <- sigma(m_reml)
intercept <- as.numeric(fixef(m_reml)["(Intercept)"])

cat("\n--- Variance components (REML, for power analysis) ---\n")
cat(sprintf("  SD(essay intercept) : %.4f\n", sd_essay))
cat(sprintf("  SD(residual)        : %.4f\n", sd_resid))
cat(sprintf("  intercept           : %+.4f\n", intercept))

degenerate <- (!is.finite(se)) || se > 5 || sd_essay > 10

# ---- persist the fit -------------------------------------------------------
# get directory that this script is in
args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", args_all[grep(file_arg, args_all)])
script_dir <- dirname(normalizePath(script_path))
rds_path <- file.path(script_dir, paste0(model_name, "_pilot_model.rds"))
saveRDS(list(model = m, model_reml = m_reml, beta = beta, se = se,
             sd_essay = sd_essay, sd_resid = sd_resid, intercept = intercept,
             n_essays = nlevels(long$essay_id)),
        file = rds_path)
cat(sprintf("\nSaved fit to %s\n", rds_path))

# ---- append one row to the shared fit_results.csv --------------------------
# summarize.py reads this file to fill the B_stakes slot automatically.
fit_row <- data.frame(
  experiment  = "essay",
  model       = model_name,
  beta        = beta,
  se          = se,
  p_one_sided = p1,
  ci_lo       = ci_lo,
  ci_hi       = ci_hi,
  n           = nlevels(long$essay_id),
  sd_random   = sd_essay,
  degenerate  = degenerate
)


fr_path <- file.path(script_dir, paste0(model_name, "_fit_results.csv"))
write.table(fit_row, fr_path, sep = ",", row.names = FALSE,
            col.names = !file.exists(fr_path), append = file.exists(fr_path))
cat(sprintf("Appended fit row to %s  (experiment=essay, model=%s)\n",
            fr_path, model_name))