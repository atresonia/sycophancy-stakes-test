#  =============================================================================
# Primary model fit -- ELEPHANT moral-sycophancy stakes experiment
#
# Model:  moral_sycophancy ~ stakes_num + (1 | pair_id)   family = binomial
# Usage:  Rscript elephant_model_fitting.R <long_form.csv> [<model_tag>]
# =============================================================================

suppressPackageStartupMessages({
  library(lme4)
})
 
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2)
  stop("Usage: Rscript elephant_model_fitting.R <long_form.csv> [<model_tag>]")
long_csv  <- args[1]
model_tag <- if (length(args) >= 2) args[2] else "model"
 
long <- read.csv(long_csv)
stopifnot(all(c("pair_id", "stakes_num", "moral_sycophancy") %in% names(long)))
long$pair_id <- factor(long$pair_id)

cat("=================================================================\n")
cat("ELEPHANT MORAL-SYCOPHANCY -- PRIMARY MODEL FIT (logistic mixed model)\n")
cat("=================================================================\n")
cat(sprintf("Rows: %d  |  Pairs: %d  |  stakes levels: %s\n",
            nrow(long), nlevels(long$pair_id),
            paste(sort(unique(long$stakes_num)), collapse = ",")))

# split up moral_sycophancy into columns for each stakes level and take mean
cat("\nmoral_sycophancy rate by stakes_num:\n")
print(do.call(rbind, lapply(split(long$moral_sycophancy, long$stakes_num),
              function(x) c(rate = mean(x), n = length(x)))))

cat("\n--- Model: moral_sycophancy ~ stakes_num + (1|pair_id), binomial ---\n")
m <- glmer(moral_sycophancy ~ stakes_num + (1 | pair_id),
           data = long, family = binomial)

cat("\nFixed effects (log-odds scale):\n")
print(round(summary(m)$coefficients, 5))

co   <- summary(m)$coefficients
beta <- co["stakes_num", "Estimate"]
se   <- co["stakes_num", "Std. Error"]
p2   <- co["stakes_num", "Pr(>|z|)"]
p1   <- if (beta < 0) p2 / 2 else 1 - p2 / 2   # one-sided H1: B_stakes < 0

cat("\n--- Stakes coefficient (B_stakes, log-odds per stakes step) ---\n")
cat(sprintf("  estimate     : %+.5f  (odds ratio %.4f)\n", beta, exp(beta)))
cat(sprintf("  std. error   : %.5f\n", se))
cat(sprintf("  two-sided p  : %.5f\n", p2))
cat(sprintf("  one-sided p  : %.5f   (H1: B_stakes < 0)\n", p1))
ci <- tryCatch(confint(m, parm = "stakes_num", method = "Wald"),
               error = function(e) NULL)
ci_lo <- if (!is.null(ci)) ci[1, 1] else NA
ci_hi <- if (!is.null(ci)) ci[1, 2] else NA
if (!is.null(ci))
  cat(sprintf("  95%% CI (Wald): [%+.5f, %+.5f]  (log-odds)\n", ci[1, 1], ci[1, 2]))
 
sd_pair   <- as.numeric(attr(VarCorr(m)$pair_id, "stddev"))
intercept <- as.numeric(fixef(m)["(Intercept)"])
cat("\n--- Components ---\n")
cat(sprintf("  SD(pair intercept), log-odds : %.4f\n", sd_pair))
cat(sprintf("  intercept (log-odds)         : %+.4f\n", intercept))
cat(sprintf("  => baseline-stakes rate      : %.4f\n", plogis(intercept)))
 
# degeneracy flag -- the Elephant pilots hit exactly this
degenerate <- (!is.finite(se)) || se > 5 || sd_pair > 5
if (degenerate) {
  cat("\n*** WARNING: fit looks DEGENERATE ...\n")
  cat("*** Per the pre-registered fallback, use glm() with pair-clustered\n")
  cat("*** robust SEs instead. Do NOT report the numbers above.\n")
}
 
# ---- persist the fit -------------------------------------------------------
# get directory that this script is in
args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir  <- if (length(script_path)) dirname(normalizePath(script_path)) else getwd()
rds_path <- file.path(script_dir, paste0(model_tag, "_pilot_model.rds"))
saveRDS(list(model = m, beta = beta, se = se,
             sd_pair = sd_pair, intercept = intercept,
             n_pairs = nlevels(long$pair_id)),
        file = rds_path)
cat(sprintf("\nSaved fit to %s\n", rds_path))

# ---- append one row to the shared fit_results.csv --------------------------
# summarize.py reads this file to fill the B_stakes slot automatically.
fit_row <- data.frame(
  experiment  = "elephant",
  model       = model_tag,
  beta        = beta,
  se          = se,
  p_one_sided = p1,
  ci_lo       = ci_lo,
  ci_hi       = ci_hi,
  n           = nlevels(long$pair_id),
  sd_pair     = sd_pair,
  degenerate  = degenerate
)

fr_path <- file.path(script_dir, paste0(model_tag, "_fit_results.csv"))
write.table(fit_row, fr_path, sep = ",", row.names = FALSE,
            col.names = !file.exists(fr_path), append = file.exists(fr_path))
cat(sprintf("Appended fit row to %s  (experiment=elephant, model=%s)\n",
            fr_path, model_tag))
if (degenerate)
  cat("  NOTE: fit flagged DEGENERATE -- B_stakes unreliable, use the glm fallback.\n")