# =============================================================================
# Primary model fit -- MMLU bet-stakes experiment

# Model: flipped_to_user ~ stakes_num + (1|question_id) family = binomial
# - flipped_to_user: 1 if model abandons its kb answer and adopts user's (incorrect) answer. 0 otherwise
# - stakes_num: 0, 1, 2 for low, medium, high stakes
# - logistic: glmer, B_stakes is on log-odds scale

# Pre-registration: H1: B_stakes < 0
# (log-odds of flipping to user's incorrect answer decreases as stakes increase). One-sided.

# Input: incorrect-answer long-form CSV from mmlu_metrics.build_long_form_csv()
# Columns: question_id, stakes_num, flipped_to_user, flipped_away_from_both, accuracy

# Optional: correct-answer long-form CSV from mmlu_metrics.build_long_form_csv()

# Usage: Rscript mmlu_model_fitting.R <incorrect_long_form_csv> <model_name> [correct_long_form_csv]
# Example: Rscript mmlu_model_fitting.R data/mmlu/mmlu_gemini_100_long_form.csv gemini data/mmlu/mmlu_gemini_100_long_form_correct.csv
# =============================================================================

suppressPackageStartupMessages({
  library(lme4)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2)
    stop("Usage: Rscript mmlu_model_fitting.R <incorrect_long_form_csv> [correct_long_form_csv] <model_name>")

incorrect_long_csv <- args[1]
model_name <- args[2]

long_inc <- read.csv(incorrect_long_csv)
stopifnot(all(c("question_id", "stakes_num", "flipped_to_user") %in% names(long_inc)))
long_inc$question_id <- factor(long_inc$question_id)

cat("=================================================================\n")
cat("MMLU BET-STAKES -- PRIMARY MODEL FIT (logistic mixed model)\n")
cat("=================================================================\n")
cat(sprintf("Rows: %d  |  Questions: %d  |  stakes levels: %s\n",
            nrow(long_inc), nlevels(long_inc$question_id),
            paste(sort(unique(long_inc$stakes_num)), collapse = ",")))
cat("\nflipped_to_user rate by stakes_num:\n")
print(do.call(rbind, lapply(split(long_inc$flipped_to_user, long_inc$stakes_num),
              function(x) c(rate = mean(x), n = length(x)))))

cat("\n--- Model: flipped_to_user ~ stakes_num + (1|question_id), binomial ---\n")
m <- glmer(flipped_to_user ~ stakes_num + (1|question_id), data = long_inc, family = binomial)
print(round(summary(m)$coefficients, 5))

co   <- summary(m)$coefficients
beta <- co["stakes_num", "Estimate"]
se   <- co["stakes_num", "Std. Error"]
p2   <- co["stakes_num", "Pr(>|z|)"]      # glmer gives a z test, not t
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

sd_q <- as.numeric(attr(VarCorr(m)$question_id, "stddev"))
intercept <- as.numeric(fixef(m)["(Intercept)"])
cat("\n--- Components for the power analysis ---\n")
cat(sprintf("  SD(question intercept), log-odds : %.4f\n", sd_q))
cat(sprintf("  intercept (log-odds)             : %+.4f\n", intercept))
cat(sprintf("  => baseline flip rate at stakes=0: %.4f\n",
            plogis(intercept)))

# degeneracy flag
degenerate <- (!is.finite(se)) || se > 5 || sd_q > 5
if (degenerate) {
  cat("\n*** WARNING: fit looks DEGENERATE (SD_q or SE implausibly large).\n")
  cat("*** Per the pre-registered fallback, use glm() with question-clustered\n")
  cat("*** robust SEs instead. Do NOT report the numbers above.\n")
}

# ---- persist the fit -------------------------------------------------------
args_all    <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args_all[grep("--file=", args_all)])
script_dir  <- if (length(script_path)) dirname(normalizePath(script_path)) else getwd()

rds_path <- file.path(script_dir, paste0(model_name, "_mmlu_pilot_model.rds"))
saveRDS(list(model = m, beta = beta, se = se,
             sd_q = sd_q, intercept = intercept,
             n_questions = nlevels(long_inc$question_id)),
        file = rds_path)
cat(sprintf("\nSaved fit to %s\n", rds_path))

# ---- optional: control model on the _correct placement --------------------
if (length(args) > 2) {
  cat("\n-----------------------------------------------------------------\n")
  cat("CONTROL MODEL -- user holds the CORRECT answer (no trend predicted)\n")
  correct_long_csv <- args[3]
  long_c <- read.csv(correct_long_csv)
  long_c$question_id <- factor(long_c$question_id)
  mc <- glmer(flipped_to_user ~ stakes_num + (1 | question_id),
              data = long_c, family = binomial)
  cc <- summary(mc)$coefficients
  cat(sprintf("  B_stakes = %+.5f  SE = %.5f  two-sided p = %.5f\n",
              cc["stakes_num", "Estimate"], cc["stakes_num", "Std. Error"],
              cc["stakes_num", "Pr(>|z|)"]))
  cat("  (expectation: B_stakes not significantly different from 0)\n")
}
cat("=================================================================\n")

# ---- append one row to <model_name>_fit_results.csv ------------------------
# summarize.py reads this file to fill the B_stakes slot automatically.
fit_row <- data.frame(
  experiment  = "mmlu",
  model       = model_name,
  beta        = beta,
  se          = se,
  p_one_sided = p1,
  ci_lo       = ci_lo,
  ci_hi       = ci_hi,
  n           = nlevels(long_inc$question_id),
  sd_random   = sd_q,
  degenerate  = degenerate
)
fr_path <- file.path(script_dir, paste0(model_name, "_fit_results.csv"))
write.table(fit_row, fr_path, sep = ",", row.names = FALSE,
            col.names = !file.exists(fr_path), append = file.exists(fr_path))
cat(sprintf("Appended fit row to %s  (experiment=mmlu, model=%s)\n",
            fr_path, model_name))
if (degenerate)
  cat("  NOTE: fit flagged DEGENERATE -- B_stakes unreliable, use the glm fallback.\n")
