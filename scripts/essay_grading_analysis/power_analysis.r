# Power analysis for the essay grading stakes experiment using Monte Carlo simulation

# Load the necessary libraries
suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

# allow args to select which model file to use
args <- commandArgs(trailingOnly = TRUE)
fitted_models_file <- if (length(args) >= 1) args[1] else "fitted_models.rds"
model_label <- if (length(args) >= 2) args[2] else "model"
n_sims <- if (length(args) >= 3) as.numeric(args[3]) else 100

cat(sprintf("Loading fitted models from: %s\n", fitted_models_file))
cat(sprintf("Model label: %s\n", model_label))
cat(sprintf("N_SIMS per cell: %d\n\n", n_sims))

# Load model 2 (found to be the best model from mixed_effects_test.r)
m2 <- readRDS(fitted_models_file)$m2

beta_obs <- fixef(m2)["stakes_ordinal"]
intercept <- fixef(m2)["(Intercept)"]
vc <- VarCorr(m2)
sd_essay <- sqrt(attr(vc$essay_id, "stddev")^2)
sd_variant <- sqrt(attr(vc$variant_idx, "stddev")^2)
sigma_resid <- sigma(m2)

cat("Pilot M2 parameters:\n")
cat(sprintf("  Intercept     = %.4f\n", intercept))
cat(sprintf("  beta_observed = %.5f\n", beta_obs))
cat(sprintf("  SD(essay)     = %.4f\n", as.numeric(sd_essay)))
cat(sprintf("  SD(variant)   = %.4f\n", as.numeric(sd_variant)))
cat(sprintf("  SD(residual)  = %.4f\n", sigma_resid))
cat("\n")

# Direction detection
direction <- if (beta_obs < 0) "negative" else "positive"
cat(sprintf("Detected pilot direction: %s\n", direction))
cat(sprintf("One-sided test direction: alternative = '%s'\n",
            if (direction == "negative") "less" else "greater"))
cat("\n")


simulate_data <- function(n_essays, n_stakes, n_variants, true_beta, intercept,
                          sd_essay, sd_variant, sd_resid, seed) {
    set.seed(seed)
    design <- expand.grid(
        essay_id = 1:n_essays,
        stakes_ordinal = 0:(n_stakes - 1),
        variant_idx = 1:n_variants
    )

    # random effects
    re_essay <- rnorm(n_essays, mean = 0, sd = sd_essay)
    re_variant <- rnorm(n_variants, mean = 0, sd = sd_variant)

    # linear predictor m2 ~ stakes_ordinal + (1 | essay_id) + (1 | variant_idx)
    mu <- intercept + true_beta * design$stakes_ordinal + re_essay[design$essay_id] + re_variant[design$variant_idx]

    # residual noise
    design$grade_num <- mu + rnorm(nrow(design), mean = 0, sd = sd_resid)
    design$essay_id <- factor(design$essay_id)
    design$variant_idx <- factor(design$variant_idx)
    return(design)
}

# Fit M2 to simulated data and extract p
fit_and_test <- function(sim_df) {
  result <- tryCatch({
    m <- suppressMessages(suppressWarnings(
      lmer(grade_num ~ stakes_ordinal + (1 | essay_id) + (1 | variant_idx),
           data = sim_df, REML = FALSE)
    ))
    coefs <- summary(m)$coefficients
    list(beta = coefs["stakes_ordinal", "Estimate"],
         p = coefs["stakes_ordinal", "Pr(>|t|)"])
  }, error = function(e) list(beta = NA, p = NA))
  return(result)
}
 
# Compute power: fraction of simulations with p<alpha AND direction matches
compute_power <- function(n_essays, true_beta, n_sims, alpha = 0.05) {
  n_sig <- 0
  n_valid <- 0
  for (s in 1:n_sims) {
    sim_df <- simulate_data(n_essays, n_stakes = 3, n_variants = 10,
                             true_beta = true_beta,
                             intercept = intercept,
                             sd_essay = as.numeric(sd_essay),
                             sd_variant = as.numeric(sd_variant),
                             sd_resid = sigma_resid,
                             seed = 10000 + s + 1000 * n_essays)
    res <- fit_and_test(sim_df)
    if (!is.na(res$p)) {
      n_valid <- n_valid + 1
        # One-sided p < alpha AND direction matches predicted
      if (res$p / 2 < alpha && sign(res$beta) == sign(true_beta)) {
          n_sig <- n_sig + 1
      }
    }
  }
  return(list(power = n_sig / n_valid, n_valid = n_valid, n_sims = n_sims))
}
 
# Run power analysis
cat("======================================================================\n")
cat("Monte Carlo power analysis (M2)\n")
cat("======================================================================\n")
cat("Target power threshold: 0.80 (Cohen 1988)\n")
cat("One-sided alpha = 0.05, predicted direction: negative beta\n\n")
 
# Reduced number of sims for speed; for paper use n_sims >= 500
N_SIMS <- 100
 
direction_sign <- sign(beta_obs)
sesoi_small <- direction_sign * 0.05
sesoi_large <- direction_sign * 0.10
 
targets <- list(
  list(label = sprintf("Pilot observed (%+0.4f)", beta_obs),
       beta = beta_obs),
  list(label = sprintf("Half pilot (%+0.4f)", beta_obs / 2),
       beta = beta_obs / 2),
  list(label = sprintf("SESOI (%+0.3f)", sesoi_small),
       beta = sesoi_small),
  list(label = sprintf("SESOI (%+0.3f)", sesoi_large),
       beta = sesoi_large)
)
ns <- c(95, 200, 300, 500)
 
cat(sprintf("%-32s", "Beta"))
for (n in ns) cat(sprintf("N=%-7d", n))
cat("\n")
cat(strrep("-", 32 + length(ns) * 9), "\n", sep="")
 
for (target in targets) {
  cat(sprintf("%-32s", target$label))
  for (n in ns) {
    res <- compute_power(n, target$beta, N_SIMS)
    cat(sprintf("%.2f (%2d) ", res$power, res$n_valid))
  }
  cat("\n")
}