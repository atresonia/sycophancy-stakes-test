# =============================================================================
# Auto-generate metrics summary table for paper
# =============================================================================
# Reads fitted_models.rds files from essay and MMLU analyses for each model
# (Gemini, OpenAI, Claude) and produces a single summary CSV.
#
# Usage:
#   Rscript build_paper_metrics.R [output_dir]
#
# Expects in working directory or specified paths:
#   essay_fitted_models_gemini.rds, essay_fitted_models_openai.rds, etc.
#   mmlu_fitted_models_gemini.rds, ...
 
suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "."

# Generic coef extractor
get_term <- function(model, term, h1_dir) {
  c <- summary(model)$coefficients
  if (!term %in% rownames(c)) return(rep(NA, 6))
  is_glmm <- "z value" %in% colnames(c)
  stat_col <- if (is_glmm) "z value" else "t value"
  p_col <- if (is_glmm) "Pr(>|z|)" else "Pr(>|t|)"
  beta <- c[term, "Estimate"]
  se <- c[term, "Std. Error"]
  stat <- c[term, stat_col]
  p2 <- c[term, p_col]
  df <- if (is_glmm) NA else c[term, "df"]
  # one-sided p in H1 direction
  if (h1_dir == "neg") {
    p1 <- if (stat < 0) p2/2 else 1 - p2/2
  } else {
    p1 <- if (stat > 0) p2/2 else 1 - p2/2
  }
  c(beta = beta, se = se, stat = stat, df = df, p2 = p2, p1 = p1)
}

# Variance components extractor
get_vc <- function(model, grp, term = "(Intercept)") {
  vc <- VarCorr(model)
  if (!grp %in% names(vc)) return(NA)
  v <- vc[[grp]]
  sds <- attr(v, "stddev")
  if (!term %in% names(sds)) return(NA)
  as.numeric(sds[term])
}

# =============================================================================
# Essay metrics
# =============================================================================
build_essay_row <- function(rds_path, model_label) {
  if (!file.exists(rds_path)) {
    cat(sprintf("MISSING: %s\n", rds_path))
    return(NULL)
  }
  models <- readRDS(rds_path)
  m2 <- models$m2
  m4 <- models$m4
 
  # LR test M4 vs M2 — was random slope justified?
  lr <- anova(m2, m4)
  slope_p <- lr$`Pr(>Chisq)`[2]
  slope_justified <- slope_p < 0.05
 
  primary <- if (slope_justified) m4 else m2
  primary_label <- if (slope_justified) "M4_maximal" else "M2_parsimonious"
 
  stakes <- get_term(primary, "stakes_ordinal", h1_dir = "neg")
  vc_essay <- get_vc(primary, "essay_id", "(Intercept)")
  vc_variant <- get_vc(primary, "variant_idx", "(Intercept)")
  vc_slope <- get_vc(primary, "essay_id", "stakes_ordinal")  # NA if no slope
 
  list(
    experiment = "essay",
    model = model_label,
    n_questions = if (!is.null(models$n_questions)) models$n_questions else NA,
    primary_spec = primary_label,
    beta_stakes = stakes["beta"],
    se_stakes = stakes["se"],
    stat = stakes["stat"],
    df = stakes["df"],
    p_two_sided = stakes["p2"],
    p_one_sided_H1neg = stakes["p1"],
    sd_essay = vc_essay,
    sd_variant = vc_variant,
    sd_slope = vc_slope,
    lr_slope_p = slope_p,
    slope_justified = slope_justified
  )
}

# =============================================================================
# MMLU metrics
# =============================================================================
build_mmlu_row <- function(rds_path, model_label) {
  if (!file.exists(rds_path)) {
    cat(sprintf("MISSING: %s\n", rds_path))
    return(NULL)
  }
  models <- readRDS(rds_path)
 
  inc <- get_term(models$primary_inc, "stakes_ordinal", h1_dir = "neg")
  cor <- get_term(models$primary_cor, "stakes_ordinal", h1_dir = "pos")
  int_term <- "stakes_ordinal:user_correct"
  intc <- get_term(models$interaction, int_term, h1_dir = "pos")
 
  # Composition
  comp_inc <- models$composition_inc
  comp_cor <- models$composition_cor
  pct_a <- if ("A" %in% names(comp_inc))
              comp_inc["A"] / sum(comp_inc) else NA
  pct_bprime <- if ("B_prime" %in% names(comp_cor))
                   comp_cor["B_prime"] / sum(comp_cor) else NA
 
  list(
    experiment = "mmlu",
    model = model_label,
    pct_case_A = as.numeric(pct_a),
    pct_case_Bprime = as.numeric(pct_bprime),
    # Incorrect direction
    inc_beta = inc["beta"], inc_se = inc["se"],
    inc_t = inc["stat"], inc_df = inc["df"],
    inc_p_two = inc["p2"], inc_p_one_H1neg = inc["p1"],
    # Correct direction
    cor_beta = cor["beta"], cor_se = cor["se"],
    cor_t = cor["stat"], cor_df = cor["df"],
    cor_p_two = cor["p2"], cor_p_one_H1pos = cor["p1"],
    # Interaction
    int_beta = intc["beta"], int_se = intc["se"],
    int_z = intc["stat"],
    int_p_two = intc["p2"], int_p_one_H1pos = intc["p1"],
    # BH-FDR adjusted (from saved p_adj)
    bh_inc = if (!is.null(models$p_adj)) models$p_adj["incorrect"] else NA,
    bh_cor = if (!is.null(models$p_adj)) models$p_adj["correct"] else NA,
    bh_int = if (!is.null(models$p_adj)) models$p_adj["interaction"] else NA
  )
}
 
  
# =============================================================================
# Build summary tables
# =============================================================================
labels <- c("gemini", "openai", "anthropic")
 
# Essay
essay_rows <- list()
for (lbl in labels) {
  row <- build_essay_row(sprintf("essay_fitted_models_%s.rds", lbl), lbl)
  if (!is.null(row)) essay_rows[[lbl]] <- row
}
if (length(essay_rows) > 0) {
  essay_df <- do.call(rbind, lapply(essay_rows, as.data.frame))
  rownames(essay_df) <- NULL
  write.csv(essay_df, file.path(out_dir, "paper_metrics_essay.csv"),
            row.names = FALSE)
  cat("=== ESSAY METRICS ===\n")
  print(round(essay_df[, sapply(essay_df, is.numeric)], 4))
  cat("\n")
}
 
# MMLU
mmlu_rows <- list()
for (lbl in labels) {
  row <- build_mmlu_row(sprintf("mmlu_fitted_models_%s.rds", lbl), lbl)
  if (!is.null(row)) mmlu_rows[[lbl]] <- row
}
if (length(mmlu_rows) > 0) {
  mmlu_df <- do.call(rbind, lapply(mmlu_rows, as.data.frame))
  rownames(mmlu_df) <- NULL
  write.csv(mmlu_df, file.path(out_dir, "paper_metrics_mmlu.csv"),
            row.names = FALSE)
  cat("=== MMLU METRICS ===\n")
  print(round(mmlu_df[, sapply(mmlu_df, is.numeric)], 4))
  cat("\n")
}

# =============================================================================
# Cross-experiment direction summary
# =============================================================================
direction_label <- function(beta, p1, threshold = 0.05) {
  if (is.na(beta)) return("NA")
  if (p1 < threshold) {
    if (beta < 0) "neg sig" else "pos sig"
  } else {
    if (beta < -0.005) "neg ns" else if (beta > 0.005) "pos ns" else "~0"
  }
}
 
if (length(essay_rows) > 0 && length(mmlu_rows) > 0) {
  cross <- data.frame(
    model = labels,
    essay_direction = sapply(labels, function(l) {
      r <- essay_rows[[l]]
      if (is.null(r)) "NA" else direction_label(r$beta_stakes, r$p_one_sided_H1neg)
    }),
    mmlu_incorrect_direction = sapply(labels, function(l) {
      r <- mmlu_rows[[l]]
      if (is.null(r)) "NA" else direction_label(r$inc_beta, r$inc_p_one_H1neg)
    }),
    mmlu_correct_direction = sapply(labels, function(l) {
      r <- mmlu_rows[[l]]
      if (is.null(r)) "NA" else {
        # H1 for correct is positive — flip sign for direction_label
        direction_label(-r$cor_beta, r$cor_p_one_H1pos)
      }
    }),
    mmlu_interaction_direction = sapply(labels, function(l) {
      r <- mmlu_rows[[l]]
      if (is.null(r)) "NA" else {
        direction_label(-r$int_beta, r$int_p_one_H1pos)
      }
    })
  )
  write.csv(cross, file.path(out_dir, "paper_metrics_cross.csv"),
            row.names = FALSE)
  cat("=== CROSS-EXPERIMENT DIRECTION SUMMARY ===\n")
  print(cross)
  cat("\nDirection labels: 'neg sig' = H1 supported (one-sided p<0.05 in H1 direction)\n")
  cat("                  'pos sig' = OPPOSITE of H1 direction, significant\n")
  cat("                  'neg ns' / 'pos ns' = direction but not significant\n")
  cat("                  '~0' = no effect\n")
}
 
cat(sprintf("\nWrote summary CSVs to %s\n", out_dir))
 