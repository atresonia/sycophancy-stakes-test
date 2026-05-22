suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)   # provides p-values via Satterthwaite df by default
  library(clinfun)    # Jonckheere-Terpstra test
})

# allow args to select which long_df file to use and which model file to save to
base_dir <- "/Users/soniaatre/Cornell2025/Research/sycophancy-stakes-test/data/essay_grading/stakes-variants/"
args <- commandArgs(trailingOnly = TRUE)
long_df_file <- if (length(args) >= 1) args[1] else "gemini_long_per_variant_df.csv"
long_df <- read.csv(paste0(base_dir, long_df_file))
cat(sprintf("Loaded long_df from: %s\n", paste0(base_dir, long_df_file)))
model_file <- if (length(args) >= 2) args[2] else "openai_fitted_models.rds"
cat(sprintf("Saving models to: %s\n", model_file))

long_df$stakes_level <- factor(long_df$stakes_level, levels=c("low","medium","high"))
long_df$variant_idx <- factor(long_df$variant_idx)

long_df$essay_id <- factor(long_df$essay_id)
long_df$variant_idx <- factor(long_df$variant_idx)

m2_linear <- lmer(grade_num ~ stakes_ordinal +
                  (1|essay_id) + (1|variant_idx),
                  data = long_df, REML = FALSE)

m2_quad <- lmer(grade_num ~ stakes_ordinal + I(stakes_ordinal^2) +
                (1|essay_id) + (1|variant_idx),
                data = long_df, REML = FALSE)

anova(m2_linear, m2_quad)
summary(m2_quad)$coefficients

cat("Per-variant long form:", nrow(long_df), "rows\n")

m1 <- lmer(grade_num ~ stakes_ordinal + (1 | essay_id),
           data = long_df, REML  = FALSE)
 
m2 <- lmer(grade_num ~ stakes_ordinal + (1 | essay_id) + (1 | variant_idx),
           data = long_df, REML = FALSE)
 
m3 <- lmer(grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id),
           data = long_df, REML = FALSE)
 
m4 <- lmer(grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id) +
                       (1 | variant_idx),
           data = long_df, REML = FALSE)

extract_stakes <- function(model, label) {
  coefs <- summary(model)$coefficients
  beta <- coefs["stakes_ordinal", "Estimate"]
  se   <- coefs["stakes_ordinal", "Std. Error"]
  # lmerTest provides t and df via Satterthwaite
  t    <- coefs["stakes_ordinal", "t value"]
  df   <- coefs["stakes_ordinal", "df"]
  p_two <- coefs["stakes_ordinal", "Pr(>|t|)"]
  p_one <- if (t < 0) p_two / 2 else 1 - p_two / 2
  conv <- isSingular(model)  # TRUE = singular fit (problem)
  cat(sprintf("%-50s beta=%+.5f  SE=%.5f  t=%+.3f  df=%.1f  p2=%.4f  p1=%.4f%s\n",
              label, beta, se, t, df, p_two, p_one,
              if (conv) "  [SINGULAR]" else ""))
  invisible(model)
}

cat("\n--- Stakes coefficient (with Satterthwaite df via lmerTest) ---\n")
extract_stakes(m1, "M1: (1|essay)")
extract_stakes(m2, "M2: (1|essay) + (1|variant)")
extract_stakes(m3, "M3: (1+stakes|essay)")
extract_stakes(m4, "M4 [maximal]: (1+stakes|essay) + (1|variant)")

cat("\n--- Variance components for M4 (maximal) ---\n")
print(VarCorr(m4))
 
cat("\n--- Variance components for M2 (parsimonious) ---\n")
print(VarCorr(m2))

 
cat("\nLR test 1: M2 vs M1 (does (1|variant) improve fit beyond (1|essay)?)\n")
print(anova(m1, m2))
 
cat("\nLR test 2: M3 vs M1 (does random stakes slope improve fit?)\n")
print(anova(m1, m3))
 
cat("\nLR test 3: M4 vs M2 (does random stakes slope improve fit on top of M2?)\n")
print(anova(m2, m4))
 
cat("\nLR test 4: M4 vs M3 (does (1|variant) help on top of random slope?)\n")
print(anova(m3, m4))

saveRDS(list(m1=m1, m2=m2, m3=m3, m4=m4), file=model_file)
cat(sprintf("\nSaved fitted models to %s\n", model_file))