# Mixed effects fitting for MMLU stakes experiment

# Load the necessary libraries
suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else "."
out_dir <- if (length(args) >= 2) args[2] else "."
model_label <- if (length(args) >= 3) args[3] else "model"

extract_coef <- function(model, label, term) {
    c <- summary(model)$coefficients
    if (!term %in% rownames(c)) return (invisible(NULL))
    if ("z value" %in% colnames(c)) {
        cat(sprintf("%-50s beta=%+.5f SE=%.5f z=%+.3f p2=%.4f\n",
                label, c[term,"Estimate"], c[term,"Std. Error"],
                c[term,"z value"], c[term,"Pr(>|z|)"]))
    } else {
        cat(sprintf("%-50s beta=%+.5f SE=%.5f t=%+.3f df=%.1f p2=%.4f\n",
                label, c[term,"Estimate"], c[term,"Std. Error"],
                c[term,"t value"], c[term,"df"], c[term,"Pr(>|t|)"]))
    }
}

# load data
long_inc <- read.csv(file.path(data_dir, "mmlu_long_incorrect.csv"))
long_cor <- read.csv(file.path(data_dir, "mmlu_long_correct.csv"))
agg_inc <- read.csv(file.path(data_dir, "mmlu_agg_incorrect.csv"))
agg_cor <- read.csv(file.path(data_dir, "mmlu_agg_correct.csv"))

long_inc$question_id <- factor(long_inc$question_id)
long_inc$variant_idx <- factor(long_inc$variant_idx)
long_cor$question_id <- factor(long_cor$question_id)
long_cor$variant_idx <- factor(long_cor$variant_idx)
agg_inc$question_id <- factor(agg_inc$question_id)
agg_cor$question_id <- factor(agg_cor$question_id)

# PART 0: KB-alignment composition
cat("===== PART 0: KB-alignment composition =====\n")
q_inc <- unique(long_inc[, c("question_id","kb_alignment")])
q_cor <- unique(long_cor[, c("question_id","kb_alignment")])
cat("INCORRECT direction:\n"); print(table(q_inc$kb_alignment))
cat("CORRECT direction:\n"); print(table(q_cor$kb_alignment))
cat("\n")

cat("===== PART 1: User-INCORRECT (H1: beta < 0) =====\n")
m_primary_inc <- lmer(FR_cond ~ stakes_ordinal + (1 | question_id), data = agg_inc, REML = FALSE)
extract_coef(m_primary_inc, "PRIMARY all-cases lmer", "stakes_ordinal")

if ("kb_alignment" %in% colnames(agg_inc) && sum(agg_inc$kb_alignment == "A") > 0) {
    m_inc_a <- lmer(FR_cond ~ stakes_ordinal + (1|question_id),
                  data = agg_inc[agg_inc$kb_alignment == "A",], REML = FALSE)
    extract_coef(m_inc_a, "KB-ALIGNMENT A", "stakes_ordinal")
}

m_max_inc <- glmer(flipped_to_user ~ stakes_ordinal +
                   (1+stakes_ordinal|question_id) + (1|variant_idx),
                   data = long_inc, family = binomial,
                   control = glmerControl(optimizer="bobyqa",
                                          optCtrl=list(maxfun=1e5)))
m_simple_inc <- glmer(flipped_to_user ~ stakes_ordinal +
                      (1|question_id) + (1|variant_idx),
                      data = long_inc, family = binomial,
                      control = glmerControl(optimizer="bobyqa",
                                             optCtrl=list(maxfun=1e5)))
extract_coef(m_simple_inc, "SECONDARY glmer simple", "stakes_ordinal")
extract_coef(m_max_inc,    "SECONDARY glmer maximal", "stakes_ordinal")
lr_inc <- anova(m_simple_inc, m_max_inc)
cat(sprintf("LR test random slope: p=%.4f\n\n", lr_inc$`Pr(>Chisq)`[2]))

# PART 2: User-CORRECT analysis
cat("===== PART 2: User-CORRECT (H1: beta > 0) =====\n")
m_primary_cor <- lmer(conf_shift ~ stakes_ordinal + (1 | question_id), data = agg_cor, REML = FALSE)
extract_coef(m_primary_cor, "PRIMARY all-cases lmer", "stakes_ordinal")

if ("kb_alignment" %in% colnames(agg_cor) && sum(agg_cor$kb_alignment == "B_prime") > 10) {
    m_cor_b_prime <- lmer(conf_shift ~ stakes_ordinal + (1|question_id),
                          data = agg_cor[agg_cor$kb_alignment == "B_prime",], REML = FALSE)
    extract_coef(m_cor_b_prime, "KB-ALIGNMENT B_prime (KB wrong, user correct)", "stakes_ordinal")
}

m_max_cor <- glmer(probe_matches_correct ~ stakes_ordinal +
                   (1+stakes_ordinal|question_id) + (1|variant_idx),
                   data = long_cor, family = binomial,
                   control = glmerControl(optimizer="bobyqa",
                                          optCtrl=list(maxfun=1e5)))
m_simple_cor <- glmer(probe_matches_correct ~ stakes_ordinal +
                      (1|question_id) + (1|variant_idx),
                      data = long_cor, family = binomial,
                      control = glmerControl(optimizer="bobyqa",
                                             optCtrl=list(maxfun=1e5)))
extract_coef(m_simple_cor, "SECONDARY glmer simple", "stakes_ordinal")
extract_coef(m_max_cor,    "SECONDARY glmer maximal", "stakes_ordinal")
lr_cor <- anova(m_simple_cor, m_max_cor)
cat(sprintf("LR test random slope: p=%.4f\n\n", lr_cor$`Pr(>Chisq)`[2]))

# PART 3: INTERACTION model
cat("===== PART 3: INTERACTION model =====\n")
cat("H1 (anti-sycophancy): stakes:user_correct > 0\n")
cat("  i.e. stakes increases agreement when user correct,\n")
cat("       and decreases agreement when user incorrect\n")
cat("Alt (anti-deference): both main effects negative, interaction near 0\n\n")
 
combined_long <- rbind(
    data.frame(question_id = long_inc$question_id,
               variant_idx = long_inc$variant_idx,
               stakes_ordinal = long_inc$stakes_ordinal,
               matches_user = long_inc$flipped_to_user,
               user_correct = 0L),
    data.frame(question_id = long_cor$question_id,
               variant_idx = long_cor$variant_idx,
               stakes_ordinal = long_cor$stakes_ordinal,
               matches_user = long_cor$probe_matches_correct,
               user_correct = 1L)
)
cat(sprintf("Combined long: %d rows\n", nrow(combined_long)))
m_interaction <- glmer(
    matches_user ~ stakes_ordinal * user_correct +
                    (1 + stakes_ordinal | question_id) + (1 | variant_idx),
    data = combined_long, family = binomial,
    control = glmerControl(optimizer="bobyqa",
                           optCtrl=list(maxfun=1e5))
)

cat("\nInteraction fixed effects:\n")
print(round(summary(m_interaction)$coefficients, 4))
int_coef <- summary(m_interaction)$coefficients["stakes_ordinal:user_correct",]
p_int_1sided <- if (int_coef["z value"] > 0) int_coef["Pr(>|z|)"] / 2 else 1 - int_coef["Pr(>|z|)"] / 2
cat(sprintf("p-value (1-sided): %.4f\n\n", p_int_1sided))

# PART 4: BH-FDR
cat("\n===== PART 4: BH-FDR across primary tests =====\n")
p_inc <- {
  c <- summary(m_primary_inc)$coefficients
  t <- c["stakes_ordinal","t value"]; p2 <- c["stakes_ordinal","Pr(>|t|)"]
  if (t < 0) p2/2 else 1-p2/2
}
p_cor <- {
  c <- summary(m_primary_cor)$coefficients
  t <- c["stakes_ordinal","t value"]; p2 <- c["stakes_ordinal","Pr(>|t|)"]
  if (t > 0) p2/2 else 1-p2/2
}
p_raw <- c(incorrect = p_inc, correct = p_cor, interaction = p_int_1sided)
p_adj <- p.adjust(p_raw, method = "BH")
cat("Raw p-values (one-sided):\n"); print(round(p_raw, 5))
cat("BH-FDR adjusted:\n"); print(round(p_adj, 5))
 
 # Save
saveRDS(list(
  primary_inc = m_primary_inc, primary_cor = m_primary_cor,
  max_inc = m_max_inc, simple_inc = m_simple_inc,
  max_cor = m_max_cor, simple_cor = m_simple_cor,
  interaction = m_interaction,
  p_raw = p_raw, p_adj = p_adj,
  composition_inc = table(q_inc$kb_alignment),
  composition_cor = table(q_cor$kb_alignment)
), file = file.path(out_dir, sprintf("mmlu_fitted_models_%s.rds", model_label)))
 
cat(sprintf("\nSaved to %s\n",
            file.path(out_dir, sprintf("mmlu_fitted_models_%s.rds", model_label))))