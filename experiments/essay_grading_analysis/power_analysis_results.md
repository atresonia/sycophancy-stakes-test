# ANTHROPIC
(sycophancy-stakes) ➜  essay_grading_analysis git:(feature/clean-up-code-rerun-essay) ✗ Rscript power_analysis.r anthropic_fitted_models.rds
Loading fitted models from: anthropic_fitted_models.rds
Model label: model
N_SIMS per cell: 100

Pilot M2 parameters:
  Intercept     = 1.6840
  beta_observed = -0.02245
  SD(essay)     = 0.7456
  SD(variant)   = 0.0144
  SD(residual)  = 0.1653

Detected pilot direction: negative
One-sided test direction: alternative = 'less'

======================================================================
Monte Carlo power analysis (M2)
======================================================================
Target power threshold: 0.80 (Cohen 1988)
One-sided alpha = 0.05, predicted direction: negative beta

Beta                            N=95     N=200    N=300    N=500
--------------------------------------------------------------------
Pilot observed (-0.0224)        1.00 (100) 1.00 (100) 1.00 (100) 1.00 (100)
Half pilot (-0.0112)            0.90 (100) 1.00 (100) 1.00 (100) 1.00 (100)
SESOI (-0.050)                  1.00 (100) 1.00 (100) 1.00 (100) 1.00 (100)
SESOI (-0.100)                  1.00 (100) 1.00 (100) 1.00 (100) 1.00 (100)

# OPENAI
(sycophancy-stakes) ➜  essay_grading_analysis git:(feature/clean-up-code-rerun-essay) ✗ Rscript power_analysis.r openai_fitted_models.rds
Loading fitted models from: openai_fitted_models.rds
Model label: model
N_SIMS per cell: 100

Pilot M2 parameters:
  Intercept     = 1.4647
  beta_observed = 0.11737
  SD(essay)     = 0.6861
  SD(variant)   = 0.0567
  SD(residual)  = 0.3009

Detected pilot direction: positive
One-sided test direction: alternative = 'greater'

======================================================================
Monte Carlo power analysis (M2)
======================================================================
Target power threshold: 0.80 (Cohen 1988)
One-sided alpha = 0.05, predicted direction: negative beta

Beta                            N=95     N=200    N=300    N=500
--------------------------------------------------------------------
Pilot observed (+0.1174)        1.00 (100) 1.00 (100) 1.00 (100) 1.00 (100)
Half pilot (+0.0587)            1.00 (100) 1.00 (100) 1.00 (100) 1.00 (100)
SESOI (+0.050)                  1.00 (100) 1.00 (100) 1.00 (100) 1.00 (100)
SESOI (+0.100)                  1.00 (100) 1.00 (100) 1.00 (100) 1.00 (100)

# GEMINI
(sycophancy-stakes) ➜  essay_grading_analysis git:(feature/clean-up-code-rerun-essay) ✗ Rscript power_analysis.r gemini_fitted_models.rds
Loading fitted models from: gemini_fitted_models.rds
Model label: model
N_SIMS per cell: 100

Pilot M2 parameters:
  Intercept     = 1.9311
  beta_observed = -0.01632
  SD(essay)     = 0.9608
  SD(variant)   = 0.0528
  SD(residual)  = 0.3577

Detected pilot direction: negative
One-sided test direction: alternative = 'less'

======================================================================
Monte Carlo power analysis (M2)
======================================================================
Target power threshold: 0.80 (Cohen 1988)
One-sided alpha = 0.05, predicted direction: negative beta

Beta                            N=95     N=200    N=300    N=500
--------------------------------------------------------------------
Pilot observed (-0.0163)        0.63 (100) 0.89 (100) 0.97 (100) 1.00 (100)
Half pilot (-0.0082)            0.22 (100) 0.45 (100) 0.50 (100) 0.71 (100)
SESOI (-0.050)                  1.00 (100) 1.00 (100) 1.00 (100) 1.00 (100)
SESOI (-0.100)                  1.00 (100) 1.00 (100) 1.00 (100) 1.00 (100)