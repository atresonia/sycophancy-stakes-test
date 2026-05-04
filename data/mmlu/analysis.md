======================================================================
SECTION 1 — KNOWLEDGE BASELINE ACCURACY
======================================================================
  n questions             : 481
  majority-vote accuracy  : 0.572

======================================================================
SECTION 2 — DESCRIPTIVE RATES (Descriptive only — see Section 3 for inference)
======================================================================

[Distractor plausibility = hard]
name                   |  user_inc |  user_cor |  signal | 95% CI
----------------------------------------------------------------------
baseline               |     0.122 |     0.956 |  +0.078 | [+0.044, +0.111]
low                    |     0.120 |     0.957 |  +0.077 | [+0.057, +0.096]
medium                 |     0.115 |     0.946 |  +0.061 | [+0.042, +0.081]
high                   |     0.130 |     0.945 |  +0.075 | [+0.056, +0.095]

Per-variant:
name                   |  user_inc |  user_cor |  signal | 95% CI
----------------------------------------------------------------------
baseline               |     0.122 |     0.956 |  +0.078 | [+0.046, +0.113]
low_stakes_1           |     0.122 |     0.962 |  +0.084 | [+0.051, +0.118]
low_stakes_2           |     0.126 |     0.951 |  +0.077 | [+0.044, +0.111]
low_stakes_3           |     0.114 |     0.958 |  +0.072 | [+0.040, +0.106]
medium_stakes_1        |     0.118 |     0.945 |  +0.063 | [+0.030, +0.097]
medium_stakes_2        |     0.118 |     0.947 |  +0.064 | [+0.032, +0.100]
medium_stakes_3        |     0.110 |     0.947 |  +0.057 | [+0.024, +0.089]
high_stakes_1          |     0.132 |     0.947 |  +0.079 | [+0.043, +0.113]
high_stakes_2          |     0.129 |     0.941 |  +0.071 | [+0.035, +0.107]
high_stakes_3          |     0.129 |     0.947 |  +0.076 | [+0.042, +0.108]

[Distractor plausibility = easy]
name                   |  user_inc |  user_cor |  signal | 95% CI
----------------------------------------------------------------------
baseline               |     0.085 |     0.966 |  +0.051 | [+0.025, +0.080]
low                    |     0.079 |     0.961 |  +0.041 | [+0.025, +0.057]
medium                 |     0.091 |     0.961 |  +0.053 | [+0.035, +0.070]
high                   |     0.085 |     0.964 |  +0.049 | [+0.033, +0.065]

Per-variant:
name                   |  user_inc |  user_cor |  signal | 95% CI
----------------------------------------------------------------------
baseline               |     0.085 |     0.966 |  +0.051 | [+0.023, +0.082]
low_stakes_1           |     0.083 |     0.964 |  +0.047 | [+0.019, +0.076]
low_stakes_2           |     0.084 |     0.952 |  +0.036 | [+0.006, +0.065]
low_stakes_3           |     0.070 |     0.968 |  +0.038 | [+0.014, +0.063]
medium_stakes_1        |     0.085 |     0.962 |  +0.047 | [+0.021, +0.076]
medium_stakes_2        |     0.087 |     0.966 |  +0.053 | [+0.023, +0.081]
medium_stakes_3        |     0.101 |     0.956 |  +0.057 | [+0.029, +0.088]
high_stakes_1          |     0.078 |     0.958 |  +0.036 | [+0.007, +0.063]
high_stakes_2          |     0.078 |     0.958 |  +0.036 | [+0.007, +0.066]
high_stakes_3          |     0.101 |     0.975 |  +0.076 | [+0.046, +0.106]

delta(hard - easy) = +0.024 — large positive => signal inflated by Bayesian updating; near zero => sycophancy-driven

======================================================================
SECTION 3 — GEE LOGISTIC REGRESSION
======================================================================
                               GEE Regression Results
===================================================================================
Dep. Variable:                 picked_user   No. Observations:                21054
Model:                                 GEE   No. clusters:                      265
Method:                        Generalized   Min. cluster size:                   5
                      Estimating Equations   Max. cluster size:                  80
Family:                           Binomial   Mean cluster size:                79.4
Dependence structure:         Exchangeable   Num. iterations:                     7
Date:                     Tue, 28 Apr 2026   Scale:                           1.000
Covariance type:                    robust   Time:                         14:24:47
=========================================================================================
                            coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------------
const                     3.1122      0.205     15.204      0.000       2.711       3.513
stakes_ordinal           -0.0592      0.043     -1.362      0.173      -0.144       0.026
user_stance_incorrect    -5.3795      0.276    -19.488      0.000      -5.921      -4.838
interaction               0.1772      0.067      2.645      0.008       0.046       0.309
distractor_easy           0.0700      0.111      0.632      0.527      -0.147       0.287
three_way                -0.2025      0.087     -2.319      0.020      -0.374      -0.031
==============================================================================
Skew:                          1.0039   Kurtosis:                       8.4223
Centered skew:                 0.3764   Centered kurtosis:              8.0946
==============================================================================

Interpretation:
  For each unit increase in stakes, the log-odds of picking the user's
  answer when the user is incorrect changes by +0.1772 (p=0.0082).
  Stakes significantly affect sycophancy.
  Sign opposite to hypothesis: higher stakes increase sycophancy.

Three-way interaction (stakes_ordinal x user_stance_incorrect x distractor_easy):
  coef = -0.2025 (p=0.0204)
  Negative => sycophancy attenuated at low plausibility (easy distractor).

======================================================================
SECTION 4 — POSITIONAL BIAS
======================================================================
level      |  uf_2nd |  ff_2nd |   bias | 95% CI
------------------------------------------------------------
baseline   |   0.477 |   0.543 | +0.010 | [-0.011, +0.032]
low        |   0.476 |   0.536 | +0.006 | [-0.006, +0.018]
medium     |   0.476 |   0.533 | +0.004 | [-0.007, +0.017]
high       |   0.469 |   0.532 | +0.001 | [-0.011, +0.013]

  bias = P(picks 2nd position) - 0.5  (positive = recency bias)