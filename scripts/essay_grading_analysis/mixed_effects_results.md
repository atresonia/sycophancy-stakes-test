GEMINI:
(sycophancy-stakes) ➜  essay_grading_analysis git:(feature/clean-up-code-rerun-essay) ✗ Rscript mixed_effects_test.r
Per-variant long form: 2850 rows
boundary (singular) fit: see help('isSingular')
Warning message:
In checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv,  :
  Model failed to converge with max|grad| = 0.00322878 (tol = 0.002, component 1)
  See ?lme4::convergence and ?lme4::troubleshooting.

--- Stakes coefficient (with Satterthwaite df via lmerTest) ---
M1: (1|essay)                                      beta=-0.01632  SE=0.00829  t=-1.968  df=2755.0  p2=0.0492  p1=0.0246
M2: (1|essay) + (1|variant)                        beta=-0.01632  SE=0.00821  t=-1.988  df=2746.0  p2=0.0469  p1=0.0234
M3: (1+stakes|essay)                               beta=-0.01632  SE=0.00837  t=-1.950  df=1348.6  p2=0.0513  p1=0.0257  [SINGULAR]
M4 [maximal]: (1+stakes|essay) + (1|variant)       beta=-0.01632  SE=0.00834  t=-1.957  df=96.0  p2=0.0533  p1=0.0266

--- Variance components for M4 (maximal) ---
 Groups      Name           Std.Dev. Corr
 essay_id    (Intercept)    0.949712
             stakes_ordinal 0.014611 0.757
 variant_idx (Intercept)    0.052836
 Residual                   0.357514

--- Variance components for M2 (parsimonious) ---
 Groups      Name        Std.Dev.
 essay_id    (Intercept) 0.960751
 variant_idx (Intercept) 0.052818
 Residual                0.357721

LR test 1: M2 vs M1 (does (1|variant) improve fit beyond (1|essay)?)
Data: long_df
Models:
m1: grade_num ~ stakes_ordinal + (1 | essay_id)
m2: grade_num ~ stakes_ordinal + (1 | essay_id) + (1 | variant_idx)
   npar    AIC    BIC  logLik -2*log(L)  Chisq Df Pr(>Chisq)
m1    4 2803.2 2827.0 -1397.6    2795.2
m2    5 2767.4 2797.2 -1378.7    2757.4 37.744  1  8.066e-10 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

LR test 2: M3 vs M1 (does random stakes slope improve fit?)
Data: long_df
Models:
m1: grade_num ~ stakes_ordinal + (1 | essay_id)
m3: grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id)
   npar    AIC    BIC  logLik -2*log(L)  Chisq Df Pr(>Chisq)
m1    4 2803.2 2827.0 -1397.6    2795.2
m3    6 2805.4 2841.1 -1396.7    2793.4 1.8024  2     0.4061

LR test 3: M4 vs M2 (does random stakes slope improve fit on top of M2?)
Data: long_df
Models:
m2: grade_num ~ stakes_ordinal + (1 | essay_id) + (1 | variant_idx)
m4: grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id) + (1 | variant_idx)
   npar    AIC    BIC  logLik -2*log(L)  Chisq Df Pr(>Chisq)
m2    5 2767.4 2797.2 -1378.7    2757.4
m4    7 2769.6 2811.3 -1377.8    2755.6 1.8472  2     0.3971

LR test 4: M4 vs M3 (does (1|variant) help on top of random slope?)
Data: long_df
Models:
m3: grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id)
m4: grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id) + (1 | variant_idx)
   npar    AIC    BIC  logLik -2*log(L)  Chisq Df Pr(>Chisq)
m3    6 2805.4 2841.1 -1396.7    2793.4
m4    7 2769.6 2811.3 -1377.8    2755.6 37.789  1  7.883e-10 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Saved fitted models to fitted_models.rds


OPENAI:

(sycophancy-stakes) ➜  essay_grading_analysis git:(feature/clean-up-code-rerun-essay) ✗ Rscript mixed_effects_test.r
Data: long_df
Models:
m2_linear: grade_num ~ stakes_ordinal + (1 | essay_id) + (1 | variant_idx)
m2_quad: grade_num ~ stakes_ordinal + I(stakes_ordinal^2) + (1 | essay_id) + (1 | variant_idx)
          npar  AIC    BIC  logLik -2*log(L) Chisq Df Pr(>Chisq)
m2_linear    5 1754 1783.8 -872.02      1744
m2_quad      6 1737 1772.7 -862.49      1725 19.06  1  1.267e-05 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
                       Estimate Std. Error        df   t value     Pr(>|t|)
(Intercept)          1.44736842 0.07329708  104.6574 19.746605 4.439495e-37
stakes_ordinal       0.22157895 0.02480143 2745.9898  8.934121 7.354425e-19
I(stakes_ordinal^2) -0.05210526 0.01191422 2745.9898 -4.373367 1.268999e-05
Per-variant long form: 2850 rows

--- Stakes coefficient (with Satterthwaite df via lmerTest) ---
M1: (1|essay)                                      beta=+0.11737  SE=0.00702  t=+16.727  df=2755.0  p2=0.0000  p1=1.0000
M2: (1|essay) + (1|variant)                        beta=+0.11737  SE=0.00690  t=+17.004  df=2746.0  p2=0.0000  p1=1.0000
M3: (1+stakes|essay)                               beta=+0.11737  SE=0.01118  t=+10.494  df=95.0  p2=0.0000  p1=1.0000
M4 [maximal]: (1+stakes|essay) + (1|variant)       beta=+0.11737  SE=0.01118  t=+10.494  df=95.0  p2=0.0000  p1=1.0000

--- Variance components for M4 (maximal) ---
 Groups      Name           Std.Dev. Corr
 essay_id    (Intercept)    0.678112
             stakes_ordinal 0.087298 0.029
 variant_idx (Intercept)    0.056895
 Residual                   0.291984

--- Variance components for M2 (parsimonious) ---
 Groups      Name        Std.Dev.
 essay_id    (Intercept) 0.686115
 variant_idx (Intercept) 0.056742
 Residual                0.300877

LR test 1: M2 vs M1 (does (1|variant) improve fit beyond (1|essay)?)
Data: long_df
Models:
m1: grade_num ~ stakes_ordinal + (1 | essay_id)
m2: grade_num ~ stakes_ordinal + (1 | essay_id) + (1 | variant_idx)
   npar    AIC    BIC  logLik -2*log(L)  Chisq Df Pr(>Chisq)
m1    4 1820.7 1844.5 -906.34    1812.7
m2    5 1754.0 1783.8 -872.02    1744.0 68.642  1  < 2.2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

LR test 2: M3 vs M1 (does random stakes slope improve fit?)
Data: long_df
Models:
m1: grade_num ~ stakes_ordinal + (1 | essay_id)
m3: grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id)
   npar    AIC    BIC  logLik -2*log(L)  Chisq Df Pr(>Chisq)
m1    4 1820.7 1844.5 -906.34    1812.7
m3    6 1761.3 1797.0 -874.63    1749.3 63.433  2  1.681e-14 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

LR test 3: M4 vs M2 (does random stakes slope improve fit on top of M2?)
Data: long_df
Models:
m2: grade_num ~ stakes_ordinal + (1 | essay_id) + (1 | variant_idx)
m4: grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id) + (1 | variant_idx)
   npar    AIC    BIC  logLik -2*log(L)  Chisq Df Pr(>Chisq)
m2    5 1754.0 1783.8 -872.02    1744.0
m4    7 1689.2 1730.9 -837.59    1675.2 68.868  2  1.111e-15 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

LR test 4: M4 vs M3 (does (1|variant) help on top of random slope?)
Data: long_df
Models:
m3: grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id)
m4: grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id) + (1 | variant_idx)
   npar    AIC    BIC  logLik -2*log(L)  Chisq Df Pr(>Chisq)
m3    6 1761.3 1797.0 -874.63    1749.3
m4    7 1689.2 1730.9 -837.59    1675.2 74.077  1  < 2.2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Saved fitted models to openai_fitted_models.rds


ANTHROPIC:
(sycophancy-stakes) ➜  essay_grading_analysis git:(feature/clean-up-code-rerun-essay) ✗ Rscript mixed_effects_test.r anthropic_long_per_variant_df.csv anthropic_fitted_models.rds
Loaded long_df from: /Users/soniaatre/Cornell2025/Research/sycophancy-stakes-test/data/essay_grading/stakes-variants/anthropic_long_per_variant_df.csv
Saving models to: anthropic_fitted_models.rds
Data: long_df
Models:
m2_linear: grade_num ~ stakes_ordinal + (1 | essay_id) + (1 | variant_idx)
m2_quad: grade_num ~ stakes_ordinal + I(stakes_ordinal^2) + (1 | essay_id) + (1 | variant_idx)
          npar     AIC     BIC logLik -2*log(L)  Chisq Df Pr(>Chisq)
m2_linear    5 -1589.8 -1559.9 799.93   -1599.8
m2_quad      6 -1595.0 -1559.1 803.53   -1607.0 7.2006  1   0.007288 **
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
                       Estimate  Std. Error         df   t value     Pr(>|t|)
(Intercept)          1.68979592 0.075639289   99.33423 22.340188 1.603945e-40
stakes_ordinal      -0.05714286 0.013448494 2832.99607 -4.249015 2.216403e-05
I(stakes_ordinal^2)  0.01734694 0.006460447 2832.99607  2.685099 7.293097e-03
Per-variant long form: 2940 rows

--- Stakes coefficient (with Satterthwaite df via lmerTest) ---
M1: (1|essay)                                      beta=-0.02245  SE=0.00375  t=-5.990  df=2842.0  p2=0.0000  p1=0.0000
M2: (1|essay) + (1|variant)                        beta=-0.02245  SE=0.00373  t=-6.011  df=2833.0  p2=0.0000  p1=0.0000
M3: (1+stakes|essay)                               beta=-0.02245  SE=0.00662  t=-3.392  df=98.0  p2=0.0010  p1=0.0005
M4 [maximal]: (1+stakes|essay) + (1|variant)       beta=-0.02245  SE=0.00662  t=-3.392  df=98.0  p2=0.0010  p1=0.0005

--- Variance components for M4 (maximal) ---
 Groups      Name           Std.Dev. Corr
 essay_id    (Intercept)    0.760664
             stakes_ordinal 0.055044 -0.306
 variant_idx (Intercept)    0.014675
 Residual                   0.158876

--- Variance components for M2 (parsimonious) ---
 Groups      Name        Std.Dev.
 essay_id    (Intercept) 0.745600
 variant_idx (Intercept) 0.014432
 Residual                0.165341

LR test 1: M2 vs M1 (does (1|variant) improve fit beyond (1|essay)?)
Data: long_df
Models:
m1: grade_num ~ stakes_ordinal + (1 | essay_id)
m2: grade_num ~ stakes_ordinal + (1 | essay_id) + (1 | variant_idx)
   npar     AIC     BIC logLik -2*log(L)  Chisq Df Pr(>Chisq)
m1    4 -1582.3 -1558.4 795.17   -1590.3
m2    5 -1589.8 -1559.9 799.93   -1599.8 9.5163  1   0.002037 **
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

LR test 2: M3 vs M1 (does random stakes slope improve fit?)
Data: long_df
Models:
m1: grade_num ~ stakes_ordinal + (1 | essay_id)
m3: grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id)
   npar     AIC     BIC logLik -2*log(L)  Chisq Df Pr(>Chisq)
m1    4 -1582.3 -1558.4 795.17   -1590.3
m3    6 -1686.7 -1650.8 849.36   -1698.7 108.39  2  < 2.2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

LR test 3: M4 vs M2 (does random stakes slope improve fit on top of M2?)
Data: long_df
Models:
m2: grade_num ~ stakes_ordinal + (1 | essay_id) + (1 | variant_idx)
m4: grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id) + (1 | variant_idx)
   npar     AIC     BIC logLik -2*log(L)  Chisq Df Pr(>Chisq)
m2    5 -1589.8 -1559.9 799.93   -1599.8
m4    7 -1695.9 -1654.0 854.96   -1709.9 110.07  2  < 2.2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

LR test 4: M4 vs M3 (does (1|variant) help on top of random slope?)
Data: long_df
Models:
m3: grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id)
m4: grade_num ~ stakes_ordinal + (1 + stakes_ordinal | essay_id) + (1 | variant_idx)
   npar     AIC     BIC logLik -2*log(L) Chisq Df Pr(>Chisq)
m3    6 -1686.7 -1650.8 849.36   -1698.7
m4    7 -1695.9 -1654.0 854.96   -1709.9  11.2  1  0.0008181 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Saved fitted models to anthropic_fitted_models.rds