
| Metric                                      | Estimate   | 95% CI           | p-value   |
| ------------------------------------------- | ---------- | ---------------- | --------- |
| KB accuracy                                 | **66.90%** | [64.84%, 68.96%] | 1.15e-54  |
| Sycophantic accuracy                        | **56.85%** | [54.68%, 59.02%] | 7.58e-10  |
| Flip rate toward user answer                | **22.90%** | [21.06%, 24.74%] | 3.49e-153 |
| Flip rate away from both KB and user answer | **12.55%** | [11.10%, 14.00%] | 3.09e-60  |
| No-change rate                              | **67.95%** | [65.90%, 70.00%] | 6.34e-62  |



| Metric                                      | Low stakes                                | Medium stakes                             | High stakes                               |
| ------------------------------------------- | ----------------------------------------- | ----------------------------------------- | ----------------------------------------- |
| Sycophantic accuracy                        | **61.10%** [58.96%, 63.24%] p = 9.18e-24  | **61.75%** [59.62%, 63.88%] p = 1.65e-26  | **61.00%** [58.86%, 63.14%] p = 2.34e-23  |
| Flip rate toward user answer                | **14.65%** [13.10%, 16.20%] p = 4.50e-303 | **13.80%** [12.29%, 15.31%] p = 0.00      | **14.15%** [12.62%, 15.68%] p = 0.00      |
| Flip rate away from both KB and user answer | **16.70%** [15.06%, 18.34%] p = 2.11e-81  | **16.25%** [14.63%, 17.87%] p = 4.66e-79  | **16.65%** [15.02%, 18.28%] p = 3.84e-81  |
| No-change rate                              | **78.10%** [76.29%, 79.91%] p = 5.48e-167 | **78.70%** [76.90%, 80.50%] p = 9.85e-176 | **78.45%** [76.65%, 80.25%] p = 4.80e-172 |


### Diagnostic: which (if any) of these renders?

**A. Sibling, relative:**
![A](test-image.png)

**B. Subfolder, relative no-dot:**
![B](image/analysis/1778264426027.png)

**C. Subfolder, dot-relative:**
![C](./image/analysis/1778264426027.png)

**D. Absolute file URI:**
![D](file:///Users/soniaatre/Cornell2025/Research/sycophancy-stakes-test/scripts/mmlu_analysis/image/analysis/1778264426027.png)

**E. HTML img sibling:**
<img src="test-image.png" alt="E" />