# The Role of Stakes in Sycophancy — Results Summary

Ran three experiments — essay grading, MMLU bet-stakes, ELEPHANT moral sycophancy. 
Gemini 2.5 Flash at n = 2000, and Mistral Ministral-3-14B at n = 100 (pilot).

## Part 1 — Gemini 2.5 Flash, n = 2000

### 1.1 Essay grading

Primary metric: grade shift **GI = grade(cond) − grade(kb)**; positive = inflation.

| Condition | GI mean | 95% CI | n | ProgRate | RegRate |
|-----------|--------:|:------------------:|-----:|---------:|--------:|
| User syc. | +0.076 | [+0.053, +0.099] | 1999 | 0.110 | 0.102 |
| Low | −0.720 | [−0.755, −0.685] | 1998 | 0.076 | 0.373 |
| Medium | −0.726 | [−0.761, −0.691] | 1996 | 0.072 | 0.379 |
| High | −0.992 | [−1.031, −0.953] | 1997 | 0.056 | 0.469 |

- **Precondition:** GI(user) = +0.076, **d = 0.148** (p < 0.0001). Baseline
  sycophancy is present but very small.
- **Stakes effect:** **B_stakes = −0.1358**, SE = 0.0065, one-sided p < 0.0001,
  95% CI **[−0.149, −0.123]**.
- **Stakes low->high change:** **−0.272 grade units.**
- **Interpretation — Baseline sycophancy is present but very small. There is a decrease in GI from low to high (low -> medium barely has any change).
  However, looking at the ProgRate and RegRate. ProgRate indicates model shift from KB answer towards ground truth answer after user framing. RegRate indicates model shift from KB answer away from ground truth answer after user framing. RegRate increases with stakes, while ProgRate decreases. In general, models tend to grade deflate and "overcompensate" with higher stakes, leading to a shift away from ground truth answer (which is a higher letter grade than what the LLM outputs after stakes framing).

![Essay grade shift, Gemini n=2000](plot_essay_gemini_2000.png)

### 1.2 MMLU bet-stakes

Primary metric: **flip_to_user** (model adopts the user's incorrect answer).
Knowledge-baseline accuracy = 0.688 (n = 2000).

| Condition | flip_to_user | 95% CI | flip_away | accuracy | n | ctrl flip |
|-----------|-------------:|:--------------:|----------:|---------:|-----:|----------:|
| Low | 0.155 | [0.139, 0.171] | 0.060 | 0.612 | 2000 | 0.179 |
| Medium | 0.154 | [0.138, 0.170] | 0.058 | 0.617 | 2000 | 0.180 |
| High | 0.147 | [0.132, 0.163] | 0.058 | 0.622 | 2000 | 0.174 |

Baselines (descriptive): user-sycophancy flip_to_user = 0.218; bet-no-money = 0.165.

- **Precondition:** flip_to_user = 0.218, **d = 0.528**, p < 0.0001 — baseline
  sycophancy present because flip_to_user is greater than 0.
- **Stakes effect:** not significant. Overlapping CIs across stake levels and low, medium, high flip rates 
  are all very close to each other.
- **B_stakes:** −0.306 (p = 0.017)

![MMLU flip-to-user, Gemini n=2000](plot_mmlu_gemini_2000.png)

### 1.3 ELEPHANT moral sycophancy

Primary metric: **moral_sycophancy** (model outputs NTA on *both* perspectives).

| Condition | moral_syc | 95% CI | n | dropped |
|-----------|----------:|:--------------:|-----:|--------:|
| Baseline | 0.343 | [0.319, 0.366] | 1585 | 6 |
| Low | 0.388 | [0.364, 0.412] | 1578 | 13 |
| Medium | 0.380 | [0.356, 0.404] | 1587 | 4 |
| High | 0.366 | [0.342, 0.390] | 1582 | 9 |

- **Precondition:** moral_sycophancy = 0.343, **d = 0.722**, p < 0.0001 —
  baseline sycophancy present.
- **Stakes effect:** not significant. Negligible change across the three stakes
  levels (0.39 / 0.38 / 0.37). **Stakes low→high change: −0.022.**
- **B_stakes:**−0.308 (p = 0.0003)

![ELEPHANT moral sycophancy, Gemini n=2000](plot_elephant_gemini_2000.png)

---

## Part 2 — Mistral Ministral-3-14B, n = 100 (pilot)

### 2.1 Essay grading — Mistral n = 100

| Condition | GI mean | 95% CI | n | ProgRate | RegRate |
|-----------|--------:|:------------------:|----:|---------:|--------:|
| User syc. | +0.200 | [+0.098, +0.302] | 100 | 0.070 | 0.080 |
| Low | +0.210 | [+0.107, +0.313] | 100 | 0.050 | 0.100 |
| Medium | +0.230 | [+0.125, +0.335] | 100 | 0.060 | 0.120 |
| High | +0.220 | [+0.124, +0.316] | 100 | 0.040 | 0.150 |

- **B_stakes = +0.005**, SE = 0.020, one-sided p = 0.601 — no stakes effect.
- **Diverges from Gemini:** Mistral *inflates* grades (GI ≈ +0.2 throughout);
  Gemini strongly *deflates* (GI → −0.99). Opposite baseline direction; stakes
  move neither.

![Essay grade shift, Mistral n=100](plot_essay_mistral_100.png)

### 2.2 MMLU bet-stakes — Mistral n = 100

| Condition | flip_to_user | 95% CI | flip_away | accuracy | n | ctrl flip |
|-----------|-------------:|:--------------:|----------:|---------:|----:|----------:|
| Low | 0.110 | [0.049, 0.171] | 0.110 | 0.430 | 100 | 0.120 |
| Medium | 0.130 | [0.064, 0.196] | 0.110 | 0.410 | 100 | 0.140 |
| High | 0.130 | [0.064, 0.196] | 0.150 | 0.410 | 100 | 0.150 |

Baselines: user-sycophancy flip_to_user = 0.400; bet-no-money = 0.190.

- **Precondition:** flip_to_user = 0.400, d = 0.812 — baseline sycophancy present.
- **Stakes effect:** not significant (CIs very wide). Low->high change +0.020.

![MMLU flip-to-user, Mistral n=100](plot_mmlu_mistral_100.png)

### 2.3 ELEPHANT moral sycophancy — Mistral n = 100

| Condition | moral_syc | 95% CI | n | dropped |
|-----------|----------:|:--------------:|----:|--------:|
| Baseline | 0.570 | [0.473, 0.667] | 100 | 0 |
| Low | 0.700 | [0.610, 0.790] | 100 | 0 |
| Medium | 0.750 | [0.665, 0.835] | 100 | 0 |
| High | 0.790 | [0.710, 0.870] | 100 | 0 |

- **Precondition:** moral_sycophancy = 0.570, d = 1.146 — strong baseline.
- **Stakes effect runs OPPOSITE to H1.** Rate rises 0.57 → 0.79; the
  (degenerate) fit and the descriptive rates agree on a *positive* slope —
  higher stakes make Mistral *more* morally sycophantic. **H1 (B_stakes < 0)
  is not supported; the effect, if any, is reversed.**

![ELEPHANT moral sycophancy, Mistral n=100](plot_elephant_mistral_100.png)
---

## Part 3 — Cross-experiment summary

Stakes 0→2 implied change in each experiment's primary metric:

| Experiment | Gemini n = 2000 | Mistral n = 100 (pilot) |
|------------|----------------:|------------------------:|
| Essay (grade units) | **−0.272** | +0.010 |
| MMLU (flip rate) | −0.008 | +0.020 |
| ELEPHANT (moral-syc rate) | −0.022 | +0.090 |

For Gemini n = 2000 and Mistral n = 100 (pilot):
1. **Stakes do not generally show sycophantic behavior.** For Gemini n = 2000, two
   of three tasks (MMLU, ELEPHANT) show no meaningful stakes effect.
2. **For gemini, essay grading task shows regressive over-correction** — stakes deflate grades *away* from the true grade.
3. **The effect is model-dependent.** Mistral diverges from Gemini: it inflates
   essays where Gemini deflates, and its ELEPHANT moral sycophancy *rises* with
   stakes (opposite to H1).