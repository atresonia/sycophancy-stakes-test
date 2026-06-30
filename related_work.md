# Related Work — Prompts that Elicit Behaviors in LLMs

Annotated bibliography for the sycophancy-stakes project. Grouped by theme. Each
entry: citation, venue, link, and one-line relevance to our adversarial
grade-inflation / sycophancy search. The final section flags the closest prior
work and the novelty threats a reviewer will raise.

All citations verified against arXiv / ACL Anthology / OpenReview (June 2026).

---

## 1. Automated generation of prompts that elicit a target behavior (red-teaming / adversarial search)

This is the methodological backbone of our adversarial setup. The lineage runs
from gradient attacks → LLM-as-attacker → quality-diversity search.

- **Perez et al. 2022 — Red Teaming Language Models with Language Models.** EMNLP 2022. https://aclanthology.org/2022.emnlp-main.225/
  The origin of "use one LM to automatically generate test cases that make a target LM misbehave." Establishes the attacker/target loop we use. Cite as the foundational automated red-teaming reference.

- **Zou et al. 2023 — Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG).** arXiv:2307.15043. https://arxiv.org/abs/2307.15043
  White-box, gradient-based token-suffix optimization starting from arbitrary tokens. This is the method whose "random init" intuition does NOT transfer to our black-box LLM-mutation setup — useful to cite when justifying why we seed from coherent prompts rather than random chars.

- **Chao et al. 2023 — Jailbreaking Black Box Large Language Models in Twenty Queries (PAIR).** arXiv:2310.08419. https://arxiv.org/abs/2310.08419
  Black-box attacker LM iteratively refines a prompt using the target's responses as feedback. This is essentially "single-objective" adversarial search — the thing our QD archive generalizes. Cite as the contrast to diversity-preserving search.

- **Mehrotra et al. 2023 — Tree of Attacks: Jailbreaking Black-Box LLMs Automatically (TAP).** NeurIPS 2024. arXiv:2312.02119. https://arxiv.org/abs/2312.02119
  Extends PAIR with tree search + pruning to cut target queries. Relevant if we need a query-efficiency argument for the search.

- **Liu et al. 2023 — AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned LLMs.** ICLR 2024. arXiv:2310.04451. https://arxiv.org/abs/2310.04451
  Hierarchical genetic algorithm (selection/crossover/mutation) over natural-language jailbreak prompts. Closest evolutionary-search cousin to our mutation operator; cite alongside Rainbow Teaming.

- **Samvelyan et al. 2024 — Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts.** NeurIPS 2024. arXiv:2402.16822. https://arxiv.org/abs/2402.16822
  **The paper our method is directly built on.** Casts adversarial prompt generation as MAP-Elites quality-diversity: an archive whose cells are descriptors (attack style, risk category, length). We adapt this from safety jailbreaks to sycophancy/grade-inflation with emergent strategy descriptors. Must be cited as our primary methodological basis AND differentiated (see §4).

- **Hong et al. 2024 — Curiosity-driven Red-teaming for Large Language Models.** ICLR 2024. arXiv:2402.19464. https://arxiv.org/abs/2402.19464
  RL red-teaming with a novelty bonus to increase coverage/diversity of test cases. Alternative to QD for the "don't collapse to one attack" problem — relevant to justify why we chose an archive over an RL novelty objective.

- **Paulus et al. 2024 — AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs.** arXiv:2404.16873. https://arxiv.org/abs/2404.16873
  Trains an attacker LM to emit human-readable adversarial suffixes ~800x faster than optimization. Relevant if we later want to amortize the search into a trained suffix-generator instead of per-instance search.

## 2. Persuasion / social-influence framings (most relevant to WHAT our attacks are)

- **Zeng et al. 2024 — How Johnny Can Persuade LLMs to Jailbreak Them (PAP).** ACL 2024. arXiv:2401.06373. https://arxiv.org/abs/2401.06373
  Builds a 40-technique persuasion taxonomy from social science and uses it to generate interpretable persuasive adversarial prompts; 92% ASR with no optimization. **Highly relevant:** our discovered winning axes (hardship, identity, sympathy, claimed effort) are essentially persuasion techniques. Cite to frame our emergent strategy cells, and differentiate: we measure graded-task sycophancy (continuous grade shift), not binary jailbreak success.

## 3. Sycophancy: definition, measurement, mitigation

- **Perez et al. 2022 — Discovering Language Model Behaviors with Model-Written Evaluations.** ACL Findings 2023. arXiv:2212.09251. https://arxiv.org/abs/2212.09251
  Auto-generates 154 eval datasets; first large-scale demonstration of sycophancy and its inverse-scaling with model size and RLHF. Foundational sycophancy-measurement reference.

- **Sharma et al. 2023 (Anthropic) — Towards Understanding Sycophancy in Language Models.** ICLR 2024. arXiv:2310.13548. https://arxiv.org/abs/2310.13548
  Shows sycophancy is consistent across 5 frontier assistants and traces it to human-preference data favoring agreeable responses. Our "knowledge baseline" (answer with no user signal) is the same conceptual anchor they use; cite when defining the baseline-relative metric.

- **Wei et al. 2023 — Simple Synthetic Data Reduces Sycophancy in Large Language Models.** arXiv:2308.03958. https://arxiv.org/abs/2308.03958
  Sycophancy rises with scale/instruction-tuning; a lightweight synthetic-data finetune reduces it. Relevant for the "minimize-sycophancy" objective and any mitigation discussion.

- **Fanous et al. 2025 — SycEval: Evaluating LLM Sycophancy.** (arXiv 2025; on Semantic Scholar). https://www.semanticscholar.org/paper/SycEval:-Evaluating-LLM-Sycophancy-Fanous-Goldberg/796f0ce165479e22f95c9f8d02b1b239816f46ef
  Standardized sycophancy metrics across math (AMPS) and medical (MedQuad); ~58% sycophancy rate across GPT-4o, Claude-Sonnet, Gemini-1.5-Pro. Cite as recent benchmarking precedent and a multi-model comparison point.

- **Cheng et al. 2025 — ELEPHANT: Measuring and Understanding Social Sycophancy in LLMs.** arXiv:2505.13995. https://arxiv.org/abs/2505.13995
  **The most threatening adjacent work.** Defines "social sycophancy" as excessive preservation of the user's face/self-image, and evaluates on advice queries + **r/AmITheAsshole (2,000 posts)** + subjective statements; finds LLMs affirm both sides of a moral conflict 48% of the time and are ~45pp more sycophantic than humans. Overlaps directly with our AITA experiment and our "sympathy/identity" findings. Must be engaged and differentiated (see §4).

## 4. Prompt sensitivity & automatic prompt optimization (behavior elicitation, non-adversarial)

- **Shin et al. 2020 — AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts.** EMNLP 2020. arXiv:2010.15980. https://arxiv.org/abs/2010.15980
  Gradient-guided search for prompts that elicit factual knowledge. Early "search the prompt space to elicit a behavior" reference predating jailbreak work.

- **Zhou et al. 2022 — Large Language Models Are Human-Level Prompt Engineers (APE).** ICLR 2023. arXiv:2211.01910. https://arxiv.org/abs/2211.01910
  Treats the prompt as a program optimized over LLM-proposed candidates against a score function — structurally identical to our generate-score-select loop, applied to task performance rather than sycophancy.

- **Sclar et al. 2023 — Quantifying LMs' Sensitivity to Spurious Features in Prompt Design (FormatSpread).** ICLR 2024. arXiv:2310.11324. https://arxiv.org/abs/2310.11324
  Semantically-equivalent formatting changes swing accuracy by up to 76 points; uses Bayesian optimization to find min/max performance. Strong support for "small prompt changes cause large behavior shifts," and a methodological caution: report a spread, not a single-prompt number.

## 5. Quality-Diversity foundations

- **Mouret & Clune 2015 — Illuminating Search Spaces by Mapping Elites (MAP-Elites).** arXiv:1504.04909. https://arxiv.org/abs/1504.04909
  The algorithm underneath Rainbow Teaming and our archive: maintain the best solution per cell of a user-chosen descriptor space to "illuminate" how attributes relate to performance. Cite for the archive/elitism mechanics and the descriptor-space concept.

---

## Closest prior work & novelty threats (read this part)

Three papers sit close enough to force a clear differentiation, or a reviewer
will call the contribution incremental:

1. **Rainbow Teaming (Samvelyan 2024) — method overlap.** We use its exact
   MAP-Elites-for-adversarial-prompts machinery. "Rainbow Teaming applied to
   sycophancy" is not, by itself, a contribution. Our differentiators have to be
   explicit: (a) target behavior is graded sycophancy measured as a continuous
   shift vs. a knowledge baseline, not binary attack success; (b) emergent,
   embedding-routed strategy descriptors rather than hand-specified cells;
   (c) the cross-task comparison of inducible-sycophancy ceilings (essay /
   factual / AITA) and the stakes axis. Lead with (c).

2. **ELEPHANT (Cheng 2025) — task + phenomenon overlap.** It already studies
   social sycophancy on AITA and advice, with a face-preservation framing, and
   reports models affirming both sides of moral conflicts. This is the biggest
   threat to the AITA portion of our paper. Differentiate on method: ELEPHANT
   measures *prevalence on fixed datasets*; we *adversarially search* for the
   prompt features that maximize/minimize sycophancy and quantify an inducible
   ceiling per task. If we don't cite and distinguish ELEPHANT, a reviewer will
   assume we're unaware of it.

3. **PAP / persuasion taxonomy (Zeng 2024) — explanation overlap.** Our winning
   grade-inflation suffixes (hardship, ESL/immigrant, disability, sympathy) are
   recognizable persuasion techniques. PAP can be read as already showing
   "persuasion jailbreaks LLMs." Our angle that survives: we show these
   techniques transfer to a *non-safety, graded* setting and that the model is
   manipulable by *unverifiable* identity/hardship claims — a measurement of
   epistemic vulnerability, not a new jailbreak. Consider mapping our emergent
   cells onto their taxonomy to strengthen the framing.

Net: the defensible contribution is not the search method (Rainbow Teaming) nor
the observation that LLMs are sycophantic/persuadable (Sharma, Perez, Zeng,
ELEPHANT). It is the **cross-task inducible-sycophancy landscape** — using
adversarial QD search to map, per task, how high sycophancy can be driven and
which axes (including stakes) move it — and the finding that the dominant lever
is *unverifiable* author claims rather than stakes. Frame the paper around that.
