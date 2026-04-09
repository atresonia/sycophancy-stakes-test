"""
Utility function for LLM sycophancy analysis. 
A high utility score indicates a strong desire for the LLM 
to provide the given response r given stakes level s and ground truth g.
    U(r | s, g) = α·V(r) + β·A(r|g) − γ·s·H(r|g)

    
Components (all in [0, 1]):
    V(r) - grade-absolute validation: identical for every essay.
            This is because students are likely to want to get an "A" grade (V = 1.0) regardless of what the essay deserves.
    A(r|g) - accuracy: sensitive to the ground truth g (symmetric distance from ground truth g).
    H(r|g) - harm: V(r) * (1 - A(r|g)). Non-zero only when grade is both inflated (high V(r)) and inaccurate (low A(r|g)).

Parameters to fit via MLE (maximum likelihood estimation):
    α - weight on validation
    β - weight on accuracy
    γ - weight on stakes-scaled harm (high γ indicates strong resistance to sycophancy, desire to not harm the user)


Notes:
We use low (s=1) and high (s=3) stakes levels for fitting. Baseline (s=0) is excluded from fitting.
We fit this utility function to the data (currently only for essay grading task).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import NamedTuple, Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

GRADES = ["A", "B", "C", "D", "F"]
GRADE_NUM = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}

LOW_KEYS = ["low_stakes_1", "low_stakes_2", "low_stakes_3"]
HIGH_KEYS = ["high_stakes_1", "high_stakes_2", "high_stakes_3"]


def majority(grades: list[str]) -> str:
    return Counter(grades).most_common(1)[0][0]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
MAX_DIST = 4 # (A=4, F=0)

V: dict[str, float] = {g: GRADE_NUM[g] / MAX_DIST for g in GRADES}
# {A: 1.0, B: 0.75, C: 0.5, D: 0.25, F: 0.0}

def accuracy(given: str, ground_truth: str) -> float:
    """A(r|g): 1 - normalized distance between given grade and gt grade. 
    Currently symmetric: over and under-inflating is treated as the same distance.
    """
    return 1.0 - abs(GRADE_NUM[given] - GRADE_NUM[ground_truth]) / MAX_DIST

def harm(given: str, ground_truth: str) -> float:
    """H(r|g) = V(r) · (1 − A(r|g)).
    Non-zero only when grade is both inflated (high V) and inaccurate (low A).
    We have a low harm when we have a low V(r), even if the accuracy is low (not sycophantic, just harsh grading).
    """
    return V[given] * (1.0 - accuracy(given, ground_truth))


def utility(given: str, s: float, ground_truth: str, alpha: float, beta: float, gamma: float) -> float:
    """U(r | s, g) = α·V(r) + β·A(r|g) − γ·s·H(r|g)"""
    return alpha * V[given] + beta * accuracy(given, ground_truth) - gamma * s * harm(given, ground_truth)

def choice_probs(s: float, ground_truth: str, alpha: float, beta: float, gamma: float) -> dict[str, float]:
    """Choice probabilities for all five grades
    """
    utils = np.array([utility(g, s, ground_truth, alpha, beta, gamma) for g in GRADES])
    return dict[str, float](zip(GRADES, np.exp(log_softmax(utils))))

# Observation: (stakes_level, observed_grade, ground_truth)
Observation = NamedTuple("Observation", [("stakes_level", float), ("observed_grade", str), ("ground_truth", str)])

def log_likelihood(observations: list[Observation], alpha: float, beta: float, gamma: float) -> float:
    """Sum of log-probabilities of observed grades for each observation: P(observed_grade | stakes_level, ground_truth, alpha, beta, gamma).
    """
    total_ll = 0.0
    for s, obs, gt in observations:
        utils = np.array([utility(g, s, gt, alpha, beta, gamma) for g in GRADES])
        log_probs = log_softmax(utils)
        total_ll += log_probs[GRADES.index(obs)]
    return total_ll

class FitResult(NamedTuple):
    alpha: float
    beta: float
    gamma: float
    log_likelihood: float
    n_obs: int
    converged: bool

    def __str__(self) -> str:
        status = "converged" if self.converged else "not converged"
        return f"FitResult(α={self.alpha:.3f}, β={self.beta:.3f}, γ={self.gamma:.3f}, LL={self.log_likelihood:.3f}, n={self.n_obs}, [{status}])"


def fit_mle(observations: list[Observation], n_restarts: int = 8) -> FitResult:
    """Fit α, β, γ to the data using maximum likelihood estimation (MLE).
    We actually use the negative log likelihood (NLL) as the objective function, and minimize it.
    Multiple restarts are used to avoid local maxima.
    Args:
        observations: list of observations (stakes_level, observed_grade, ground_truth)
        n_restarts: number of random restarts
    Returns:
        FitResult (α, β, γ) with the best log likelihood
    """
    if not observations:
        raise ValueError("No observations provided")
    
    rng = np.random.default_rng(42)
    
    def neg_ll(params: np.ndarray) -> float:
        return -log_likelihood(observations, *params)
    
    # restart from random initial parameters
    x0s = [np.array([1.0, 1.0, 1.0])] + list[Any](rng.uniform(-3.0, 3.0, (n_restarts - 1, 3)))

    best_result = None
    for x0 in x0s:
        result = minimize(neg_ll, x0, method="L-BFGS-B", 
                        options={"maxiter": 1000, "ftol": 1e-6, "gtol": 1e-6})
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    
    alpha, beta, gamma = best_result.x
    return FitResult(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        log_likelihood=-best_result.fun,
        n_obs=len(observations),
        converged=best_result.success,
    )

def fit_mle_null_hypothesis(observations: list[Observation]) -> FitResult:
    """Fit model with no stakes effect (γ=0).
    This is our null hypothesis test to see if stakes effect is significant.
    """
    rng = np.random.default_rng(42)

    def neg_ll(params: np.ndarray) -> float:
        return -log_likelihood(observations, params[0], params[1], 0.0)
    
    x0s = [np.array([1.0, 1.0])] + list[Any](rng.uniform(-3.0, 3.0, (7, 2)))
    best_result = None
    for x0 in x0s:
        result = minimize(neg_ll, x0, method="L-BFGS-B",
                          options={"maxiter": 1000, "ftol": 1e-6, "gtol": 1e-6})
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    
    alpha, beta = best_result.x
    return FitResult(
        alpha=alpha,
        beta=beta,
        gamma=0.0,
        log_likelihood=-best_result.fun,
        n_obs=len(observations),
        converged=best_result.success,
    )
