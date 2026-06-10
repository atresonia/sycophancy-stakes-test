"""Per-condition extraction tracker shared across pattern-based extractors.

Each extractor (MMLU letter, Elephant YTA/NTA) returns a pattern_idx alongside
its answer; pattern_idx is 1/2/3 for which regex matched, or None if nothing
matched. This tracker tallies those outcomes per condition and reports a
p3+none rate -- a high rate means the model's output shape is drifting away
from what the extractor expects.

Live experiment runs set display_progress=True so the tracker periodically
prints and aborts when the none-rate exceeds a threshold. Metrics scripts
operate over a saved CSV and pass display_progress=False to silently
re-tabulate without triggering aborts.
"""
from collections import defaultdict


class PatternExtractionTracker:
    PATTERN_KEYS = ("p1", "p2", "p3", "none")

    def __init__(
        self,
        *,
        title: str,
        abort_check_every: int,
        print_every: int = 1000,
        abort_none_rate: float = 0.10,
        display_progress: bool = True,
    ):
        self.title = title
        self.abort_check_every = abort_check_every
        self.print_every = print_every
        self.abort_none_rate = abort_none_rate
        self.display_progress = display_progress
        self._stats: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._call_count = 0

    def set_display_progress(self, display_progress: bool) -> None:
        self.display_progress = display_progress

    def record(self, condition: str, pattern_idx: int | None) -> None:
        key = f"p{pattern_idx}" if pattern_idx is not None else "none"
        self._stats[condition][key] += 1
        self._call_count += 1
        if not self.display_progress:
            return
        if self._call_count % self.print_every == 0:
            self.print_summary()
        if self._call_count % self.abort_check_every == 0:
            self._check_abort()

    def _check_abort(self) -> None:
        none_total = sum(c["none"] for c in self._stats.values())
        rate = none_total / self._call_count
        success = self._call_count - none_total
        print(
            f"[calls={self._call_count}] extraction success: "
            f"{success}/{self._call_count} ({1 - rate:.1%})"
        )
        if rate > self.abort_none_rate:
            self.print_summary()
            raise RuntimeError(
                f"Extraction failure rate {none_total}/{self._call_count} "
                f"({rate:.1%}) exceeds {self.abort_none_rate:.0%} threshold. Aborting."
            )

    def print_summary(self) -> None:
        width = 80
        print("\n" + "=" * width)
        print(self.title)
        print("=" * width)
        print(
            f"{'Condition':<34} {'p1':>6} {'p2':>6} {'p3':>6} "
            f"{'none':>6} {'p3+none%':>10}"
        )
        print("-" * width)
        for condition, counts in sorted(self._stats.items()):
            p1, p2, p3, none = (counts[k] for k in self.PATTERN_KEYS)
            total = p1 + p2 + p3 + none
            if total == 0:
                continue
            problem = 100 * (p3 + none) / total
            print(
                f"{condition:<34} {p1:>6} {p2:>6} {p3:>6} "
                f"{none:>6} {problem:>9.1f}%"
            )
        print("\nConcerning: p3+none > 5% suggests data quality issue.\n")
