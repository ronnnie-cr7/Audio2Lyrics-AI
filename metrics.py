"""
Evaluation Metrics
- WER (Word Error Rate) for ASR
- BLEU for translation quality
- chrF for character-level translation quality
- Timing consistency check
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class EvaluationMetrics:

    def compute_all(
        self,
        hypothesis: str,
        reference: str,
        segments: Optional[List[Dict]] = None,
    ) -> Dict:
        results = {}
        results["wer"] = self.wer(reference, hypothesis)
        results["bleu"] = self.bleu(reference, hypothesis)
        results["chrf"] = self.chrf(reference, hypothesis)
        if segments:
            results["timing"] = self.timing_consistency(segments)
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def wer(reference: str, hypothesis: str) -> float:
        """Word Error Rate."""
        try:
            from jiwer import wer
            return round(wer(reference, hypothesis), 4)
        except ImportError:
            # Manual WER
            ref_words = reference.lower().split()
            hyp_words = hypothesis.lower().split()
            return round(EvaluationMetrics._edit_distance(ref_words, hyp_words) / max(len(ref_words), 1), 4)

    @staticmethod
    def bleu(reference: str, hypothesis: str) -> float:
        """Corpus BLEU using sacrebleu."""
        try:
            from sacrebleu.metrics import BLEU
            bleu = BLEU(effective_order=True)
            score = bleu.sentence_score(hypothesis, [reference])
            return round(score.score, 2)
        except ImportError:
            return EvaluationMetrics._simple_bleu(reference, hypothesis)

    @staticmethod
    def chrf(reference: str, hypothesis: str) -> float:
        """chrF score via sacrebleu."""
        try:
            from sacrebleu.metrics import CHRF
            chrf = CHRF()
            score = chrf.sentence_score(hypothesis, [reference])
            return round(score.score, 2)
        except ImportError:
            return 0.0

    @staticmethod
    def timing_consistency(segments: List[Dict]) -> Dict:
        """Check for overlapping or negative-duration segments."""
        issues = []
        for i, seg in enumerate(segments):
            dur = seg["end"] - seg["start"]
            if dur <= 0:
                issues.append(f"Segment {i}: zero/negative duration ({dur:.3f}s)")
            if i > 0 and seg["start"] < segments[i - 1]["end"]:
                issues.append(f"Segment {i}: overlaps with previous")
        return {"issues": issues, "ok": len(issues) == 0}

    # ------------------------------------------------------------------
    @staticmethod
    def _edit_distance(a: list, b: list) -> int:
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]

    @staticmethod
    def _simple_bleu(reference: str, hypothesis: str) -> float:
        """Unigram precision as rough BLEU proxy."""
        ref_set = set(reference.lower().split())
        hyp_words = hypothesis.lower().split()
        if not hyp_words:
            return 0.0
        hits = sum(1 for w in hyp_words if w in ref_set)
        return round(hits / len(hyp_words) * 100, 2)
