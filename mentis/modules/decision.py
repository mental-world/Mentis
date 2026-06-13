from __future__ import annotations

from mentis.config import MentisConfig
from mentis.schema import ScoreResult


class DecisionModule:
    def __init__(self, config: MentisConfig) -> None:
        self.config = config

    def decide(self, scores: list[ScoreResult]) -> tuple[str, dict]:
        ordered = sorted(scores, key=self._sort_key, reverse=True)
        final = ordered[0].option_id if ordered else ""
        return final, {
            "weights": self.config.scoring.weights,
            "tie_break_order": self.config.scoring.tie_break_order,
            "ranked_options": [
                {
                    "option_id": s.option_id,
                    "mentally_consistent": s.mentally_consistent,
                    "physically_plausible": s.physically_plausible,
                    "socially_appropriate": s.socially_appropriate,
                    "safety_legality_veto": s.safety_legality_veto,
                    "raw_value_score": s.raw_value_score,
                    "overall_score": s.overall_score,
                }
                for s in ordered
            ],
        }

    def _sort_key(self, score: ScoreResult) -> tuple:
        key = [score.overall_score]
        for item in self.config.scoring.tie_break_order:
            key.append(_tie_break_value(score, item))
        return tuple(key)


def _option_order(option_id: str) -> int:
    if len(option_id) == 1 and option_id.isalpha():
        return ord(option_id.upper()) - ord("A")
    try:
        return int(option_id)
    except ValueError:
        return 999


def _tie_break_value(score: ScoreResult, item: str) -> float:
    values = {
        "physically_plausible": score.physically_plausible,
        "mentally_consistent": score.mentally_consistent,
        "socially_appropriate": score.socially_appropriate,
        "option_order": -_option_order(score.option_id),
    }
    return float(values.get(item, 0.0))
