import logging
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from human_reasons_based_supervision.config import Config
from human_reasons_based_supervision.exceptions import (
    InvalidReasonScoreError,
    InvalidReasonThresholdError,
)
from human_reasons_based_supervision.models import ReasonScore, ReasonThreshold
from human_reasons_based_supervision.utils import calculate_flow_theory_score

logger = logging.getLogger(__name__)

class HumanReasonsBasedSupervision:
    def __init__(self, config: Config):
        self.config = config
        self.reason_scores: Dict[str, ReasonScore] = {}
        self.reason_thresholds: Dict[str, ReasonThreshold] = {}

    def calculate_reason_scores(self, reason_data: Dict[str, float]) -> Dict[str, ReasonScore]:
        """
        Calculate reason scores based on the provided reason data.

        Args:
            reason_data (Dict[str, float]): A dictionary containing reason data.

        Returns:
            Dict[str, ReasonScore]: A dictionary containing calculated reason scores.
        """
        try:
            reason_scores = {}
            for reason, value in reason_data.items():
                if reason not in self.reason_scores:
                    self.reason_scores[reason] = ReasonScore(reason)
                self.reason_scores[reason].update(value)
                reason_scores[reason] = self.reason_scores[reason].calculate_score()
            return reason_scores
        except Exception as e:
            logger.error(f"Error calculating reason scores: {e}")
            raise InvalidReasonScoreError("Failed to calculate reason scores")

    def trigger_replanning(self, reason_scores: Dict[str, ReasonScore]) -> bool:
        """
        Trigger replanning based on the provided reason scores.

        Args:
            reason_scores (Dict[str, ReasonScore]): A dictionary containing reason scores.

        Returns:
            bool: Whether replanning should be triggered.
        """
        try:
            replanning_triggered = False
            for reason, score in reason_scores.items():
                if score > self.reason_thresholds[reason].threshold:
                    replanning_triggered = True
                    break
            return replanning_triggered
        except Exception as e:
            logger.error(f"Error triggering replanning: {e}")
            raise InvalidReasonThresholdError("Failed to trigger replanning")

    def update_reason_thresholds(self, reason_thresholds: Dict[str, ReasonThreshold]) -> None:
        """
        Update reason thresholds.

        Args:
            reason_thresholds (Dict[str, ReasonThreshold]): A dictionary containing reason thresholds.
        """
        self.reason_thresholds = reason_thresholds

    def calculate_flow_theory_score(self, reason_data: Dict[str, float]) -> float:
        """
        Calculate flow theory score based on the provided reason data.

        Args:
            reason_data (Dict[str, float]): A dictionary containing reason data.

        Returns:
            float: The calculated flow theory score.
        """
        return calculate_flow_theory_score(reason_data)


class ReasonScore:
    def __init__(self, reason: str):
        self.reason = reason
        self.score = 0.0

    def update(self, value: float) -> None:
        """
        Update the reason score.

        Args:
            value (float): The new value to update the score with.
        """
        self.score = value

    def calculate_score(self) -> float:
        """
        Calculate the reason score.

        Returns:
            float: The calculated reason score.
        """
        return self.score


class ReasonThreshold:
    def __init__(self, reason: str, threshold: float):
        self.reason = reason
        self.threshold = threshold


class Config:
    def __init__(self):
        self.reason_thresholds = {
            "reason1": ReasonThreshold("reason1", 0.5),
            "reason2": ReasonThreshold("reason2", 0.7),
        }


class InvalidReasonScoreError(Exception):
    pass


class InvalidReasonThresholdError(Exception):
    pass


def calculate_flow_theory_score(reason_data: Dict[str, float]) -> float:
    """
    Calculate flow theory score based on the provided reason data.

    Args:
        reason_data (Dict[str, float]): A dictionary containing reason data.

    Returns:
        float: The calculated flow theory score.
    """
    try:
        flow_theory_score = 0.0
        for reason, value in reason_data.items():
            flow_theory_score += value
        return flow_theory_score
    except Exception as e:
        logger.error(f"Error calculating flow theory score: {e}")
        raise Exception("Failed to calculate flow theory score")


if __name__ == "__main__":
    config = Config()
    human_reasons_based_supervision = HumanReasonsBasedSupervision(config)
    reason_data = {
        "reason1": 0.3,
        "reason2": 0.8,
    }
    reason_scores = human_reasons_based_supervision.calculate_reason_scores(reason_data)
    replanning_triggered = human_reasons_based_supervision.trigger_replanning(reason_scores)
    print(f"Replanning triggered: {replanning_triggered}")