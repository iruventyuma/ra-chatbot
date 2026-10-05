"""
Implements the 2010 ACR/EULAR Classification Criteria for Rheumatoid
Arthritis. Takes structured symptom answers (collected via the chatbot's
questioning step) and returns a score breakdown + classification.

Score >= 6/10 -> classified as RA.
Reference: Aletaha et al., 2010, Arthritis & Rheumatism.
"""
from dataclasses import dataclass


@dataclass
class ACREularInput:
    small_joints_involved: int      # count of small joints (e.g. MCP, PIP, wrist) swollen/tender
    large_joints_involved: int      # count of large joints (shoulder, elbow, hip, knee, ankle) swollen/tender
    rf_or_accp_positive: str        # "negative" | "low_positive" | "high_positive"
    crp_or_esr_abnormal: bool       # True if CRP or ESR is elevated
    symptom_duration_weeks: float   # how long symptoms have lasted


def _joint_involvement_score(small: int, large: int) -> int:
    """
    2010 ACR/EULAR Domain A (joint involvement), evaluated from the
    highest category down:
      >10 joints total, with at least 1 small joint  -> 5
      4-10 small joints (large joints may also be involved)  -> 3
      1-3 small joints (large joints may also be involved)   -> 2
      2-10 large joints only                                 -> 1
      1 large joint only                                     -> 0
    """
    total = small + large
    if total > 10 and small >= 1:
        return 5
    if small >= 4:
        return 3
    if small >= 1:
        return 2
    if large >= 2:
        return 1
    return 0  # 0 or 1 large joint, no small joints


def _serology_score(rf_accp_status: str) -> int:
    return {
        "negative": 0,
        "low_positive": 2,
        "high_positive": 3,
    }.get(rf_accp_status, 0)


def _acute_phase_score(abnormal: bool) -> int:
    return 1 if abnormal else 0


def _duration_score(weeks: float) -> int:
    return 1 if weeks >= 6 else 0


def calculate_acr_eular(data: ACREularInput) -> dict:
    joint_score = _joint_involvement_score(data.small_joints_involved, data.large_joints_involved)
    serology_score = _serology_score(data.rf_or_accp_positive)
    acute_phase_score = _acute_phase_score(data.crp_or_esr_abnormal)
    duration_score = _duration_score(data.symptom_duration_weeks)

    total = joint_score + serology_score + acute_phase_score + duration_score
    classified_as_ra = total >= 6

    return {
        "joint_involvement_score": joint_score,
        "serology_score": serology_score,
        "acute_phase_score": acute_phase_score,
        "duration_score": duration_score,
        "total_score": total,
        "max_score": 10,
        "classified_as_ra": classified_as_ra,
        "summary": (
            f"ACR/EULAR score: {total}/10 -- "
            f"{'meets' if classified_as_ra else 'does not meet'} the threshold "
            f"for RA classification (>=6/10). This is a classification suggestion, "
            f"not a diagnosis -- confirm with a rheumatologist."
        ),
    }
