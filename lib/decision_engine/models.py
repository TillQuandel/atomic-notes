"""ADTs für decision_engine.

# CLAUDE-PATTERN: types.py wurde zu models.py umbenannt — vermeidet stdlib-Shadowing
# und den `if __name__ == "types"`-Workaround. Modul-Internes File-Layout ist
# besser als Konventions-Bruch (Python's `types` ist stdlib seit Anfang).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Label(Enum):
    SUPPORTED_EXACT = "supported_exact"
    SUPPORTED_PARAPHRASE = "supported_paraphrase"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_IN_CONTEXT = "not_in_context"
    CONTRADICTED = "contradicted"
    PARSE_ERROR = "parse_error"
    RETRIEVAL_UNCERTAIN = "retrieval_or_parse_uncertain"


SYSTEM_LABELS = frozenset({Label.PARSE_ERROR, Label.RETRIEVAL_UNCERTAIN})
NORMAL_LABELS = frozenset(Label) - SYSTEM_LABELS

# Strenge-Ordering: nur für NORMAL_LABELS. Systemlabels sind außerhalb der Ordnung
# — sie werden nie durch Audit überschrieben (siehe rule_audit_stricter_override).
STRICTNESS_RANK = {
    Label.SUPPORTED_EXACT: 0,
    Label.SUPPORTED_PARAPHRASE: 1,
    Label.PARTIALLY_SUPPORTED: 2,
    Label.NOT_IN_CONTEXT: 3,
    Label.CONTRADICTED: 4,
}


class QualityFlag(Enum):
    EVIDENCE_UNVERIFIED = "evidence_unverified"
    EVIDENCE_FABRICATED = "evidence_fabricated"
    RETRIEVAL_LOW_COSINE = "retrieval_low_cosine"
    LOW_COSINE = "low_cosine"  # deprecated alias
    AUDIT_OVERRIDDEN = "audit_overridden"
    AUDIT_OVERRODE = "audit_overrode"  # deprecated alias
    JUDGE_UNEINIG = "judge_uneinig"
    AUDIT_DISAGREES_SOFTER = "audit_disagrees_softer"
    AUDIT_DISAGREES_WITH_SYSTEM = "audit_disagrees_with_system"
    PARSE_ERROR = "parse_error"


@dataclass(frozen=True)
class ClaimInput:
    """Input für die Decision-Pipeline. __post_init__ validiert harte Invarianten.

    # CLAUDE-PATTERN: Validierung im __post_init__ statt im Caller verschiebt
    # die Garantie zur Modul-Boundary — Rules dürfen davon ausgehen dass cosine
    # finite und in [0,1] ist, parse_failed konsistent ist, etc.
    """

    primary_label: Label
    audit_label: Label | None
    cosine: float
    evidence_verified: bool | None
    parse_failed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.primary_label, Label):
            raise TypeError(f"primary_label must be Label, got {type(self.primary_label).__name__}")
        if self.audit_label is not None and not isinstance(self.audit_label, Label):
            raise TypeError(f"audit_label must be Label or None, got {type(self.audit_label).__name__}")
        if not isinstance(self.cosine, (int, float)) or isinstance(self.cosine, bool):
            raise TypeError(f"cosine must be number, got {type(self.cosine).__name__}")
        if math.isnan(self.cosine) or math.isinf(self.cosine):
            raise ValueError(f"cosine must be finite, got {self.cosine}")
        if not 0.0 <= self.cosine <= 1.0:
            raise ValueError(f"cosine must be in [0, 1], got {self.cosine}")
        if self.evidence_verified is not None and not isinstance(self.evidence_verified, bool):
            raise TypeError(f"evidence_verified must be bool or None, got {type(self.evidence_verified).__name__}")
        if not isinstance(self.parse_failed, bool):
            raise TypeError(f"parse_failed must be bool, got {type(self.parse_failed).__name__}")
        # Logische Konsistenz: parse_failed=True und evidence_verified gesetzt sind widersprüchlich.
        if self.parse_failed and self.evidence_verified is not None:
            raise ValueError("parse_failed=True is inconsistent with evidence_verified being set")


@dataclass(frozen=True)
class ClaimDecision:
    label: Label
    flags: frozenset[QualityFlag]
    source: str  # "primary" | "audit_override" | "system" | "downgrade"


@dataclass(frozen=True)
class Metric:
    """Rate-Metrik mit Validity-Flag. value=-1.0 sentinel wenn nicht messbar."""

    value: float
    valid: bool

    @classmethod
    def invalid(cls) -> "Metric":
        return cls(-1.0, False)
