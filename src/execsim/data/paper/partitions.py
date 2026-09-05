"""Locked chronological folds and session policy for the paper study."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class PaperFold:
    """Define one global date partition shared by every instrument."""

    fold_id: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date

    def partition(self, session_date: date) -> str | None:
        """Return the unique partition for a date, or ``None`` outside the fold."""
        if self.train_start <= session_date <= self.train_end:
            return "train"
        if self.validation_start <= session_date <= self.validation_end:
            return "validation"
        if self.test_start <= session_date <= self.test_end:
            return "test"
        return None


PAPER_FOLDS = (
    PaperFold(
        "fold-1",
        date(2022, 1, 3),
        date(2023, 12, 29),
        date(2024, 1, 2),
        date(2024, 3, 28),
        date(2024, 4, 1),
        date(2024, 6, 28),
    ),
    PaperFold(
        "fold-2",
        date(2022, 1, 3),
        date(2024, 6, 28),
        date(2024, 7, 1),
        date(2024, 9, 30),
        date(2024, 10, 1),
        date(2024, 12, 31),
    ),
    PaperFold(
        "fold-3",
        date(2022, 1, 3),
        date(2024, 12, 31),
        date(2025, 1, 2),
        date(2025, 6, 30),
        date(2025, 7, 1),
        date(2025, 12, 31),
    ),
)


def paper_fold(fold_id: str) -> PaperFold:
    """Resolve one locked fold identifier or fail closed."""
    for fold in PAPER_FOLDS:
        if fold.fold_id == fold_id:
            return fold
    raise ValueError(f"Unknown paper fold: {fold_id}")


def resolve_fold_partition(fold_id: str, session_date: date) -> str:
    """Derive a session partition from the locked fold and date."""
    partition = paper_fold(fold_id).partition(session_date)
    if partition is None:
        raise ValueError(f"{session_date} is outside locked {fold_id} partitions.")
    return partition


def fold_training_cutoff(fold_id: str) -> date:
    """Return the last training date for a locked fold."""
    return paper_fold(fold_id).train_end


def validate_fold_membership(records: list[tuple[str, str, date, str]]) -> None:
    """Reject duplicate fold/instrument/date rows and caller-assigned partitions."""
    seen: set[tuple[str, str, date]] = set()
    for fold_id, instrument_id, session_date, partition in records:
        key = (fold_id, instrument_id, session_date)
        if key in seen:
            raise ValueError(
                f"Duplicate fold membership for {fold_id}/{instrument_id} on {session_date}."
            )
        seen.add(key)
        expected = resolve_fold_partition(fold_id, session_date)
        if partition != expected:
            raise ValueError(
                f"Fold partition mismatch for {fold_id}/{session_date}: "
                f"expected {expected}, got {partition}."
            )
