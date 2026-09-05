"""Point-in-time corpus infrastructure for the sparse-JEPA paper study."""

from execsim.data.paper.partitions import PAPER_FOLDS, PaperFold
from execsim.data.paper.schemas import PaperDataConfig, PaperUniverseMember

__all__ = ["PAPER_FOLDS", "PaperDataConfig", "PaperFold", "PaperUniverseMember"]
