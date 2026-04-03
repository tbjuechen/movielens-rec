from dataclasses import dataclass
from pathlib import Path

from src.config.settings import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_STORE_DIR,
    MODEL_WEIGHTS_DIR,
    ALIGNMENT_MODE,
)


STRICT_MINONICC_MODE = "strict_minonicc"


@dataclass(frozen=True)
class AlignmentPaths:
    mode: str
    processed_dir: Path
    feature_store_dir: Path
    model_weights_dir: Path
    reports_dir: Path


def resolve_alignment_mode(cli_alignment: str | None = None) -> str:
    mode = (cli_alignment or ALIGNMENT_MODE or "").strip()
    return mode


def get_alignment_paths(cli_alignment: str | None = None) -> AlignmentPaths:
    mode = resolve_alignment_mode(cli_alignment)
    if mode == STRICT_MINONICC_MODE:
        base = BASE_DIR / "data" / "aligned" / STRICT_MINONICC_MODE
        return AlignmentPaths(
            mode=mode,
            processed_dir=base / "processed",
            feature_store_dir=base / "feature_store",
            model_weights_dir=base / "model_weights",
            reports_dir=base / "reports",
        )
    return AlignmentPaths(
        mode="",
        processed_dir=Path(PROCESSED_DATA_DIR),
        feature_store_dir=Path(FEATURE_STORE_DIR),
        model_weights_dir=Path(MODEL_WEIGHTS_DIR),
        reports_dir=BASE_DIR / "data" / "reports",
    )

