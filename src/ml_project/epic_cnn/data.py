"""Data preparation for strand-oriented EPIC CNN inputs."""

from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..sequence_context import extract_oriented_context


def prepare_oriented_cnn_batch(
    sequences: Mapping[str, np.ndarray],
    rows: pd.DataFrame,
    *,
    flank: int = 130,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare strand-oriented one-hot windows and targets on CPU."""

    if not isinstance(flank, int) or flank < 0:
        raise ValueError("flank должен быть неотрицательным целым числом")

    required_columns = {
        "contig",
        "coordinate_0based",
        "strand",
        "count",
    }
    missing_columns = required_columns - set(rows.columns)
    if missing_columns:
        raise ValueError(f"Отсутствуют колонки: {sorted(missing_columns)}")

    offsets = np.arange(-flank, flank + 1)
    context_codes = extract_oriented_context(sequences, rows, offsets)
    codes = torch.from_numpy(context_codes).long()

    x = (
        F.one_hot(codes, num_classes=5)
        .permute(0, 2, 1)
        .float()
        .contiguous()
    )
    count = torch.from_numpy(
        rows["count"].to_numpy(dtype=np.float32)
    ).reshape(-1, 1)
    target = (count > 0).float()

    return x, target, count


__all__ = ["prepare_oriented_cnn_batch"]
