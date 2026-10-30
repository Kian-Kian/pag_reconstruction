"""Runtime helpers shared by training entry points."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str | Path, **state: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def json_dump(data: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(data):
        data = asdict(data)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
