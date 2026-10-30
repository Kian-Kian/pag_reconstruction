"""Machine-readable experiment provenance records."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_manifest(
    *,
    experiment_name: str,
    command: list[str],
    config: dict[str, object],
    source_files: Iterable[str | Path],
    data_files: Iterable[str | Path],
    results: dict[str, object],
) -> dict[str, object]:
    sources = {str(Path(path)): sha256_file(path) for path in source_files}
    data = {str(Path(path)): sha256_file(path) for path in data_files}
    return {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config": config,
        "source_sha256": sources,
        "data_sha256": data,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "results": results,
    }


def write_experiment_manifest(value: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
