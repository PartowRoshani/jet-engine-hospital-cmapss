from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def make_json_compatible(value: Any) -> Any:
    """
    Convert NumPy and Path objects into JSON-compatible values.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {
            str(key): make_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_compatible(item)
            for item in value
        ]

    return value


def save_json(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    """
    Save a dictionary as formatted UTF-8 JSON.
    """
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    compatible_data = make_json_compatible(
        data
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            compatible_data,
            file,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    return output_path


def load_json(
    path: str | Path,
) -> dict[str, Any]:
    """
    Load a JSON artifact.
    """
    input_path = Path(path)

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_joblib(
    artifact: Any,
    path: str | Path,
    compress: int = 3,
) -> Path:
    """
    Save a Python or scikit-learn artifact.
    """
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        output_path,
        compress=compress,
    )

    return output_path


def calculate_sha256(
    path: str | Path,
) -> str:
    """
    Calculate the SHA-256 checksum of a file.
    """
    input_path = Path(path)

    digest = hashlib.sha256()

    with input_path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def describe_artifact_file(
    path: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Return file metadata used in the artifact manifest.

    When root is supplied, a portable relative path is also
    stored so that the repository can be moved to another
    computer.
    """
    artifact_path = Path(path).resolve()

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {artifact_path}"
        )

    result = {
        "path": str(artifact_path),
        "filename": artifact_path.name,
        "size_bytes": artifact_path.stat().st_size,
        "sha256": calculate_sha256(
            artifact_path
        ),
    }

    if root is not None:
        root_path = Path(root).resolve()

        try:
            relative_path = artifact_path.relative_to(
                root_path
            )

            result["relative_path"] = (
                relative_path.as_posix()
            )

        except ValueError:
            result["relative_path"] = None

    return result