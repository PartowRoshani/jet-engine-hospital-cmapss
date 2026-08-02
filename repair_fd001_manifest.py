from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


TEXT_EXTENSIONS = {".json", ".txt", ".csv", ".md", ".py", ".yaml", ".yml", ".toml"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_artifact_path(space_dir: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    candidates = [
        space_dir / rel,
        space_dir / "artifacts" / rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve artifact path {relative_path!r}. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize FD001 text artifacts to LF line endings and "
            "refresh size/SHA-256 values in fd001_manifest.json."
        )
    )
    parser.add_argument(
        "--space-dir",
        default="huggingface_space",
        help="Path to the deployed Streamlit app directory.",
    )
    args = parser.parse_args()

    space_dir = Path(args.space_dir).resolve()
    manifest_path = (
        space_dir
        / "artifacts"
        / "metadata"
        / "fd001_manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    backup_path = manifest_path.with_suffix(".json.bak")
    shutil.copy2(manifest_path, backup_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("Manifest does not contain a non-empty 'artifacts' mapping.")

    changes: list[dict[str, object]] = []

    for artifact_name, info in artifacts.items():
        if not isinstance(info, dict):
            raise RuntimeError(f"Invalid manifest entry for {artifact_name!r}")

        relative_path = info.get("relative_path")
        if not relative_path:
            raise RuntimeError(
                f"Manifest entry {artifact_name!r} has no relative_path."
            )

        artifact_path = resolve_artifact_path(space_dir, str(relative_path))
        original = artifact_path.read_bytes()
        normalized = original

        if artifact_path.suffix.lower() in TEXT_EXTENSIONS:
            normalized = original.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            if normalized != original:
                artifact_path.write_bytes(normalized)

        actual = artifact_path.read_bytes()
        old_size = info.get("size_bytes")
        old_sha = info.get("sha256")
        new_size = len(actual)
        new_sha = sha256_bytes(actual)

        info["size_bytes"] = new_size
        info["sha256"] = new_sha

        changes.append(
            {
                "artifact": artifact_name,
                "file": artifact_path.name,
                "old_size": old_size,
                "new_size": new_size,
                "line_endings_changed": normalized != original,
                "sha_changed": old_sha != new_sha,
            }
        )

    manifest_text = json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
    ) + "\n"
    manifest_path.write_bytes(manifest_text.encode("utf-8"))

    print(f"Backup created: {backup_path}")
    print(f"Updated manifest: {manifest_path}")
    print()
    print("Artifact audit")
    print("=" * 100)
    for row in changes:
        print(
            f"{row['artifact']}: {row['file']} | "
            f"size {row['old_size']} -> {row['new_size']} | "
            f"LF normalized={row['line_endings_changed']} | "
            f"SHA updated={row['sha_changed']}"
        )
    print("=" * 100)
    print("FD001 manifest refresh completed successfully.")


if __name__ == "__main__":
    main()
