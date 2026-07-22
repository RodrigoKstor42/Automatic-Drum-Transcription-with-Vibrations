"""Helpers for resolving legacy dataset roots stored in old run configs."""
from pathlib import Path
import warnings


LEGACY_DATASET_DIR = "PHRASE_GENERATOR"
CURRENT_DATASET_DIR = "DATASET_GENERATOR"


def _resolve_path(path_value, repo_root=None):
    path = Path(path_value).expanduser()
    if not path.is_absolute() and repo_root is not None:
        path = Path(repo_root) / path
    return path.resolve()


def _replace_legacy_part(path):
    parts = list(Path(path).parts)
    for index, part in enumerate(parts):
        if part.upper() == LEGACY_DATASET_DIR:
            parts[index] = CURRENT_DATASET_DIR
            return Path(*parts)
    text = str(path)
    if LEGACY_DATASET_DIR in text:
        return Path(text.replace(LEGACY_DATASET_DIR, CURRENT_DATASET_DIR, 1))
    return None


def _display_path(path, repo_root=None):
    path = Path(path)
    if repo_root is not None:
        try:
            return path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
        except ValueError:
            pass
    return str(path)


def _result(original, resolved, source, attempted_migrated=None, repo_root=None):
    return {
        "dataset_root_original_from_run_config": str(original) if original is not None else None,
        "dataset_root_resolved": str(Path(resolved).resolve()),
        "dataset_root_source": source,
        "dataset_root_migrated_candidate": (
            str(Path(attempted_migrated).resolve()) if attempted_migrated is not None else None
        ),
        "dataset_root_display": _display_path(resolved, repo_root),
    }


def resolve_dataset_root(config_dataset_root, cli_dataset_root=None, repo_root=None):
    """Resolve a run dataset root, migrating PHRASE_GENERATOR to DATASET_GENERATOR if needed."""
    repo_root = Path(repo_root).resolve() if repo_root is not None else None
    original = _resolve_path(config_dataset_root, repo_root) if config_dataset_root else None
    attempted_migrated = None

    if cli_dataset_root:
        cli_path = _resolve_path(cli_dataset_root, repo_root)
        if cli_path.is_dir():
            return _result(original, cli_path, "cli_override", repo_root=repo_root)
        raise FileNotFoundError(
            "Dataset root override not found.\n"
            f"- cli_dataset_root: {cli_path}\n"
            f"- dataset_root_original_from_run_config: {original}\n"
            "Suggestion: pass an existing dataset path with --dataset-root."
        )

    if original is not None and original.is_dir():
        return _result(original, original, "config", repo_root=repo_root)

    if original is not None and LEGACY_DATASET_DIR in str(original):
        attempted_migrated = _replace_legacy_part(original)
        if attempted_migrated is not None:
            attempted_migrated = attempted_migrated.resolve()
            if attempted_migrated.is_dir():
                display = _display_path(attempted_migrated, repo_root)
                warnings.warn(f"Using migrated dataset root: {display}", RuntimeWarning)
                return _result(
                    original,
                    attempted_migrated,
                    "migrated_from_legacy",
                    attempted_migrated,
                    repo_root,
                )

    raise FileNotFoundError(
        "Dataset root not found.\n"
        f"- dataset_root_original_from_run_config: {original}\n"
        f"- migrated_candidate_tried: {attempted_migrated}\n"
        "Suggestion: use --dataset-root with the current DATASET_GENERATOR dataset path."
    )
