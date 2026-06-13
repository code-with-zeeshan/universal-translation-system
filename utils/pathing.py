from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def to_path(path: PathLike) -> Path:
    return Path(path)


def resolve(path: str | Path) -> Path:
    return to_path(path).resolve()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: PathLike, parents: bool = True, exist_ok: bool = True) -> Path:
    p = to_path(path)
    p.mkdir(parents=parents, exist_ok=exist_ok)
    return p


def safe_join(*parts: PathLike) -> Path:
    return Path(*[str(p) for p in parts])


def filename_from(path: PathLike) -> str:
    return to_path(path).name


def stem_from(path: PathLike) -> str:
    return to_path(path).stem


def ext_from(path: PathLike) -> str:
    return to_path(path).suffix


def exists_and_nonempty(path: PathLike) -> bool:
    p = to_path(path)
    return p.exists() and p.stat().st_size > 0


def is_relative_to(path: PathLike, *other: PathLike) -> bool:
    try:
        to_path(path).relative_to(Path(*[str(o) for o in other]))
        return True
    except ValueError:
        return False


def sanitize_path(path: PathLike) -> Path:
    p = to_path(path)
    return p.resolve()


def check_path_traversal(path: PathLike, allowed_base: PathLike) -> bool:
    p = sanitize_path(path)
    base = sanitize_path(allowed_base)
    try:
        p.relative_to(base)
        return True
    except ValueError:
        return False
