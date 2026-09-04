from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "repo_manifest.yaml"


def load_manifest() -> dict[str, Any]:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("areas"), dict):
        raise ValueError("repo_manifest.yaml must contain an areas mapping")
    return payload


def manifest_hash() -> str:
    return hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()


def _normalise(value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"Path is outside the repository: {value}") from exc
    return candidate.as_posix().lstrip("./")


def _matches(path: str, root: str) -> bool:
    root_path = PurePosixPath(root)
    path_value = PurePosixPath(path)
    return path_value == root_path or root_path in path_value.parents


def select_for_path(manifest: dict[str, Any], value: str) -> list[str]:
    path = _normalise(value)
    matches: list[tuple[int, str]] = []
    for name, area in manifest["areas"].items():
        lengths = [len(root) for root in area.get("roots", []) if _matches(path, root)]
        if lengths:
            matches.append((max(lengths), name))
    return [name for _, name in sorted(matches, key=lambda item: (-item[0], item[1]))]


def area_payload(manifest: dict[str, Any], names: list[str]) -> dict[str, Any]:
    return {
        "manifest_sha256": manifest_hash(),
        "areas": {name: manifest["areas"][name] for name in names},
        "authority": manifest.get("authority", {}),
    }


def check_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ("purpose", "roots", "specifications", "tests", "verify")
    for name, area in manifest["areas"].items():
        missing = [key for key in required if not area.get(key)]
        if missing:
            errors.append(f"area {name} missing values: {', '.join(missing)}")
        for spec in area.get("specifications", []):
            if not (ROOT / spec).exists():
                errors.append(f"area {name} references missing specification: {spec}")
        for root in area.get("roots", []):
            if root.startswith("src/") and (ROOT / root).suffix and not (ROOT / root).exists():
                errors.append(f"area {name} references missing source file: {root}")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select deterministic ExecSim repository context.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list", action="store_true", help="List registered areas.")
    group.add_argument("--area", help="Show one registered area.")
    group.add_argument("--path", help="Find the registered owner(s) for a repository path.")
    group.add_argument("--check", action="store_true", help="Validate manifest references.")
    parser.add_argument("--json", action="store_true", help="Emit deterministic JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest()
    if args.check:
        errors = check_manifest(manifest)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print(f"repository context: ok | areas={len(manifest['areas'])} | sha256={manifest_hash()}")
        return 0

    if args.area:
        if args.area not in manifest["areas"]:
            raise SystemExit(f"Unknown area: {args.area}")
        names = [args.area]
    elif args.path:
        names = select_for_path(manifest, args.path)
        if not names:
            raise SystemExit(f"No registered area for path: {args.path}")
    else:
        names = sorted(manifest["areas"])

    payload = area_payload(manifest, names)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for name in names:
            area = manifest["areas"][name]
            print(f"{name}: {area['purpose']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
