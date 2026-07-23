"""Build deterministic, anonymized pairwise human-evaluation packets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


DIMENSIONS = (
    "reference_resemblance", "shape_plausibility", "part_correctness",
    "editability", "topology_cleanliness", "material_plausibility",
    "starting_asset_usefulness", "uncertainty_trustworthiness",
)


def _order(case_id: str, methods: Tuple[str, str], seed: str) -> Tuple[str, str]:
    digest = hashlib.sha256((seed + "\0" + case_id).encode()).digest()
    return methods if digest[0] % 2 == 0 else (methods[1], methods[0])


def build_packet(dataset: Path, method_assets: Dict[str, str], output: Path,
                 seed: str = "recon3d-eval29-v1") -> Dict:
    if len(method_assets) != 2:
        raise ValueError("pairwise packet requires exactly two methods")
    methods = tuple(sorted(method_assets))
    public = output / "public"
    private = output / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    items, key = [], {}
    for case in sorted(path for path in dataset.iterdir()
                       if path.is_dir() and (path / "input.png").is_file()):
        case_id = case.name
        first, second = _order(case_id, methods, seed)
        item_id = hashlib.sha256((seed + case_id).encode()).hexdigest()[:12]
        item_dir = public / item_id
        item_dir.mkdir(exist_ok=True)
        reference = item_dir / "reference.png"
        shutil.copy2(case / "input.png", reference)
        slots = {}
        for slot, method in (("A", first), ("B", second)):
            source = Path(method_assets[method].format(case_id=case_id))
            if not source.is_file():
                raise FileNotFoundError("missing %s asset for %s: %s" % (
                    method, case_id, source))
            suffix = source.suffix.lower() or ".png"
            destination = item_dir / (slot + suffix)
            shutil.copy2(source, destination)
            slots[slot] = str(destination.relative_to(output))
        items.append({"item_id": item_id, "reference": str(
            reference.relative_to(output)), "slots": slots})
        key[item_id] = {"case_id": case_id, "A": first, "B": second}
    public_manifest = {"schema_version": 1, "blind": True,
                       "dimensions": list(DIMENSIONS), "items": items}
    (public / "manifest.json").write_text(json.dumps(
        public_manifest, indent=2, sort_keys=True))
    (private / "answer_key.json").write_text(json.dumps(
        {"seed": seed, "items": key}, indent=2, sort_keys=True))
    columns = ["evaluator_id", "expertise", "item_id"]
    for slot in ("A", "B"):
        columns.extend("%s_%s" % (slot, dimension)
                       for dimension in DIMENSIONS)
    columns += ["preference", "notes"]
    with (public / "ratings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for item in items:
            writer.writerow(["", "", item["item_id"]] +
                            [""] * (len(columns) - 3))
    return {"item_count": len(items), "methods": list(methods),
            "public_manifest": str(public / "manifest.json"),
            "private_answer_key": str(private / "answer_key.json")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/benchmark/dataset")
    parser.add_argument("--method", action="append", required=True,
                        help="name=path/template/{case_id}.png; exactly two")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", default="recon3d-eval29-v1")
    args = parser.parse_args()
    methods = {}
    for raw in args.method:
        if "=" not in raw:
            parser.error("--method must be name=path-template")
        name, template = raw.split("=", 1)
        methods[name] = template
    result = build_packet(Path(args.dataset), methods, Path(args.out), args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
