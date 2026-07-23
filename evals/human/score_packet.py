"""Score completed blind Eval 29 rating sheets with the private answer key."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .build_packet import DIMENSIONS


def score(ratings_path: Path, key_path: Path) -> dict:
    key = json.loads(key_path.read_text())["items"]
    values = defaultdict(lambda: defaultdict(list))
    preferences = defaultdict(int)
    evaluators, rows_scored = set(), 0
    with ratings_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            item_id = row.get("item_id", "")
            if item_id not in key or not row.get("evaluator_id"):
                continue
            evaluators.add(row["evaluator_id"])
            for slot in ("A", "B"):
                method = key[item_id][slot]
                for dimension in DIMENSIONS:
                    raw = row.get("%s_%s" % (slot, dimension), "")
                    try:
                        rating = float(raw)
                    except ValueError:
                        raise ValueError("non-numeric rating for %s" % item_id)
                    if not 1.0 <= rating <= 5.0:
                        raise ValueError("rating outside 1..5 for %s" % item_id)
                    values[method][dimension].append(rating)
            preference = row.get("preference", "").strip().upper()
            if preference in ("A", "B"):
                preferences[key[item_id][preference]] += 1
            elif preference == "TIE":
                preferences["tie"] += 1
            else:
                raise ValueError("preference must be A, B, or tie")
            rows_scored += 1
    medians = {method: {dimension: float(np.median(ratings))
                        for dimension, ratings in dimensions.items()}
               for method, dimensions in values.items()}
    decisive = sum(count for method, count in preferences.items()
                   if method != "tie")
    preference_rates = {method: count / decisive for method, count in preferences.items()
                        if method != "tie"} if decisive else {}
    return {"evaluator_count": len(evaluators), "rating_row_count": rows_scored,
            "median_scores": medians, "preference_counts": dict(preferences),
            "decisive_preference_rates": preference_rates}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = score(Path(args.ratings), Path(args.answer_key))
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
