import csv
import json

from evals.human.build_packet import DIMENSIONS, build_packet
from evals.human.score_packet import score


def test_blind_packet_separates_public_manifest_and_private_key(tmp_path):
    dataset = tmp_path / "dataset"
    methods = tmp_path / "methods"
    for case in ("a", "b"):
        (dataset / case).mkdir(parents=True)
        (dataset / case / "input.png").write_bytes(b"ref")
        for method in ("full", "baseline"):
            (methods / method).mkdir(parents=True, exist_ok=True)
            (methods / method / (case + ".png")).write_bytes(method.encode())
    out = tmp_path / "packet"
    result = build_packet(dataset, {
        "full": str(methods / "full" / "{case_id}.png"),
        "baseline": str(methods / "baseline" / "{case_id}.png"),
    }, out)
    public = json.loads((out / "public" / "manifest.json").read_text())
    private = json.loads((out / "private" / "answer_key.json").read_text())
    assert result["item_count"] == 2
    assert "case_id" not in json.dumps(public)
    assert set(private["items"]) == {item["item_id"] for item in public["items"]}
    rows = list(csv.reader((out / "public" / "ratings.csv").open()))
    assert len(rows) == 3
    assert all("A_%s" % dimension in rows[0] for dimension in DIMENSIONS)


def test_score_packet_maps_blind_slots_back_to_methods(tmp_path):
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"items": {"item": {
        "case_id": "case", "A": "full", "B": "baseline"}}}))
    ratings = tmp_path / "ratings.csv"
    columns = ["evaluator_id", "expertise", "item_id"]
    for slot in ("A", "B"):
        columns += ["%s_%s" % (slot, dimension) for dimension in DIMENSIONS]
    columns += ["preference", "notes"]
    with ratings.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow(["artist", "3d_artist", "item"] +
                        [5] * len(DIMENSIONS) + [2] * len(DIMENSIONS) +
                        ["A", ""])
    result = score(ratings, key)
    assert result["median_scores"]["full"]["editability"] == 5.0
    assert result["decisive_preference_rates"]["full"] == 1.0
