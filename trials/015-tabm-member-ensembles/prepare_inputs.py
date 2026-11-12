import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/local-t015"


def main():
    batch = ROOT / "artifacts/kaggle-t014"
    report = json.loads((batch / "results.json").read_text())
    if report["status"] != "COMPLETE" or len(report["results"]) != 19:
        raise ValueError("Wait for the complete nineteen-model batch before ensembling")
    protocol = report["protocol"]
    assert protocol["training_rows"] == 1230110 and protocol["validation_rows"] == 307528
    by_name = {row["name"]: row for row in report["results"]}
    candidates, families = {}, {}
    for name, members in [
        ("cross-masked-plr", 8), ("ensemble-16", 16), ("ensemble-32", 32),
    ]:
        spec = by_name[name]["spec"]
        assert spec.get("members", 8) == members
        assert spec["vector_crosses"] and spec["drop_high"]
        assert spec["numerical_encoding"] == "plr"
        candidates[f"tabm-k{members}"] = str(
            (batch / name / "validation_predictions.npz").relative_to(ROOT)
        )
        families[f"tabm-k{members}"] = "TabM"
    candidates["lightgbm"] = "artifacts/local-t011/lightgbm/validation_predictions.npz"
    families["lightgbm"] = "LightGBM"
    candidates["previous-ensemble"] = "artifacts/local-t013/validation_predictions.npz"
    families["previous-ensemble"] = "Previous ensemble"
    for path in candidates.values():
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "candidates.json").write_text(json.dumps({
        "candidates": candidates, "families": families, "core_names": list(candidates),
        "scope": "new PLR/cross/masked TabM k8/k16/k32 predictions on the full primary holdout",
    }, indent=2) + "\n")
    print("Prepared all five aligned full-holdout ensemble inputs")


if __name__ == "__main__":
    main()
