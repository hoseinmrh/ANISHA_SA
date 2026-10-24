"""Wine-quality descriptive analysis review script, version 1."""

import csv
from pathlib import Path
from statistics import mean


def load_wines(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"alcohol", "volatile acidity", "quality"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Missing required wine-quality columns")
        rows = [
            {field: float(row[field]) for field in required}
            for row in reader
        ]
    if not rows:
        raise ValueError("The dataset is empty")
    return rows


def describe_groups(rows: list[dict[str, float]]) -> None:
    for label, group in (
        ("quality < 7", [row for row in rows if row["quality"] < 7]),
        ("quality >= 7", [row for row in rows if row["quality"] >= 7]),
    ):
        if not group:
            raise ValueError(f"No rows in {label}")
        print(label, "n=", len(group))
        for field in ("alcohol", "volatile acidity"):
            print(field, "mean=", round(mean(row[field] for row in group), 3))


if __name__ == "__main__":
    describe_groups(load_wines(Path("winequality-red.csv")))
