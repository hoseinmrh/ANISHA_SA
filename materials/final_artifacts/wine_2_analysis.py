"""Wine-quality data validation review script, version 2."""

import csv
from pathlib import Path


def load_verified_rows(path: Path) -> list[dict[str, float]]:
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
    # Plausible extreme observations remain in the verified dataset.
    return rows


def describe_range(rows: list[dict[str, float]], field: str) -> None:
    values = [row[field] for row in rows]
    print(field, "minimum=", min(values), "maximum=", max(values))


if __name__ == "__main__":
    wines = load_verified_rows(Path("winequality-red.csv"))
    print("Retained rows:", len(wines))
    describe_range(wines, "alcohol")
    describe_range(wines, "volatile acidity")
