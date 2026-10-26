"""Researcher allocation and fresh-trial setup for the phone study."""

from __future__ import annotations

import csv
import hashlib
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from backend.study.materials import StudyMaterials


MATERIALS_DIR = Path(__file__).resolve().parents[2] / "materials"
COUNTERBALANCING_PATH = MATERIALS_DIR / "counterbalancing.csv"
EXPECTED_COLUMNS = (
    "slot",
    "sequence",
    "period_1_condition",
    "period_1_variant",
    "period_1_visual_sheet",
    "period_2_condition",
    "period_2_variant",
    "period_2_visual_sheet",
)


class TrialSetupRequest(BaseModel):
    participant_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    slot: str = Field(pattern=r"^[0-9]{2}$")
    period: int = Field(ge=1, le=2)

    @field_validator("slot", mode="before")
    @classmethod
    def normalize_slot(cls, value: object) -> object:
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value).zfill(2)
        if isinstance(value, str) and value.isdigit() and len(value) == 1:
            return value.zfill(2)
        return value


class TrialRecord(BaseModel):
    trial_id: str
    participant_id: str
    slot: str
    period: int
    condition: str
    interface_label: str
    wine_variant: str
    visual_sheet: str
    status: str
    created_at: str
    allocation_checksum: str
    materials_version: str
    scenario_checksum: str
    trace_checksum: str
    build_hash: str


def load_allocations(path: Path = COUNTERBALANCING_PATH) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError("counterbalancing.csv has unexpected columns")
        rows = list(reader)
    if len({row["slot"] for row in rows}) != len(rows):
        raise ValueError("counterbalancing.csv contains duplicate slots")
    for row in rows:
        if row["period_1_condition"] == row["period_2_condition"]:
            raise ValueError(f"slot {row['slot']} repeats a condition")
        for period in (1, 2):
            if row[f"period_{period}_condition"] not in {"full_anisha", "condition_c"}:
                raise ValueError(f"slot {row['slot']} has an unknown condition")
            if row[f"period_{period}_variant"] not in {"wine_1", "wine_2"}:
                raise ValueError(f"slot {row['slot']} has an unknown wine variant")
            if row[f"period_{period}_visual_sheet"] not in {"X", "Y"}:
                raise ValueError(f"slot {row['slot']} has an unknown visual sheet")
    return rows, hashlib.sha256(raw).hexdigest()


class StudySetupStore:
    """A separate database; trial creation never touches the live project workspace."""

    def __init__(self, database_path: Path, allocation_path: Path = COUNTERBALANCING_PATH):
        self.database_path = database_path
        self.allocation_path = allocation_path

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """CREATE TABLE IF NOT EXISTS study_trials (
                trial_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                slot TEXT NOT NULL,
                period INTEGER NOT NULL,
                condition TEXT NOT NULL,
                interface_label TEXT NOT NULL,
                wine_variant TEXT NOT NULL,
                visual_sheet TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                allocation_checksum TEXT NOT NULL,
                materials_version TEXT NOT NULL DEFAULT '',
                scenario_checksum TEXT NOT NULL DEFAULT '',
                trace_checksum TEXT NOT NULL DEFAULT '',
                build_hash TEXT NOT NULL DEFAULT ''
            )"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS study_material_snapshots (
                trial_id TEXT PRIMARY KEY,
                materials_json TEXT NOT NULL,
                snapshot_checksum TEXT NOT NULL,
                FOREIGN KEY (trial_id) REFERENCES study_trials(trial_id)
            )"""
        )
        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(study_trials)")
        }
        for column in (
            "materials_version",
            "scenario_checksum",
            "trace_checksum",
            "build_hash",
        ):
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE study_trials ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                )
        return connection

    def allocations(self) -> list[dict[str, str]]:
        return load_allocations(self.allocation_path)[0]

    def create_trial(self, request: TrialSetupRequest) -> TrialRecord:
        from backend.study.build import current_build_hash
        from backend.study.materials import load_study_materials
        from backend.study.runner import StudyRunner

        allocations, checksum = load_allocations(self.allocation_path)
        materials = load_study_materials()
        allocation = next((row for row in allocations if row["slot"] == request.slot), None)
        if allocation is None:
            raise ValueError(f"Unknown counterbalancing slot: {request.slot}")
        prefix = f"period_{request.period}_"
        record = TrialRecord(
            trial_id=f"study-{uuid4()}",
            participant_id=request.participant_id,
            slot=request.slot,
            period=request.period,
            condition=allocation[prefix + "condition"],
            interface_label="Interface A" if request.period == 1 else "Interface B",
            wine_variant=allocation[prefix + "variant"],
            visual_sheet=allocation[prefix + "visual_sheet"],
            status="awaiting_briefing",
            created_at=datetime.now(timezone.utc).isoformat(),
            allocation_checksum=checksum,
            materials_version=materials.version,
            scenario_checksum=materials.scenario_checksum,
            trace_checksum=materials.trace_checksum,
            build_hash=current_build_hash(),
        )
        with closing(self._connect()) as connection:
            StudyRunner.create_schema(connection)
            with connection:
                connection.execute(
                    """INSERT INTO study_trials (
                        trial_id, participant_id, slot, period, condition,
                        interface_label, wine_variant, visual_sheet, status,
                        created_at, allocation_checksum, materials_version,
                        scenario_checksum, trace_checksum, build_hash
                    ) VALUES (
                        :trial_id, :participant_id, :slot, :period, :condition,
                        :interface_label, :wine_variant, :visual_sheet, :status,
                        :created_at, :allocation_checksum, :materials_version,
                        :scenario_checksum, :trace_checksum, :build_hash
                    )""",
                    record.model_dump(),
                )
                self._save_material_snapshot(connection, record, materials)
                StudyRunner.initialize(connection, record, materials)
        return record

    @staticmethod
    def _save_material_snapshot(connection: sqlite3.Connection, trial: TrialRecord,
                                materials: StudyMaterials) -> None:
        if (
            trial.materials_version != materials.version
            or trial.scenario_checksum != materials.scenario_checksum
            or trial.trace_checksum != materials.trace_checksum
        ):
            raise ValueError("Material snapshot does not match this trial")
        contents = materials.model_dump_json()
        checksum = hashlib.sha256(contents.encode("utf-8")).hexdigest()
        connection.execute(
            """INSERT INTO study_material_snapshots
               (trial_id, materials_json, snapshot_checksum) VALUES (?, ?, ?)""",
            (trial.trial_id, contents, checksum),
        )

    def restore_material_snapshot(self, trial_id: str, materials: StudyMaterials) -> None:
        """Recover a legacy trial only from materials matching its recorded hashes."""
        trial = self.get_trial(trial_id)
        if trial is None:
            raise LookupError("Trial not found")
        with closing(self._connect()) as connection, connection:
            self._save_material_snapshot(connection, trial, materials)

    def get_trial(self, trial_id: str) -> TrialRecord | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM study_trials WHERE trial_id = ?", (trial_id,)
            ).fetchone()
        return TrialRecord(**dict(row)) if row is not None else None

    def materials_for_trial(self, trial_id: str) -> StudyMaterials | None:
        from backend.study.materials import StudyMaterials, load_study_materials

        trial = self.get_trial(trial_id)
        if trial is None:
            return None
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT materials_json, snapshot_checksum FROM study_material_snapshots WHERE trial_id = ?",
                (trial_id,),
            ).fetchone()
        if row is not None:
            if hashlib.sha256(row["materials_json"].encode("utf-8")).hexdigest() != row["snapshot_checksum"]:
                raise ValueError("Stored study material snapshot is damaged")
            materials = StudyMaterials.model_validate_json(row["materials_json"])
        else:
            materials = load_study_materials()
        if (
            trial.scenario_checksum != materials.scenario_checksum
            or trial.trace_checksum != materials.trace_checksum
            or trial.materials_version != materials.version
        ):
            raise ValueError("Study materials changed since this trial was created")
        if row is None:
            with closing(self._connect()) as connection, connection:
                self._save_material_snapshot(connection, trial, materials)
        return materials
