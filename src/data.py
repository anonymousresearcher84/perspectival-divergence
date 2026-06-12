from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .conflicts import Conflict


@dataclass(frozen=True)
class Item:
    id: str
    neutral: str
    references: dict[str, str]


def load_items(conflict: Conflict, root: str | Path = ".") -> list[Item]:
    path = Path(root) / conflict.data_path
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    keys = [s.reference_key for s in conflict.sides]
    items: list[Item] = []
    for idx, raw in enumerate(payload["messages"]):
        if "neutral" not in raw or any(k not in raw for k in keys):
            continue
        items.append(
            Item(
                id=raw.get("id", f"ex_{idx:05d}"),
                neutral=raw["neutral"].strip(),
                references={k: raw[k].strip() for k in keys},
            )
        )
    return items
