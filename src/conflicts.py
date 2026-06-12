"""
Conflict registry.

Each conflict is a pair of opposing information ecosystems.
A side fixes the target language used to produce and judge a generation,
the country slot used to instantiate country-aligned prompt templates,
and the JSON key used to read the side-oriented reference from the dataset.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Side:
    code: str
    language: str
    country: str
    reference_key: str


@dataclass(frozen=True)
class Conflict:
    code: str
    sides: tuple[Side, Side]
    data_path: str

    def side(self, code: str) -> Side:
        for s in self.sides:
            if s.code == code:
                return s
        raise KeyError(f"side {code!r} not in conflict {self.code!r}")


CONFLICTS: dict[str, Conflict] = {
    "ru_ua": Conflict(
        code="ru_ua",
        sides=(
            Side(code="ru", language="Russian", country="Russia", reference_key="ru"),
            Side(code="ua", language="Ukrainian", country="Ukraine", reference_key="ua"),
        ),
        data_path="data/ru_ua/neutral_data.json",
    ),
    "il_ps": Conflict(
        code="il_ps",
        sides=(
            Side(code="il", language="Hebrew", country="Israel", reference_key="il"),
            Side(code="ps", language="Arabic", country="Palestine", reference_key="ps"),
        ),
        data_path="data/il_ps/neutral_data.json",
    ),
}


def get_conflict(code: str) -> Conflict:
    if code not in CONFLICTS:
        raise KeyError(f"unknown conflict {code!r}; known: {sorted(CONFLICTS)}")
    return CONFLICTS[code]
