"""Die-link graph data structures for ancient mint reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator

import numpy as np


class DieSide(str, Enum):
    OBVERSE = "obverse"
    REVERSE = "reverse"


@dataclass(frozen=True)
class CoinObservation:
    """A single observed coin linking an obverse die to a reverse die."""

    coin_id: str
    obverse_die_id: str
    reverse_die_id: str
    weight_g: float
    diameter_mm: float
    find_site: str = "unknown"
    hoard_id: str | None = None
    style_vector: tuple[float, ...] = field(default_factory=tuple)

    def dies(self) -> tuple[str, str]:
        return self.obverse_die_id, self.reverse_die_id


@dataclass
class DieNode:
    """A die (obverse or reverse) in the mint production network."""

    die_id: str
    side: DieSide
    wear_index: float = 0.0
    style_vector: np.ndarray = field(default_factory=lambda: np.zeros(8))
    estimated_sequence: int | None = None
    mint_id: str = "unknown"

    def to_feature_vector(self) -> np.ndarray:
        side_flag = 1.0 if self.side == DieSide.OBVERSE else 0.0
        return np.concatenate(
            [
                np.array([self.wear_index, side_flag], dtype=np.float32),
                self.style_vector.astype(np.float32),
            ]
        )


@dataclass
class DieLinkEdge:
    """An edge indicating two dies were used together on at least one coin."""

    source_die: str
    target_die: str
    coin_count: int = 1
    confidence: float = 1.0


@dataclass
class WearTransferEdge:
    """Directed edge: source die was used before target (wear propagation)."""

    earlier_die: str
    later_die: str
    transfer_strength: float
    evidence_coins: list[str] = field(default_factory=list)


@dataclass
class DieLinkGraph:
    """Full die-link network for a mint or coin series."""

    mint_id: str
    nodes: dict[str, DieNode]
    coin_links: list[DieLinkEdge]
    wear_transfers: list[WearTransferEdge]
    observations: list[CoinObservation]

    @property
    def num_dies(self) -> int:
        return len(self.nodes)

    @property
    def num_coins(self) -> int:
        return len(self.observations)

    def die_ids(self) -> list[str]:
        return list(self.nodes.keys())

    def iter_wear_pairs(self) -> Iterator[tuple[str, str, float]]:
        for edge in self.wear_transfers:
            yield edge.earlier_die, edge.later_die, edge.transfer_strength

    def chronological_order(self) -> list[str]:
        """Return die IDs sorted by estimated production sequence."""
        ordered = [
            (die_id, node.estimated_sequence)
            for die_id, node in self.nodes.items()
            if node.estimated_sequence is not None
        ]
        ordered.sort(key=lambda x: x[1])
        return [die_id for die_id, _ in ordered]
