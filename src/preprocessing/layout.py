"""
Which landmarks we keep, and where each group ends up in the feature vector.

Per frame the saved .npy row is (492 columns):

    body        0:99     33 pts x (x,y,z)  shoulder-normalized
    left_hand  99:162    21 pts x 3        shoulder-normalized
    right_hand 162:225   21 pts x 3        shoulder-normalized
    face      225:357    44 pts x 3        shoulder-normalized
    face_local 357:489   44 pts x 3        relative to nose tip, scaled by eye distance
    presence  489:492    left hand / right hand / face detected (0 or 1)

Training picks columns with FeatureLayout.columns(...), e.g. body+hands only = 225 features.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from ..config import (BODY, FACE, FACE_EYE_LEFT_OUTER, FACE_EYE_RIGHT_OUTER, FACE_NOSE,
                      FACE_POINTS, LEFT_HAND, RIGHT_HAND)


class FeatureLayout:
    """Column map of the processed feature vector."""

    PRESENCE_NAMES = ("left_hand_present", "right_hand_present", "face_present")

    def __init__(self, n_face: int = len(FACE_POINTS)):
        sizes = [
            ("body", 33 * 3),
            ("left_hand", 21 * 3),
            ("right_hand", 21 * 3),
            ("face", n_face * 3),
            ("face_local", n_face * 3),
            ("presence", len(self.PRESENCE_NAMES)),
        ]
        self.slices: dict[str, tuple[int, int]] = {}
        start = 0
        for name, size in sizes:
            self.slices[name] = (start, start + size)
            start += size
        self.dim = start

    def columns(self, *groups: str) -> list[int]:
        """Column indices for the given groups, e.g. columns('body', 'left_hand', 'right_hand')."""
        cols = []
        for g in groups:
            s, e = self.slices[g]
            cols.extend(range(s, e))
        return cols

    def to_dict(self) -> dict:
        return {"dim": self.dim, "slices": self.slices, "presence": list(self.PRESENCE_NAMES),
                "face_points": list(FACE_POINTS)}

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


@dataclass(frozen=True)
class PointSelection:
    """Indices into the raw 576-point .pose array, resolved from a file header."""
    indices: tuple[int, ...]     # raw point indices, in order body | lhand | rhand | face
    body: slice
    left_hand: slice
    right_hand: slice
    face: slice
    left_shoulder: int           # positions inside the *selected* points
    right_shoulder: int
    face_nose: int               # positions inside the face subset
    face_eye_right: int
    face_eye_left: int


class PointSelector:
    """Builds a PointSelection from a pose header (cached per header layout)."""

    def __init__(self, face_points=FACE_POINTS):
        self.face_points = tuple(face_points)
        self._cache: dict[tuple, PointSelection] = {}

    def __call__(self, header) -> PointSelection:
        key = tuple((c.name, len(c.points)) for c in header.components)
        if key not in self._cache:
            self._cache[key] = self._build(header)
        return self._cache[key]

    def _build(self, header) -> PointSelection:
        offsets, names = {}, {}
        start = 0
        for c in header.components:
            offsets[c.name] = start
            names[c.name] = list(c.points)
            start += len(c.points)
        for comp in (BODY, LEFT_HAND, RIGHT_HAND, FACE):
            if comp not in offsets:
                raise ValueError(f"{comp} missing from pose header; found {list(offsets)}")

        body = [offsets[BODY] + i for i in range(len(names[BODY]))]
        lhand = [offsets[LEFT_HAND] + i for i in range(len(names[LEFT_HAND]))]
        rhand = [offsets[RIGHT_HAND] + i for i in range(len(names[RIGHT_HAND]))]
        face = [offsets[FACE] + names[FACE].index(str(i)) for i in self.face_points]

        nb, nl, nr, nf = len(body), len(lhand), len(rhand), len(face)
        return PointSelection(
            indices=tuple(body + lhand + rhand + face),
            body=slice(0, nb),
            left_hand=slice(nb, nb + nl),
            right_hand=slice(nb + nl, nb + nl + nr),
            face=slice(nb + nl + nr, nb + nl + nr + nf),
            left_shoulder=names[BODY].index("LEFT_SHOULDER"),
            right_shoulder=names[BODY].index("RIGHT_SHOULDER"),
            face_nose=self.face_points.index(FACE_NOSE),
            face_eye_right=self.face_points.index(FACE_EYE_RIGHT_OUTER),
            face_eye_left=self.face_points.index(FACE_EYE_LEFT_OUTER),
        )
