"""
File: PGA_2D.py
Author: Peleg Bar Sapir
Date: 6/2025
Description: a very simple 2D Projective Geomteric Algebra (PGA) "engine"
             for experimenting with this algebra.
Remark: I'm using the following basis: 1, e0, e1, e2, e01, e20, e12, e012
        and the following signatures, respectively: +1, 0, +1, +1, 0, 0, -1, 0.
"""

from __future__ import annotations

from typing import override

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

npf64 = np.float64
npdarr = npt.NDArray[np.float64]
npiarr = npt.NDArray[np.int64]

e0, e1, e2, e01, e20, e12, e012 = np.arange(1, 8)

sign_matrix: npiarr = np.array([1, 0, 1, 1, 0, 0, -1, 0])


class MultiVector:
    """Docstring for ClassName."""

    def __init__(self, vals: npdarr) -> None:
        self.vals: npdarr = vals
        self.scalar: npf64 = self.vals[0]
        self.vector: npdarr = self.vals[1:4]
        self.bivector: npdarr = self.vals[4:7]
        self.trivector: npf64 = self.vals[7]

    def __add__(self, V2: MultiVector) -> MultiVector:
        return MultiVector(vals=self.vals + V2.vals)

    @override
    def __repr__(self) -> str:
        return (
            f"{self.scalar:0.3f}*1 + {self.vector[0]:0.3f}*e0 + {self.vector[1]:0.3f}*e1 + "
            f"{self.vector[2]:0.3f}*e2 + {self.bivector[0]:0.3f}*e01 + "
            f"{self.bivector[1]:0.3f}*e20 + {self.bivector[2]:0.3f}*e12 + "
            f"{self.trivector:0.3f}*e012"
        )

    def as_point(self) -> npdarr:
        """Convert the bivector part to a point in standard Cartesian notation
        P=(x,y)."""
        return self.bivector[:2][::-1] / self.bivector[2]


def wedge(A: MultiVector, B: MultiVector) -> MultiVector:
    C: npdarr = np.zeros(8, dtype=npf64)
    C[0] = A.vals[0] * B.vals[0]
    C[1] = A.vals[1] * B.vals[0] + A.vals[0] * B.vals[1]
    C[2] = A.vals[2] * B.vals[0] + A.vals[0] * B.vals[2]
    C[3] = A.vals[3] * B.vals[0] + A.vals[0] * B.vals[3]
    C[4] = (
        A.vals[4] * B.vals[0]
        + A.vals[0] * B.vals[4]
        - A.vals[2] * B.vals[1]
        + A.vals[1] * B.vals[2]
    )
    C[5] = (
        A.vals[5] * B.vals[0]
        + A.vals[3] * B.vals[1]
        - A.vals[1] * B.vals[3]
        + A.vals[0] * B.vals[5]
    )
    C[6] = (
        A.vals[6] * B.vals[0]
        - A.vals[3] * B.vals[2]
        + A.vals[2] * B.vals[3]
        + A.vals[0] * B.vals[6]
    )
    C[7] = (
        A.vals[7] * B.vals[0]
        + A.vals[6] * B.vals[1]
        + A.vals[5] * B.vals[2]
        + A.vals[4] * B.vals[3]
        + A.vals[3] * B.vals[4]
        + A.vals[2] * B.vals[5]
        + A.vals[1] * B.vals[6]
        + A.vals[0] * B.vals[7]
    )

    return MultiVector(vals=C)


def intersection_point(a: npf64, b: npf64, c: npf64, d: npf64) -> npdarr | None:
    L: npf64 = a - b
    if L == 0.0:
        return None
    D: npf64 = (d - c) / L
    return np.array([D, a * D + c])


def test_line_intersection() -> bool:
    coeffs: npdarr = np.random.uniform(-10, 10, 6)
    G1, A1, B1 = coeffs[:3]
    G2, A2, B2 = coeffs[3:]
    if B1 == 0 or B2 == 0:
        return False
    a, c = -A1 / B1, -G1 / B1
    b, d = -A2 / B2, -G2 / B2

    I1: npdarr | None = intersection_point(a, b, c, d)
    # print("classic calc:", I1)

    c1: MultiVector = MultiVector(vals=np.array([0, *coeffs[:3], 0, 0, 0, 0]))
    c2: MultiVector = MultiVector(vals=np.array([0, *coeffs[3:], 0, 0, 0, 0]))
    c12: MultiVector = wedge(c1, c2)

    I2: npdarr | None = c12.as_point()
    # print("PGA calc:", I2)

    return np.allclose(I1, I2)


if __name__ == "__main__":
    n: int = 10000
    for i in tqdm(range(n), desc="Testing line-line intersection"):
        if not test_line_intersection():
            print(f"Comparison failed! case #{i}")
            break
