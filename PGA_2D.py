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
            f"{self.scalar:0.3f}*1 + {self.vector[0]:0.3f}*e1 + {self.vector[1]:0.3f}*e2 + "
            f"{self.vector[2]:0.3f}*e3 + {self.bivector[0]:0.3f}*e12 + "
            f"{self.bivector[1]:0.3f}*e23 + {self.bivector[2]:0.3f}*e31 + "
            f"{self.trivector:0.3f}*e123"
        )

    def as_point(self) -> npdarr:
        """Convert the bivector part to a point in standard Cartesian notation
        P=(x,y)."""
        return self.bivector[:2][::-1] / self.bivector[2]


def meet(A: MultiVector, B: MultiVector) -> MultiVector:
    C: npdarr = np.zeros(8, dtype=npf64)
    C[0] = (
        A.vals[0] * B.vals[0]
        + A.vals[2] * B.vals[2]
        + A.vals[3] * B.vals[3]
        - A.vals[6] * B.vals[6]
    )
    C[1] = (
        A.vals[0] * B.vals[1]
        + A.vals[1] * B.vals[0]
        - A.vals[2] * B.vals[4]
        + A.vals[3] * B.vals[5]
        + A.vals[4] * B.vals[2]
        - A.vals[5] * B.vals[3]
        - A.vals[6] * B.vals[7]
        - A.vals[7] * B.vals[6]
    )
    C[2] = (
        A.vals[0] * B.vals[2]
        + A.vals[2] * B.vals[0]
        - A.vals[3] * B.vals[6]
        + A.vals[6] * B.vals[3]
    )
    C[3] = (
        A.vals[0] * B.vals[3]
        + A.vals[2] * B.vals[6]
        + A.vals[3] * B.vals[0]
        - A.vals[6] * B.vals[2]
    )
    C[4] = (
        A.vals[0] * B.vals[4]
        + A.vals[1] * B.vals[2]
        - A.vals[2] * B.vals[1]
        + A.vals[3] * B.vals[7]
        + A.vals[4] * B.vals[0]
        + A.vals[5] * B.vals[6]
        - A.vals[6] * B.vals[5]
        + A.vals[7] * B.vals[3]
    )
    C[5] = (
        A.vals[0] * B.vals[5]
        - A.vals[1] * B.vals[3]
        + A.vals[2] * B.vals[7]
        + A.vals[3] * B.vals[1]
        - A.vals[4] * B.vals[6]
        + A.vals[5] * B.vals[0]
        + A.vals[6] * B.vals[4]
        + A.vals[7] * B.vals[2]
    )
    C[6] = (
        A.vals[0] * B.vals[6]
        + A.vals[2] * B.vals[3]
        - A.vals[3] * B.vals[2]
        + A.vals[6] * B.vals[0]
    )
    C[7] = (
        A.vals[0] * B.vals[7]
        + A.vals[1] * B.vals[6]
        + A.vals[2] * B.vals[5]
        + A.vals[3] * B.vals[4]
        + A.vals[4] * B.vals[3]
        + A.vals[5] * B.vals[2]
        + A.vals[6] * B.vals[1]
        + A.vals[7] * B.vals[0]
    )

    return MultiVector(vals=C)


if __name__ == "__main__":
    c1: MultiVector = MultiVector(vals=np.array([0, 0, 0, 0, 1, 2, 1, 0]))
    c2: MultiVector = MultiVector(vals=np.array([0, 0, 0, 0, 6, 3, 1, 0]))
    print(c1.as_point())
    print(c2.as_point())
    c3: MultiVector = meet(c1, c2)
    print(c3)
