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

multiplication_table: npiarr = np.array(
    [
        [1, e0, e1, e2, e01, e20, e12, e012],
        [e0, 0, e01, -e20, 0, 0, e012, 0],
        [e1, -e01, 1, e12, -e0, e012, e2, e20],
        [e2, e20, -e12, 1, e012, e0, -e1, e01],
        [e01, 0, e0, e012, 0, 0, -e20, 0],
        [e20, 0, e012, 0e0, 0, 0, e01, 0],
        [e12, e012, -e2, e1, e20, -e01, -1, -e0],
        [e012, 0, e20, e01, 0, 0, -e0, 0],
    ],
    dtype=np.int64,
)


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


if __name__ == "__main__":
    c1 = MultiVector(np.random.uniform(-1, 1, size=8))
    c2 = MultiVector(np.random.uniform(size=8))
    print(c1)
    print(c2)
    c12 = c1 + c2
    c21 = c2 + c1
    for x1, x2, x12, x21 in zip(c1.vals, c2.vals, c12.vals, c21.vals):
        assert x1 + x2 == x12 and x12 == x21
        print(f"{x1:0.2f} + {x2:0.2f} = {x12:0.2f} = {x21:0.2f}")
