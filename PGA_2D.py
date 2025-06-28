"""
File: PGA_2D.py
Author: Peleg Bar Sapir
Date: 6/2025
Description: a very simple 2D Projective Geomteric Algebra (PGA) "engine"
             for experimenting with this algebra.
Remark: I'm using the following basis: 1, e0, e1, e2, e01, e20, e12, e012
        and the following signatures, respectively: +1, 0, +1, +1, 0, 0, -1, 0.
"""

from functools import singledispatchmethod
from typing import overload, override

import numpy as np
import numpy.typing as npt

npf64 = npt.float64
npdarr = npt.NDArray[np.float64]
npiarr = npt.NDArray[np.int64]

e0, e1, e2, e01, e20, e12, e012 = np.arange(1, 8)

sign_matrix: npiarr = np.array([1, 0, 1, 1, 0, 0, -1, 0])

multiplication_table: npdarr = np.array(
    [
        [1, e0, e1, e2, e01, e20, e012],
        [e0, 0, e01, -e20, 0, 0, e012, 0],
        [e1, -e01, 1, e12, -e0, e012, e2, e20],
        [e2, e20, -e12, 1, e012, e0, -e1, e01],
        [e01, 0, e0, e012, 0, 0, -e20, 0],
        [e20, 0, e012, 0e0, 0, 0, e01, 0],
        [e12, e012, -e2, e1, e20, -e01, -1, -e0],
        [e012, 0, e20, e01, 0, 0, -e0, 0],
    ]
)


class MultiVector:
    """Docstring for ClassName."""

    def __init__(self, vals: npdarr) -> None:
        self.vals: npdarr = vals
        self.scalar: npf64 = self.vals[0]
        self.vector: npdarr = self.vals[1:4]
        self.bivector: npdarr = self.vals[4:7]
        self.trivector: npf64 = self.vals[7]

    @overload
    def __add__(self, V2: MultiVector):
        return MultiVector(vals=self.vals + V2.vals)

    @override
    def __repr__(self) -> str:
        return (
            f"{self.scalar}*1 + {self.vector[0]}*e1 + {self.vector[1]}*e2 + "
            f"{self.vector[2]}*e3 + {self.bivector[0]}*e12 + "
            f"{self.bivector[1]}*e23 + {self.bivector[2]}*e31 + "
            f"{self.trivector}*e123"
        )

    def as_point(self) -> npdarr:
        """Convert the bivector part to a point in standard Cartesian notation
        P=(x,y)."""
        return self.bivector[:2][::-1] / self.bivector[2]


if __name__ == "__main__":
    c1 = MultiVector(np.array([-1, 2.3, 0.0, 5.0, 2.0, 1.0, 0.0, -0.5]))
    print(c1)
    c1.vector[2] = 13.37
    print(c1)
