"""
Module containing Matrix class for matrix operations.
"""

import numpy as np  # Needed for comparing results with numpy


class Matrix:
    """Class representing a matrix."""

    def __init__(self, rows, cols):
        """Initialize a Matrix object with given number of rows and columns."""
        self.rows = rows
        self.cols = cols
        self.data = [[0] * cols for _ in range(rows)]

    @classmethod
    def from_input(cls):
        """Create a Matrix object from user input."""
        rows = int(input("Enter the number of rows: "))
        cols = int(input("Enter the number of columns: "))
        matrix = cls(rows, cols)
        print("Enter the elements of the matrix:")
        for i in range(rows):
            row = list(map(float, input().split()))
            if len(row) != cols:
                raise ValueError(
                    "Number of elements in a row does not match the number of columns"
                )
            matrix.data[i] = row
        return matrix

    @classmethod
    def identity(cls, size):
        """Create an identity matrix of given size."""
        matrix = cls(size, size)
        for i in range(size):
            matrix.data[i][i] = 1
        return matrix

    def add(self, other):
        """Add another matrix to this matrix."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Cannot add matrices of different sizes")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result

    def scalar_multiply(self, scalar):
        """Multiply this matrix by a scalar."""
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] * scalar
        return result

    def matrix_multiply(self, other):
        """Multiply this matrix by another matrix."""
        if self.cols != other.rows:
            raise ValueError("Cannot multiply matrices with these dimensions")
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result.data[i][j] += self.data[i][k] * other.data[k][j]
        return result

    def transpose(self):
        """Transpose this matrix."""
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    def power(self, exponent):
        """Raise this matrix to the power of exponent."""
        if self.rows != self.cols:
            raise ValueError("Matrix must be square to raise to a power")
        result = Matrix.identity(self.rows)
        for _ in range(exponent):
            result = result.matrix_multiply(self)
        return result

    def __str__(self):
        """String representation of the matrix."""
        return "\n".join([" ".join(map(str, row)) for row in self.data])


def test():
    """Test the Matrix class."""
    matrix_a = Matrix(2, 2)
    matrix_a.data = [[1, 2], [3, 4]]
    matrix_b = Matrix(2, 2)
    matrix_b.data = [[5, 6], [7, 8]]

    print("A + B:")
    print(matrix_a.add(matrix_b))
    print(np.array(matrix_a.data) + np.array(matrix_b.data))

    print("2 * A:")
    print(matrix_a.scalar_multiply(2))
    print(2 * np.array(matrix_a.data))

    print("A * B:")
    print(matrix_a.matrix_multiply(matrix_b))
    print(np.matmul(np.array(matrix_a.data), np.array(matrix_b.data)))

    print("Transposed matrix A:")
    print(matrix_a.transpose())
    print(np.transpose(np.array(matrix_a.data)))

    print("A^2:")
    print(matrix_a.power(2))
    print(np.linalg.matrix_power(np.array(matrix_a.data), 2))


test()
