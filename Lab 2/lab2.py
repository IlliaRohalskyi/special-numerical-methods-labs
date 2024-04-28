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

    def largest_eigenvalue(self, eps=1e-6):
        """
        Finds the largest eigenvalue (by magnitude) of the matrix using the
        iterative method by magnitude.

        Args:
            eps: The desired tolerance for convergence.

        Returns:
            A tuple containing the approximate largest eigenvalue and the corresponding eigenvector.
        """
        num_rows = self.rows
        eigenvector = [1.0 if i == 0 else 0.0 for i in range(num_rows)]
        previous_eigenvector = None

        while True:
            intermediate_result = [
                sum(self.data[i][j] * eigenvector[j] for j in range(num_rows))
                for i in range(num_rows)
            ]
            largest_magnitude = max(abs(val) for val in intermediate_result)
            eigenvector = [val / largest_magnitude for val in intermediate_result]

            if previous_eigenvector is None:
                previous_eigenvector = eigenvector.copy()
                continue
            convergence = (
                sum(
                    abs(eigenvector[i] - previous_eigenvector[i])
                    for i in range(num_rows)
                )
                < eps
            )

            if convergence:
                return largest_magnitude, eigenvector
            previous_eigenvector = eigenvector.copy()

    def __str__(self):
        """String representation of the matrix."""
        return "\n".join([" ".join(map(str, row)) for row in self.data])


def test():
    """Test the Matrix class."""
    a_np = np.random.rand(5, 5)

    a_matrix = Matrix(a_np.shape[0], a_np.shape[1])
    a_matrix.data = a_np.tolist()

    eigvals, _ = np.linalg.eig(a_np)
    largest_eigval_np = np.abs(eigvals).max()

    largest_eigval_matrix, _ = a_matrix.largest_eigenvalue()

    print("Matrix (NumPy):")
    print(a_np)
    print("\nMatrix (Matrix class):")
    print(a_matrix)

    assert np.allclose(
        largest_eigval_matrix, largest_eigval_np
    ), "Approximation inaccurate"

    print("Test passed! The function seems to be working correctly.")


test()
