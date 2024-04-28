"""
Module containing Matrix class for matrix operations.
"""


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

    def lu_decomposition(self):
        """
        Performs LU decomposition of the matrix.

        Returns:
            A tuple containing the L and U matrices as Matrix objects.
        """
        num_rows = self.rows  # Use descriptive variable name
        lower_triangular_matrix = Matrix.identity(num_rows)
        upper_triangular_matrix = Matrix(num_rows, num_rows)
        for i in range(num_rows):
            for j in range(i, num_rows):
                upper_triangular_matrix.data[i][j] = self.data[i][j] - sum(
                    lower_triangular_matrix.data[i][k]
                    * upper_triangular_matrix.data[k][j]
                    for k in range(i)
                )
            for j in range(i + 1, num_rows):
                lower_triangular_matrix.data[j][i] = (
                    self.data[j][i]
                    - sum(
                        lower_triangular_matrix.data[j][k]
                        * upper_triangular_matrix.data[k][i]
                        for k in range(i)
                    )
                ) / upper_triangular_matrix.data[i][i]
        return lower_triangular_matrix, upper_triangular_matrix

    def determinant(self):
        """Calculate the determinant of the matrix using LU decomposition."""
        _, upper = self.lu_decomposition()
        det = 1
        for i in range(self.rows):
            det *= upper.data[i][i]
        return det

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

    sample_matrix = Matrix(3, 3)
    sample_matrix.data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    lower, upper = sample_matrix.lu_decomposition()
    print("Lower triangular matrix (L):")
    print(lower)
    print("\nUpper triangular matrix (U):")
    print(upper)

    det = sample_matrix.determinant()
    print("\nDeterminant:", det)

    reconstructed_matrix = lower.matrix_multiply(upper)

    for i in range(3):
        for j in range(3):
            assert (
                sample_matrix.data[i][j] == reconstructed_matrix.data[i][j]
            ), "LU decomposition test failed!"

    print("\nLU decomposition test passed!")


test()
