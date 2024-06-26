"""
Module containing Matrix class for matrix operations.
"""

import math

import numpy as np


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
        num_rows = self.rows
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

    def norm(self, norm_type="1"):
        """Calculate the specified norm of the matrix."""
        if norm_type == "1":
            return max(sum(abs(row[i]) for row in self.data) for i in range(self.cols))
        if norm_type == "2":
            return math.sqrt(sum(sum(x**2 for x in row) for row in self.data))
        if norm_type == "inf":
            return max(sum(abs(x) for x in row) for row in self.data)
        raise ValueError("Invalid norm type. Use '1', '2', or 'inf'.")

    def condition_number(self):
        """Calculate the condition number of the matrix."""
        matrix_inv = self.inverse()

        norm_inv = Matrix(matrix_inv.rows, matrix_inv.cols)
        norm_inv.data = matrix_inv.data
        norm_inv_value = norm_inv.norm("2")

        norm_matrix = self.norm("2")
        cond_number = norm_inv_value * norm_matrix

        return cond_number

    def inverse(self):
        """Calculate the inverse of the matrix."""
        if self.rows != self.cols:
            raise ValueError("Matrix must be square to calculate the inverse")

        identity_matrix = Matrix.identity(self.rows)

        augmented_matrix = Matrix(self.rows, self.cols * 2)
        augmented_matrix.data = [
            row + identity_matrix.data[i] for i, row in enumerate(self.data)
        ]

        for i in range(self.rows):
            max_row = max(
                range(i, self.rows), key=lambda r, i=i: abs(augmented_matrix.data[r][i])
            )

            augmented_matrix.data[i], augmented_matrix.data[max_row] = (
                augmented_matrix.data[max_row],
                augmented_matrix.data[i],
            )

            pivot = augmented_matrix.data[i][i]
            if pivot == 0:
                raise ValueError("Matrix is singular")

            for j in range(i, 2 * self.cols):
                augmented_matrix.data[i][j] /= pivot

            for k in range(i + 1, self.rows):
                factor = augmented_matrix.data[k][i]
                for j in range(i, 2 * self.cols):
                    augmented_matrix.data[k][j] -= factor * augmented_matrix.data[i][j]

        for i in range(self.rows - 1, 0, -1):
            for k in range(i - 1, -1, -1):
                factor = augmented_matrix.data[k][i]
                for j in range(2 * self.cols):
                    augmented_matrix.data[k][j] -= factor * augmented_matrix.data[i][j]

        inverse_matrix = Matrix(self.rows, self.cols)
        inverse_matrix.data = [row[self.cols :] for row in augmented_matrix.data]

        return inverse_matrix

    def solve_linear_system(self, constants):
        """
        Solve a linear system of equations using the matrix.

        Args:
            constants (list): Constants on the right-hand side of the equations.

        Returns:
            list: Solution to the system of equations.
        """
        coefficients = self.data
        num = len(coefficients)

        for i in range(num):
            max_row = max(range(i, num), key=lambda r, i=i: abs(coefficients[r][i]))
            coefficients[i], coefficients[max_row] = (
                coefficients[max_row],
                coefficients[i],
            )
            constants[i], constants[max_row] = constants[max_row], constants[i]

            pivot = coefficients[i][i]
            for j in range(i, num):
                coefficients[i][j] /= pivot
            constants[i] /= pivot

            for k in range(i + 1, num):
                factor = coefficients[k][i]
                for j in range(i, num):
                    coefficients[k][j] -= factor * coefficients[i][j]
                constants[k] -= factor * constants[i]

        solution = [0] * num
        for i in range(num - 1, -1, -1):
            solution[i] = constants[i]
            for j in range(i + 1, num):
                solution[i] -= coefficients[i][j] * solution[j]

        return solution

    def __str__(self):
        """String representation of the matrix."""
        return "\n".join([" ".join(map(str, row)) for row in self.data])


def test():
    """Test the Matrix class."""

    sample_matrix = Matrix(3, 3)
    sample_matrix.data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    first_norm = sample_matrix.norm("1")
    second_norm = sample_matrix.norm("2")
    infinite_norm = sample_matrix.norm("inf")

    expected_first_norm = 18
    expected_second_norm = math.sqrt(285)
    expected_infinite_norm = 24

    print("\nFirst norm:", first_norm)
    print("Expected first norm:", expected_first_norm)
    print("Second norm:", second_norm)
    print("Expected second norm:", expected_second_norm)
    print("Infinite norm:", infinite_norm)
    print("Expected infinite norm:", expected_infinite_norm)

    assert first_norm == expected_first_norm, "1st norm test failed!"
    assert math.isclose(
        second_norm, expected_second_norm, rel_tol=1e-9
    ), "2nd norm test failed!"
    assert infinite_norm == expected_infinite_norm, "Infinite norm test failed!"

    print("\nNorms test passed!")

    stable_matrix = Matrix(3, 3)
    stable_matrix.data = [[4, 2, 1], [3, 1, 1], [2, 1, 4]]

    unstable_matrix = Matrix(3, 3)
    unstable_matrix.data = [[1, 2, 3], [4, 5, 6], [7, 8, 9.0001]]

    cond_stable = stable_matrix.condition_number()
    cond_unstable = unstable_matrix.condition_number()

    print("Condition number of stable matrix:", cond_stable)
    print("Condition number of unstable matrix:", cond_unstable)

    if cond_unstable > 1000:
        print("Unstable matrix detected: condition number is too large!")
    else:
        print("Stable matrix detected: condition number is within acceptable range.")

    a_matrix = Matrix(3, 3)
    a_matrix.data = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b_vector = [8, -11, -3]

    solution_matrix = a_matrix.solve_linear_system(b_vector)
    print("Solution using Matrix class:", solution_matrix)

    solution_numpy = np.linalg.solve(a_matrix.data, b_vector)
    print("Solution using NumPy:", solution_numpy)

    assert np.allclose(
        solution_matrix, solution_numpy
    ), "Linear system solution mismatch!"

    print("\nLinear system solution test passed!")


test()
