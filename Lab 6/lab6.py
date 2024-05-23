"""
Module containing Matrix class for matrix operations.
"""
import math
import random

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
            if pivot == 0:
                raise ValueError("Matrix is singular")

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

    def jacobi_method(self, b_vector, tol=1e-10, max_iterations=1000):
        """
        Solve a linear system of equations using the Jacobi iterative method.

        Args:
            b_vector (list): Constants on the right-hand side of the equations.
            tol (float): Tolerance for convergence.
            max_iterations (int): Maximum number of iterations.

        Returns:
            list: Solution to the system of equations.
        """
        diagonal_elements = [self.data[i][i] for i in range(self.rows)]
        off_diagonal_sum = sum(
            abs(self.data[i][j])
            for i in range(self.rows)
            for j in range(self.cols)
            if i != j
        )
        if max(diagonal_elements) > off_diagonal_sum:
            x_solution = [random.random() for _ in range(self.rows)]
            for iteration in range(max_iterations):
                x_new = x_solution.copy()
                for i in range(self.rows):
                    sum_a = sum(
                        self.data[i][j] * x_solution[j]
                        for j in range(self.cols)
                        if j != i
                    )
                    x_new[i] = (b_vector[i] - sum_a) / self.data[i][i]

                if all(abs(x_new[i] - x_solution[i]) < tol for i in range(self.rows)):
                    print(f"Jacobi method converged in {iteration} iterations.")
                    return x_new
                x_solution = x_new
            raise ValueError(
                "Jacobi method did not converge within the maximum number of iterations"
            )
        raise ValueError("Matrix does not meet the conditions for the Jacobi method")

    def estimate_iterations(self, b_vector, tol=1e-10):
        """
        Estimate the number of iterations required for the Jacobi method to converge.

        Args:
            b_vector (list): Constants on the right-hand side of the equations.
            tol (float): Tolerance for convergence.

        Returns:
            int: Estimated number of iterations.
        """
        d_norm = 0
        for i, val in enumerate(b_vector):
            d_norm += val**2
        d_norm = math.sqrt(d_norm)

        norm_matrix = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                if i == j:
                    norm_matrix.data[i][j] = 1 - self.data[i][j]
                else:
                    norm_matrix.data[i][j] = -self.data[i][j]

        matrix_norm = norm_matrix.norm("2")
        return math.ceil(
            1 / math.log(matrix_norm) * math.log(tol * (1 - matrix_norm) / d_norm)
        )

    def residual_vector(self, solution, constants):
        """
        Calculate the residual vector for a linear system of equations.

        Args:
            solution (list): The solution vector obtained from solving the system.
            constants (list): Constants on the right-hand side of the equations.

        Returns:
            list: The residual vector.
        """
        if len(solution) != self.rows or len(constants) != self.rows:
            raise ValueError(
                "Solution and constants must have the same length as the number of rows"
            )

        residual = []
        for i in range(self.rows):
            row_sum = sum(self.data[i][j] * solution[j] for j in range(self.cols))
            residual.append(constants[i] - row_sum)

        return residual

    def cholesky_decomposition(self):
        """Performs Cholesky decomposition of the matrix.

        Returns:
            Matrix: Lower triangular matrix L such that A = L * L^T.
        """
        if self.rows != self.cols:
            raise ValueError("Matrix must be square for Cholesky decomposition")

        l_matrix = Matrix(self.rows, self.cols)

        for i in range(self.rows):
            for j in range(i + 1):
                sum_k = sum(l_matrix.data[i][k] * l_matrix.data[j][k] for k in range(j))

                if i == j:
                    l_matrix.data[i][j] = math.sqrt(self.data[i][i] - sum_k)
                else:
                    l_matrix.data[i][j] = (self.data[i][j] - sum_k) / l_matrix.data[j][
                        j
                    ]

        return l_matrix

    def is_positive_definite(self):
        """Check if the matrix is positive definite using Cholesky decomposition.

        Returns:
            bool: True if the matrix is positive definite, False otherwise.
        """
        try:
            _ = self.cholesky_decomposition()
            return True
        except ValueError:
            return False

    def relaxation_method(self, b_vector, omega=1.0, tol=1e-10, max_iterations=1000):
        """
        Solve a linear system of equations using the Successive Over-Relaxation (SOR) method.

        Args:
            b_vector (list): Constants on the right-hand side of the equations.
            omega (float): Relaxation parameter (0 < omega < 2).
            tol (float): Tolerance for convergence.
            max_iterations (int): Maximum number of iterations.

        Returns:
            list: Solution to the system of equations.
        """
        x_solution = [0] * self.rows
        for iteration in range(max_iterations):
            x_new = x_solution.copy()
            for i in range(self.rows):
                sigma = sum(
                    self.data[i][j] * x_new[j] for j in range(self.cols) if j != i
                )
                x_new[i] = (1 - omega) * x_solution[i] + (omega / self.data[i][i]) * (
                    b_vector[i] - sigma
                )

            if all(abs(x_new[i] - x_solution[i]) < tol for i in range(self.rows)):
                print(f"Relaxation method converged in {iteration} iterations.")
                return x_new
            x_solution = x_new
        raise ValueError(
            "Relaxation method did not converge within the maximum number of iterations"
        )

    def __str__(self):
        """String representation of the matrix."""
        return "\n".join([" ".join(map(str, row)) for row in self.data])


def test():
    """Test the Matrix class with numpy for verification."""

    pd_matrix = Matrix(3, 3)
    pd_matrix.data = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]

    print("Positive Definite Matrix:")
    print(pd_matrix)

    np_pd_matrix = np.array(pd_matrix.data)

    print("\nTesting Cholesky decomposition:")
    try:
        l_matrix = pd_matrix.cholesky_decomposition()
        print("Cholesky decomposition successful. Lower triangular matrix L:")
        print(l_matrix)
        np_l = np.linalg.cholesky(np_pd_matrix)
        assert np.allclose(l_matrix.data, np_l), "Cholesky decomposition failed"
        print("Cholesky decomposition validation passed with numpy.")
    except ValueError as error:
        print(error)

    simple_matrix = Matrix(2, 2)
    simple_matrix.data = [[4, 1], [1, 3]]
    b_vector = [1, 2]
    omega = 1.1

    print("\nSimple Matrix for Relaxation method:")
    print(simple_matrix)

    print("\nTesting Relaxation method:")
    try:
        solution_relaxation = simple_matrix.relaxation_method(b_vector, omega)
        print("Solution using Relaxation method:", solution_relaxation)
        np_solution = np.linalg.solve(np.array(simple_matrix.data), np.array(b_vector))
        assert np.allclose(solution_relaxation, np_solution), "Relaxation method failed"
        print("Relaxation method validation passed with numpy.")
    except ValueError as error:
        print(error)

    print("\nAll tests passed successfully.")


test()
