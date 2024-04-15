import numpy as np  # Потрібно для порівняння результатів з numpy

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0] * cols for _ in range(rows)]

    @classmethod
    def from_input(cls):
        rows = int(input("Введіть кількість рядків: "))
        cols = int(input("Введіть кількість стовпців: "))
        matrix = cls(rows, cols)
        print("Введіть елементи матриці:")
        for i in range(rows):
            row = list(map(float, input().split()))
            if len(row) != cols:
                raise ValueError("Кількість елементів у рядку не відповідає кількості стовпців")
            matrix.data[i] = row
        return matrix

    @classmethod
    def identity(cls, n):
        matrix = cls(n, n)
        for i in range(n):
            matrix.data[i][i] = 1
        return matrix

    def add(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Неможливо додати матриці різного розміру")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result

    def scalar_multiply(self, scalar):
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] * scalar
        return result

    def matrix_multiply(self, other):
        if self.cols != other.rows:
            raise ValueError("Неможливо перемножити матриці з такими розмірами")
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result.data[i][j] += self.data[i][k] * other.data[k][j]
        return result

    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    def power(self, n):
        if self.rows != self.cols:
            raise ValueError("Матриця має бути квадратною для піднесення до степеня")
        result = Matrix.identity(self.rows)
        for _ in range(n):
            result = result.matrix_multiply(self)
        return result

    def __str__(self):
        return "\n".join([" ".join(map(str, row)) for row in self.data])

def test():
    A = Matrix(2, 2)
    A.data = [[1, 2], [3, 4]]
    B = Matrix(2, 2)
    B.data = [[5, 6], [7, 8]]

    print("A + B:")
    print(A.add(B))
    print(np.array(A.data) + np.array(B.data))

    print("2 * A:")
    print(A.scalar_multiply(2))
    print(2 * np.array(A.data))

    print("A * B:")
    print(A.matrix_multiply(B))
    print(np.matmul(np.array(A.data), np.array(B.data)))

    print("Транспонована матриця A:")
    print(A.transpose())
    print(np.transpose(np.array(A.data)))

    print("A^2:")
    print(A.power(2))
    print(np.linalg.matrix_power(np.array(A.data), 2))

test()
