import numpy as np

class Tensor(np.ndarray):
    def __new__(cls, input):
        obj = np.array(input).view(cls)
        return obj

    def __add__(self, other):
        return super().__add__(other)

    def __sub__(self, other):
        return super().__sub__(other)

    def __mul__(self, other):
        return super().__mul__(other)

    def __div__(self, other):
        return super().__mul__(other)

    def __matmul__(self, other):
        return super().__matmul__(other)

def main():
    t = Tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    a=1

if __name__ == '__main__':
    main()