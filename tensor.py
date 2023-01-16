import numpy as np
from copy import deepcopy

class Tensor(np.ndarray):
    def __new__(cls, input):
        obj = np.array(input).astype(np.float64).view(cls)
        return obj

    def __init__(self, _) -> None:
        self.grad = None
        self._prev = None
        self._op = None
        self.label = None

        self._backward = lambda: None
    
    def set_attributes(self, _prev, _op):
        self._prev = _prev # supposed to be set but Tensor is not hashable yet
        self._op = _op

        return self

    def __add__(self, other):
        out = super().__add__(other).set_attributes((self, other), '+')
        
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward
        
        return out 

    def __matmul__(self, other):
        out = super().__matmul__(other).set_attributes((self, other), '@')
        return out

    def relu(self):
        out = (self * (self > 0).astype(np.float64)).set_attributes((self,), 'relu')

        def _backward():
            g = self * (self > 0).astype(np.float64)
            self.grad = g * out.grad
        out._backward = _backward

        return out

    def backward(self):
        pass

def main():
    W = Tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])
    x = Tensor([1, 1, 1])
    b = Tensor([1, 1])

    y = W @ x
    y = y + b
    y = y.relu()

    a=1

if __name__ == '__main__':
    main()