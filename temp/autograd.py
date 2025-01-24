import numpy as np


class Variable:
    def __init__(self, value=0):
        self._value = value
        self.grad = 0

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def forward(self):
        return self.value

    def backward(self):
        self.do_backward(1)

    def do_backward(self, grad):
        self.grad += grad

    def __add__(self, other):
        other: Variable = other
        result = Variable()

        def forward_inner():
            result._value = other.forward() + self.forward()
            return result._value

        def backward_inner(grad):
            result.grad += grad
            self.do_backward(grad)
            other.do_backward(grad)

        result.forward = forward_inner
        result.do_backward = backward_inner
        return result

    def __sub__(self, other):
        other: Variable = other
        result = Variable()

        def forward_inner():
            result._value = self.forward() - other.forward()
            return result._value

        def backward_inner(grad):
            result.grad += grad
            self.do_backward(grad)
            other.do_backward(-grad)

        result.forward = forward_inner
        result.do_backward = backward_inner
        return result

    def __mul__(self, other):
        other: Variable = other
        result = Variable()

        def forward_inner():
            result._value = other.forward() * self.forward()
            return result._value

        def backward_inner(grad):
            result.grad += grad
            self.do_backward(grad * other.value)
            other.do_backward(grad * self.value)

        result.forward = forward_inner
        result.do_backward = backward_inner
        return result

    def __truediv__(self, other):
        return (other ** -1) * self

    def __pow__(self, other):
        result = Variable()

        if isinstance(other, (int, float)):
            def forward_inner():
                result._value = self.forward() ** other
                return result._value

            def backward_inner(grad):
                result.grad += grad
                self.do_backward(grad * (other * (self.value ** (other - 1))))

        elif isinstance(other, Variable):
            def forward_inner():
                result._value = self.forward() ** other.forward()
                return result._value

            def backward_inner(grad):
                result.grad += grad
                self.do_backward(grad * (other.value * (self.value ** (other.value - 1))))
                other.do_backward(grad * (np.log(self.value) * result.value))

        else:
            raise Exception
        result.forward = forward_inner
        result.do_backward = backward_inner
        return result

def sin(v: Variable):
    result = Variable()

    def forward_inner():
        result.value = np.sin(v.forward())
        return result.value

    def backward_inner(grad):
        result.grad += grad
        v.do_backward(grad * np.cos(v.value))

    result.forward = forward_inner
    result.do_backward = backward_inner
    return result


def cos(v: Variable):
    result = Variable()

    def forward_inner():
        result.value = np.cos(v.forward())
        return result.value

    def backward_inner(grad):
        result.grad += grad
        v.do_backward(-grad * np.sin(v.value))

    result.forward = forward_inner
    result.do_backward = backward_inner
    return result
