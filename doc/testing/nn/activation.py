import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        ndim = len(Z.shape)
        if self.dim < -ndim or self.dim >= ndim:
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")

        Z_max = np.max(Z, axis=self.dim, keepdims=True)
        Z_shifted = Z - Z_max
        exp_Z = np.exp(Z_shifted)
        self.A = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # dLdZ = A * (dLdA - sum(dLdA * A, dim)) — same as per-row Jacobian @ dLdA, vectorized
        inner = np.sum(dLdA * self.A, axis=self.dim, keepdims=True)
        return self.A * (dLdA - inner)
