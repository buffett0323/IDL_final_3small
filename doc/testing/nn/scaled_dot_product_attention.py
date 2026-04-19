import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax()
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        self.E = Q.shape[-1]
        self.Q, self.K, self.V = Q, K, V
        
        # Last two dims only: (..., L, E) @ (..., E, S) -> (..., L, S); K.T would transpose all axes
        K_T = np.swapaxes(self.K, -2, -1)
        scaled_dot_product = (self.Q @ K_T) / np.sqrt(self.E)

        if mask is not None:
            scaled_dot_product = np.where(mask, -self.eps, scaled_dot_product)

        self.attention_scores = self.softmax.forward(scaled_dot_product)
        output = self.attention_scores @ V
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        P = self.attention_scores
        scale = np.sqrt(self.E)

        d_V = np.swapaxes(P, -2, -1) @ d_output
        d_P = d_output @ np.swapaxes(self.V, -2, -1)
        d_logits = self.softmax.backward(d_P)

        d_Q = (d_logits @ self.K) / scale
        d_K = (np.swapaxes(d_logits, -2, -1) @ self.Q) / scale

        return d_Q, d_K, d_V
