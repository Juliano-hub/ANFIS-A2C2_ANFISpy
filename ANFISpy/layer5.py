import torch
import torch
import torch.nn as nn

# Weighted Average Method - ANFIS default 5 layer
class WeightedSumANFISLayer5:
    '''Performs defuzzification using weighted averaging (Takagi-Sugeno Layer 5).'''
    def __init__(self):
        pass

    def __call__(self, w, consequents, dim=1):
        """
        Args:
            w: Normalized weights [Batch, Rules, 1]
            consequents: Rule values ​​[Batch, Rules, Classes]
            dim: Rule dimension (where the sum occurs)
        """
        # Multiplicação elemento a elemento e soma na dimensão das regras
        # w * consequents aproveita o broadcasting do PyTorch
        y_hat = torch.sum(w * consequents, dim=dim, keepdim=True).squeeze()
        return y_hat