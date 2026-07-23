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
    
class WeightedQuadratic():
    '''Performs defuzzification using weighted quadratic averaging (RMS).'''
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, w, consequents, dim=1):
        # y_hat = sqrt(sum(w * x^2))
        quadratic_term = w * (consequents ** 2)
        y_hat = torch.sqrt(torch.sum(quadratic_term, dim=dim, keepdim=True) + self.eps)
        return y_hat.squeeze(dim)

class WeightedGeometric():
    '''Performs defuzzification using weighted geometric averaging.'''
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, w, consequents, dim=1):
        # y_hat = exp(sum(w * log(x)))
        # Clamp para evitar log(0) ou log de números negativos
        pos_consequents = torch.clamp(consequents, min=self.eps)
        log_term = w * torch.log(pos_consequents)
        y_hat = torch.exp(torch.sum(log_term, dim=dim, keepdim=True))
        return y_hat.squeeze(dim)

class WeightedHarmonic():
    '''Performs defuzzification using weighted harmonic averaging.'''
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, w, consequents, dim=1):
        # y_hat = 1 / sum(w / x)
        # Evita divisão por zero substituindo 0 por eps
        safe_consequents = torch.where(consequents == 0, torch.tensor(self.eps, device=consequents.device), consequents)
        harmonic_term = w / safe_consequents
        y_hat = 1.0 / (torch.sum(harmonic_term, dim=dim, keepdim=True) + self.eps)
        return y_hat.squeeze(dim)

class WeightedRootPower():
    '''Performs defuzzification using weighted root mean power averaging.'''
    def __init__(self, p=3, eps=1e-8):
        self.p = p
        self.eps = eps

    def __call__(self, w, consequents, dim=1):
        # y_hat = (sum(w * x^p))^(1/p)
        # Se p for ímpar ou fracionário, bases negativas quebram a potência. Clamp preventivo:
        pos_consequents = torch.clamp(consequents, min=self.eps) if self.p % 2 != 0 else consequents
        power_term = w * (pos_consequents ** self.p)
        sum_power = torch.sum(power_term, dim=dim, keepdim=True)
        y_hat = torch.pow(sum_power + self.eps, 1.0 / self.p)
        return y_hat.squeeze(dim)
    

class WeightedExponential():
    '''Performs defuzzification using weighted exponential averaging.'''
    def __init__(self, alpha=0.1, eps=1e-8):
        self.alpha = alpha
        self.eps = eps

    def __call__(self, w, consequents, dim=1):
        # y_hat = (1/alpha) * log(sum(w * exp(alpha * x)))
        # Truque de estabilização numérica subtraindo o valor máximo (evita overflow do exp)
        max_c = torch.max(consequents, dim=dim, keepdim=True)[0]
        exp_term = w * torch.exp(self.alpha * (consequents - max_c))
        sum_exp = torch.sum(exp_term, dim=dim, keepdim=True)
        y_hat = max_c + (1.0 / self.alpha) * torch.log(sum_exp + self.eps)
        return y_hat.squeeze(dim)

#-----------------------------------
class WeightedEuclidean:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, w, consequents, dim=1):
        squared_consequents = consequents ** 2
        
        weighted_sum = torch.sum(w * squared_consequents, dim=dim, keepdim=True)
        
        y_hat = torch.sqrt(weighted_sum + self.eps)
        
        return y_hat.squeeze()

class OWA:
    """
    OWA aggregation over ANFIS rule outputs.

    OWA types:
        - "max"
        - "min"
        - "mean"
    """

    def __init__(self, owa_type="max"):
        self.owa_type = owa_type.lower()

    def _build_weights(self, n_rules, device, dtype):

        # ---------------- MAX
        if self.owa_type == "max":
            weights = torch.zeros(
                n_rules,
                device=device,
                dtype=dtype
            )

            weights[0] = 1.0

        # ---------------- MIN
        elif self.owa_type == "min":
            weights = torch.zeros(
                n_rules,
                device=device,
                dtype=dtype
            )

            weights[-1] = 1.0

        # ---------------- MEAN
        elif self.owa_type == "mean":
            weights = torch.ones(
                n_rules,
                device=device,
                dtype=dtype
            ) / n_rules

        else:
            raise ValueError(
                f"Unknown OWA type: {self.owa_type}"
            )

        return weights

    def __call__(self, w, consequents, dim=1):

        """
        Args:
            w:
                Rule activations
                Shape -> [Batch, Rules, 1]

            consequents:
                Rule consequents
                Shape -> [Batch, Rules, Classes]

            dim:
                Rule dimension
        """

        weighted_rules = w * consequents

        # Sort weighted rules
        sorted_vals, sorted_indices = torch.sort(
            weighted_rules,
            dim=dim,
            descending=True
        )

        # Cria os pesos
        n_rules = consequents.shape[dim]

        owa_weights = self._build_weights(
            n_rules,
            consequents.device,
            consequents.dtype
        )

        # Formata os dados
        owa_weights = owa_weights.view(1, -1, 1)

        # Aplica os pesos
        weighted_sorted = sorted_vals * owa_weights

        # Agregação
        y_hat = torch.sum(
            weighted_sorted,
            dim=dim,
            keepdim=True
        ).squeeze()

        return y_hat