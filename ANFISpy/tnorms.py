import torch

class MinAND:
    '''Calculates the minimum t-norm along a given dimension.'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        return torch.min(x, dim=dim).values
    
class ProdAND:
    '''Calculates the product t-norm along a given dimension.'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        return torch.prod(x, dim=dim)

class FrankAND:
    '''Calculates the Frank t-norm along a given dimension.
    
    Args:
        p: float for parameter p. It should be positive and different than 1.
    '''
    def __init__(self, p=2):
        self.log_p = torch.log(torch.tensor(p))

    def __call__(self, x, dim):
        p = torch.exp(self.log_p).to(x.device)
        arg = 1 + torch.prod(p**x - 1, dim=dim) / (p - 1)
        return torch.log(arg) / self.log_p.to(x.device)

class HamacherAND:
    '''Calculates the Frank t-norm along a given dimension.
    
    Args:
        r: float for parameter r. It should be positive.
    '''
    def __init__(self, r=1):
        self.r = r

    def __call__(self, x, dim):
        prod_x = torch.prod(x, dim=dim)
        sum_x = torch.sum(x, dim=dim)
        return prod_x / (self.r + (1 - self.r) * (sum_x - prod_x))

class LukasiewiczAND:
    '''Calculates the Lukasiewicz t-norm along a given dimension.'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        n = x.shape[dim]
        sum_x = torch.sum(x, dim=dim)
        out = sum_x - (n - 1)
        return torch.clamp(out, min=0)
    
class SugenoWeberAND:
    '''Calculates the Sugeno-Weber t-norm family.'''
    def __init__(self, lmbda=0.0):
        self.lmbda = lmbda

    def __call__(self, x, dim):
        l = self.lmbda
        
        # ---- Caso 1: λ = -1 (Drastic Product) ----
        if l == -1.0:
            val_min, _ = torch.min(x, dim=dim)
            count_less_than_one = torch.sum((x < 1.0).float(), dim=dim)
            return torch.where(count_less_than_one <= 1, val_min, torch.zeros_like(val_min))

        # ---- Caso 2: λ = ∞ (Algebraic Product) ----
        elif l == float('inf'):
            return torch.prod(x, dim=dim)

        # ---- Caso 3: Otherwise ----
        else:
            res = x.select(dim, 0)
            for i in range(1, x.shape[dim]):
                a = res
                b = x.select(dim, i)
                
                # max((x + y - 1 + λxy) / (1 + λ), 0)
                val = (a + b - 1 + l * a * b) / (1 + l)
                res = torch.clamp(val, min=0.0)
            return res

class SugenoWeberOR:
    '''Calculates the Sugeno-Weber t-conorm family.'''
    def __init__(self, lmbda=0.0):
        self.lmbda = lmbda

    def __call__(self, x, dim):
        l = self.lmbda
        
        # ---- Caso 1: λ = -1 (Algebraic Sum / Probabilistic Sum) ----
        if l == -1.0:
            res = x.select(dim, 0)
            for i in range(1, x.shape[dim]):
                res = res + x.select(dim, i) - (res * x.select(dim, i))
            return res

        # ---- Caso 2: λ = ∞ (Drastic Sum) ----
        elif l == float('inf'):
            val_max, _ = torch.max(x, dim=dim)
            count_greater_than_zero = torch.sum((x > 0.0).float(), dim=dim)
            return torch.where(count_greater_than_zero <= 1, val_max, torch.ones_like(val_max))

        # ---- Caso 3: Otherwise ----
        else:
            res = x.select(dim, 0)
            for i in range(1, x.shape[dim]):
                a = res
                b = x.select(dim, i)
                
                # min(x + y + λxy, 1)
                val = a + b + l * a * b
                res = torch.clamp(val, max=1.0)
            return res