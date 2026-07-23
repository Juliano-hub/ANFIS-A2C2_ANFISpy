import torch

#==================== Agregadores conjuntivos e disjuntivos ===================================

class ProdAND:
    '''Calculates the product t-norm along a given dimension.'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        x_safe = torch.clamp(x, min=1e-6, max=1e2)
        return torch.prod(x_safe, dim=dim)

class ProdOR:
    '''Calculates the product t-conorm along a given dimension.'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        x_safe = torch.clamp(1 - x, min=1e-6, max=1e2)
        return 1 - torch.prod(x_safe, dim=dim)

class LukasiewiczAND:
    '''Calculates the Lukasiewicz t-norm along a given dimension.'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        n = x.shape[dim]
        sum_x = torch.sum(x, dim=dim)
        out = sum_x - (n - 1)
        return torch.clamp(out, min=0)

class LukasiewiczOR:
    '''Calculates the Lukasiewicz t-conorm along a given dimension.'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        n = x.shape[dim]
        sum_x = torch.sum(x, dim=dim)
        return torch.clamp(sum_x, max=1)

class MinAND:
    '''Calculates the minimum t-norm along a given dimension.'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        return torch.min(x, dim=dim).values

class MaxOR:
    '''Calculates the maximum t-conorm along a given dimension.'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        return torch.max(x, dim=dim).values

class FrankAND:
    '''Calculates the Frank t-norm along a given dimension.
    Args:
        p: float for parameter p. It should be positive and different than 1.'''
    def __init__(self, p=2.0):
        self.p = p

    def __call__(self, x, dim):
        p = float(self.p) if torch.is_tensor(self.p) else self.p
        self.log_p = torch.log(torch.tensor(p, device=x.device, dtype=x.dtype))

        # ---- Case 1: p == 0  → minimum
        if p == 0:
            return torch.min(x, dim=dim).values

        # ---- Case 2: p == 1 → product
        elif p == 1:
            return torch.prod(x, dim=dim)

        # ---- Case 3: p == +inf → Luka
        elif p == float("inf"):
            sum_x = torch.sum(x, dim=dim)
            n = x.size(dim)
            return torch.clamp(sum_x - (n - 1), min=0)

        else:
            eps = 1e-6
            res = x.select(dim, 0)
            
            p_safe = torch.clamp(torch.tensor(p, device=x.device, dtype=x.dtype), min=eps)
            
            if torch.abs(p_safe - 1.0) < eps:
                p_safe = p_safe + (eps if p_safe >= 1.0 else -eps)

            for i in range(1, x.size(dim)):

                a = torch.clamp(res, min=eps, max=1.0 - eps)
                b = torch.clamp(x.select(dim, i), min=eps, max=1.0 - eps)
                
                term_a = torch.pow(p_safe + eps, a) - 1.0
                term_b = torch.pow(p_safe + eps, b) - 1.0
                num = term_a * term_b
                

                den = p_safe - 1.0
                den = den + eps if den >= 0 else den - eps
                
                den2 = torch.log(p_safe + eps)
                den2 = den2 + eps if den2 >= 0 else den2 - eps
                
                # mudança de base, log_base(p)x = log(x) / log(p)
                argumento_log = 1.0 + torch.clamp(num / den, min=0.0)
                
                res = torch.log(argumento_log + eps) / den2
                
            return res

class FrankOR:
    '''Calculates the Frank t-conorm along a given dimension.
    Args:
        p: float for parameter p. It should be positive and different than 1.'''
    def __init__(self, p=2.0):
        self.p = p

    def __call__(self, x, dim):
        p = float(self.p) if torch.is_tensor(self.p) else self.p
        self.log_p = torch.log(torch.tensor(p, device=x.device, dtype=x.dtype))

        p_t = torch.tensor(p, device=x.device, dtype=x.dtype)

        # ---- Case 1: p == 0  → maximum
        if p == 0:
            return torch.max(x, dim=dim).values

        # ---- Case 2: p == 1 → tconorm prod
        elif p == 1:
            return 1 - torch.prod(1-x, dim=dim)

        # ---- Case 3: p == +inf → Luka OR
        elif p == float("inf"):
            sum_x = torch.sum(x, dim=dim)
            return torch.clamp(sum_x, max=1)

        else:
            eps = 1e-6
            res = x.select(dim, 0)
            
            p_safe = torch.clamp(torch.tensor(p, device=x.device, dtype=x.dtype), min=eps)
            
            if torch.abs(p_safe - 1.0) < eps:
                p_safe = p_safe + (eps if p_safe >= 1.0 else -eps)

            for i in range(1, x.size(dim)):

                a = torch.clamp(res, min=eps, max=1.0 - eps)
                b = torch.clamp(x.select(dim, i), min=eps, max=1.0 - eps)
                
                term_a = torch.pow(p_safe + eps, (1.0-a)) - 1.0
                term_b = torch.pow(p_safe + eps, (1.0-b)) - 1.0
                num = term_a * term_b
                

                den = p_safe - 1.0
                den = den + eps if den >= 0 else den - eps
                
                den2 = torch.log(p_safe + eps)
                den2 = den2 + eps if den2 >= 0 else den2 - eps
                
                # mudança de base, log_base(p)x = log(x) / log(p)
                argumento_log = 1.0 + torch.clamp(num / den, min=0.0)
                
                res = 1.0 - (torch.log(argumento_log + eps) / den2)
                
            return res

class DrasticAND:
    '''
    Calculates the Drastic Product t-norm (Eq. 3.1).
    T(x1,...,xn) = 0 if all xi < 1, else min(xi).
    '''
    def __call__(self, x, dim):
        val_min, _ = torch.min(x, dim=dim)
        
        all_less_than_one = torch.all(x < 1.0, dim=dim)
        
        return torch.where(all_less_than_one,
                           torch.zeros_like(val_min),
                           val_min)

class DrasticOR:
    '''
    Calculates the Drastic Sum t-conorm (Eq. 3.2).
    S(x1,...,xn) = 1 if all xi > 0, else max(xi).
    '''
    def __call__(self, x, dim):
        val_max, _ = torch.max(x, dim=dim)
        
        all_greater_than_zero = torch.all(x > 0.0, dim=dim)
        
        return torch.where(all_greater_than_zero,
                           torch.ones_like(val_max),
                           val_max)

class YagerAND:
    '''Calculates the Yager t-norm along a given dimension.
    
    Args:
        p: float > 0
    '''
    def __init__(self, p=2.0):
        self.p = p

    def __call__(self, x, dim):
        p = float(self.p) if torch.is_tensor(self.p) else self.p
        p_t = torch.tensor(p, device=x.device, dtype=x.dtype)

        # ---- case limite: p → ∞ → mínimo
        if p == float("inf"):
            return torch.min(x, dim=dim).values

        elif p == 0:
            val_min, _ = torch.min(x, dim=dim)
            
            count_less_than_one = torch.sum((x < 1.0).float(), dim=dim)
            
            result = torch.where(count_less_than_one <= 1, val_min, torch.zeros_like(val_min))
            return result
        
        # ---- otherwise
        else:
            # 1. Protege a base (1 - x) para não ser negativa ou zero absoluto antes da potência p_t
            base_term = torch.clamp(1 - x, min=1e-6)
            term = torch.sum(base_term**p_t, dim=dim)
            
            den = p_t
            
            # 2. Protege a base 'term' antes de aplicar o expoente fracionário (1 / den)
            term_protected = torch.clamp(term, min=1e-6)
            
            out = 1 - term_protected**(1 / (den + 1e-6))
            return torch.clamp(out, min=0)
        
class YagerOR:
    '''Calculates the Yager t-conorm along a given dimension.
    
    Args:
        p: float > 0
    '''
    def __init__(self, p=2.0):
        self.p = p

    def __call__(self, x, dim):
        p = float(self.p) if torch.is_tensor(self.p) else self.p
        p_t = torch.tensor(p, device=x.device, dtype=x.dtype)

        # ---- case limite: p → ∞ → mínimo
        if p == float("inf"):
            return torch.max(x, dim=dim).values

        elif p == 0:
            val_max, _ = torch.max(x, dim=dim)
            
            count_greater_than_zero = torch.sum((x > 0.0).float(), dim=dim)
            
            result = torch.where(count_greater_than_zero <= 1, val_max, torch.ones_like(val_max))
            return result

        # ---- otherwise
        else:
            # 1. Protege a base 'x' para evitar zero absoluto antes de elevar a p_t
            x_protected = torch.clamp(x, min=1e-6)
            term = torch.sum(x_protected**p_t, dim=dim)
            
            den = p_t
            
            # 2. Protege o 'term' para evitar zero antes do expoente fracionário
            term_protected = torch.clamp(term, min=1e-6)
            out = term_protected**(1 / (den + 1e-6))
            
            return torch.clamp(out, max=1)

class HamacherAND:
    ''' 
    Calculates the Hamacher t-norm family (Eq. 3.1). 
    Args: 
    p: The parameter lambda (λ). 
    p = 0 -> Algebraic Product 
    p = 1 -> Hamacher Product 
    p = inf -> Drastic Product 
    '''
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, x, dim):
        p = float(self.p) if torch.is_tensor(self.p) else self.p

        # λ = ∞ → Drástico
        if p == float('inf'):
            val_min, _ = torch.min(x, dim=dim)
            all_less_than_one = torch.all(x < 1.0, dim=dim)
            return torch.where(all_less_than_one,
                               torch.zeros_like(val_min),
                               val_min)

        # λ = 0 → Produto algébrico
        elif p == 0.0:
            return torch.prod(x, dim=dim)

        # Caso geral → redução sequencial
        else:
            res = x.select(dim, 0)

            for i in range(1, x.size(dim)):
                b = x.select(dim, i)
                a = res

                num = a * b
                den = p + (1 - p) * (a + b - num)
                res = num / (den+1e-6)

            return res

class HamacherOR:
    ''' 
    Calculates the Hamacher t-conorm family (Eq. 3.1). 
    Args: 
    p: The parameter lambda (λ). 
    p = 0 -> Algebraic Product 
    p = 1 -> Hamacher Product 
    p = inf -> Drastic Product 
    '''
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, x, dim):
        p = float(self.p) if torch.is_tensor(self.p) else self.p

        # λ = ∞ → Drástico
        if p == float('inf'):
            val_max, _ = torch.max(x, dim=dim)
            all_greater_than_zero = torch.all(x > 0.0, dim=dim)
            return torch.where(all_greater_than_zero,
                               torch.ones_like(val_max),
                               val_max)

        elif p == 0.0:
            res = x.select(dim, 0)
            for i in range(1, x.size(dim)):
                b = x.select(dim, i)
                res = res + b - res * b
            return res
        
        else:
            res = x.select(dim, 0)

            for i in range(1, x.size(dim)):
                b = x.select(dim, i)
                a = res

                num = a + b - a*b - (1 - p)*(a*b)
                den = 1 - (1 - p)*(a*b)
                res = num / (den+1e-6)

            return res
    
class DombiAND:
    '''Calculates the Dombi t-norm family.'''
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, x, dim):
        p = float(self.p) if torch.is_tensor(self.p) else self.p
        
        # ---- case 1: λ = 0 (Drastic Product) ----
        if p == 0.0:
            val_min, _ = torch.min(x, dim=dim)
            count_less_than_one = torch.sum((x < 1.0).float(), dim=dim)
            return torch.where(count_less_than_one <= 1, val_min, torch.zeros_like(val_min))

        # ---- case 2: λ = ∞ (Minimum) ----
        elif p == float('inf'):
            return torch.min(x, dim=dim).values

        else:
            res = x.select(dim, 0)
        
            for i in range(1, x.shape[dim]):
                a = torch.clamp(res, min=1e-6, max=1 - 1e-6)
                b = torch.clamp(x.select(dim, i), min=1e-6, max=1 - 1e-6)
                
                term_a = ((1.0 - a) / a) ** p
                term_b = ((1.0 - b) / b) ** p
                
                den = 1 + (term_a + term_b + 1e-6) ** (1 / p)
                
                # Atualiza res para a próxima iteração
                res = 1 - (1.0 / (den + 1e-6))
        return res


class DombiOR:
    '''Calculates the Dombi t-conorm family.'''
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, x, dim):
        p = float(self.p) if torch.is_tensor(self.p) else self.p
        
        # ---- case 1: λ = 0 (Drastic Sum) ----
        if p == 0.0:
            val_max, _ = torch.max(x, dim=dim)
            count_greater_than_zero = torch.sum((x > 0.0).float(), dim=dim)
            return torch.where(count_greater_than_zero <= 1, val_max, torch.ones_like(val_max))

        # ---- case 2: λ = ∞ (Maximum) ----
        elif p == float('inf'):
            return torch.max(x, dim=dim).values

        # ---- otherwise
        else:
            res = x.select(dim, 0)
        
            for i in range(1, x.shape[dim]):
                a = torch.clamp(res, min=1e-6, max=1 - 1e-6)
                b = torch.clamp(x.select(dim, i), min=1e-6, max=1 - 1e-6)
                
                term_a = (a / (1.0 - a)) ** p
                term_b = (b / (1.0 - b)) ** p
                
                den = 1 + (term_a + term_b + 1e-6) ** (1 / p)
                
                # Atualiza res para a próxima iteração
                res = 1 - (1.0 / (den + 1e-6))
        return res
        
class SugenoWeberAND:
    '''Calculates the Sugeno-Weber t-norm family.'''

    def __init__(self, p=0.0):
        self.p = p

    def __call__(self, x, dim):

        p = float(self.p) if torch.is_tensor(self.p) else self.p

        if p == -1.0:

            val_min, _ = torch.min(x, dim=dim)

            count_less_than_one = torch.sum(
                (x < 1.0).float(),
                dim=dim
            )

            return torch.where(
                count_less_than_one <= 1,
                val_min,
                torch.zeros_like(val_min)
            )

        elif p == float('inf'):

            return torch.prod(x, dim=dim)

        else:

            res = x.select(dim, 0)

            for i in range(1, x.shape[dim]):

                b = x.select(dim, i)

                num = res + b - 1.0 + p * res * b
                den = 1.0 + p

                res = torch.maximum(
                    num / (den + 1e-6),
                    torch.zeros_like(res)
                )

            return res
        
class SugenoWeberOR:
    '''Calculates the Sugeno-Weber t-conorm family.'''

    def __init__(self, p=0.0):
        self.p = p

    def __call__(self, x, dim):

        p = float(self.p) if torch.is_tensor(self.p) else self.p

        # ---- λ = -1 (Probabilistic Sum) ----
        if p == -1.0:

            res = x.select(dim, 0)

            for i in range(1, x.shape[dim]):

                b = x.select(dim, i)

                res = res + b - res * b

            return res

        # ---- λ = ∞ (Drastic Sum) ----
        elif p == float('inf'):

            res = x.select(dim, 0)

            for i in range(1, x.shape[dim]):

                b = x.select(dim, i)

                res = torch.where(
                    (res == 0) | (b == 0),
                    torch.maximum(res, b),
                    torch.ones_like(res)
                )

            return res

        else:

            res = x.select(dim, 0)

            for i in range(1, x.shape[dim]):

                b = x.select(dim, i)

                res = torch.minimum(
                    res + b + p * res * b,
                    torch.ones_like(res)
                )

            return res

class SchweizerSklarAND:
    '''Calculates the Schweizer-Sklar t-norm family.'''
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, x, dim):
        p = float(self.p) if torch.is_tensor(self.p) else self.p
        
        # ---- case 1: λ = -∞ (Minimum) ----
        if p == float('-inf'):
            return torch.min(x, dim=dim).values

        # ---- case 2: λ = 0 (Algebraic Product) ----
        elif p == 0.0:
            return torch.prod(x, dim=dim)

        # ---- case 3: λ = ∞ (Drastic Product) ----
        elif p == float('inf'):
            val_min, _ = torch.min(x, dim=dim)
            count_less_than_one = torch.sum((x < 1.0).float(), dim=dim)
            return torch.where(count_less_than_one <= 1, val_min, torch.zeros_like(val_min))

        # ---- otherwise
        else:
            eps = 1e-6
            res = x.select(dim, 0)
            
            for i in range(1, x.shape[dim]):
                # 1. Blindagem de entrada: Garante que a e b não tragam problemas cumulativos
                a = torch.clamp(res, min=eps, max=1.0-eps)
                b = torch.clamp(x.select(dim, i), min=eps, max=1.0-eps)
                
                # 2. Estabiliza o cálculo das potências
                # Adicionamos eps na base para evitar base zero se p for muito grande
                sum_pow = torch.pow(a + eps, p) + torch.pow(b + eps, p) - 1.0
                
                # 3. O Pulo do Gato: Blindagem do Clamp
                # Substituímos o min=0.0 por min=eps. 
                # Isso impede que a base da potência externa seja zero, salvando o gradiente de explodir (NaN) ou zerar.
                base_estavel = torch.clamp(sum_pow, min=eps)
                
                # 4. Cálculo do expoente (1/p) de forma segura
                den = 1.0 / (p + eps)
                
                # 5. Atualiza res para a próxima iteração
                res = torch.pow(base_estavel, den)
                
            return res
        
class SchweizerSklarOR:
    '''Calculates the Schweizer-Sklar t-conorm family (Eq. in Image).'''
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, x, dim):
        p = float(self.p) if torch.is_tensor(self.p) else self.p
        
        # ---- case 1: λ = -∞ (Maximum) ----
        if p == float('-inf'):
            return torch.max(x, dim=dim).values

        # ---- case 2: λ = 0 (Algebraic Sum / Probabilistic) ----
        elif p == 0.0:
            res = x.select(dim, 0)
            for i in range(1, x.shape[dim]):
                res = res + x.select(dim, i) - (res * x.select(dim, i))
            return res

        # ---- case 3: λ = ∞ (Drastic Sum) ----
        elif p == float('inf'):
            val_max, _ = torch.max(x, dim=dim)
            count_greater_than_zero = torch.sum((x > 0.0).float(), dim=dim)
            return torch.where(count_greater_than_zero <= 1, val_max, torch.ones_like(val_max))

        # ---- otherwise
        else:
            eps = 1e-6
            res = x.select(dim, 0)
            
            for i in range(1, x.shape[dim]):
                # 1. Blindagem de entrada: Garante que a e b não tragam problemas cumulativos
                a = torch.clamp(res, min=eps, max=1.0 - eps)
                b = torch.clamp(x.select(dim, i), min=eps, max=1.0 - eps)
                
                # 2. Estabiliza o cálculo das potências
                # Adicionamos eps na base para evitar base zero se p for muito grande
                sum_pow = torch.pow((1-a) + eps, p) + torch.pow((1-b) + eps, p) - 1.0
                
                # 3. O Pulo do Gato: Blindagem do Clamp
                # Substituímos o min=0.0 por min=eps. 
                # Isso impede que a base da potência externa seja zero, salvando o gradiente de explodir (NaN) ou zerar.
                base_estavel = torch.clamp(sum_pow, min=eps)
                
                # 4. Cálculo do expoente (1/p) de forma segura
                den = 1.0 / (p + eps)
                
                # 5. Atualiza res para a próxima iteração
                res = 1 - torch.pow(base_estavel, den)
                
            return res

class AczelAlsinaAND:
    '''Calculates the Aczél-Alsina t-norm family (Eq. in Image).'''
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, x, dim):
        p = float(self.p) if torch.is_tensor(self.p) else self.p
        
        # ---- case 1: λ = 0 (Drastic Product) ----
        if p == 0.0:
            val_min, _ = torch.min(x, dim=dim)
            count_less_than_one = torch.sum((x < 1.0).float(), dim=dim)
            return torch.where(count_less_than_one <= 1, val_min, torch.zeros_like(val_min))

        # ---- case 2: λ = ∞ (Minimum) ----
        elif p == float('inf'):
            return torch.min(x, dim=dim).values

        else:
            res = x.select(dim, 0)
            eps = 1e-6 
            
            for i in range(1, x.shape[dim]):
                a = torch.clamp(res, min=eps, max=1.0 - eps)
                b = torch.clamp(x.select(dim, i), min=eps, max=1.0 - eps)
                
                # ---> BLINDAGEM EXTRA: eps dentro do log impede divisões por zero na derivada <---
                # E o eps somado ao resultado do log protege a base da potência se p < 1
                term_a = torch.pow(-torch.log(a + eps) + eps, p)
                term_b = torch.pow(-torch.log(b + eps) + eps, p)
                
                soma_termos = term_a + term_b
                soma_termos_safe = torch.clamp(soma_termos, min=eps)
                
                # Consistência no uso do eps para o expoente
                den = 1.0 / (p + eps)
                
                # torch.exp é naturalmente estável para valores negativos grandes (vai para 0)
                res = torch.exp(-torch.pow(soma_termos_safe, den))
                
            return res
        
class AczelAlsinaOR:
    '''Calculates the Aczél-Alsina t-conorm family (Eq. in Image).'''
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, x, dim):
        p = float(self.p) if torch.is_tensor(self.p) else self.p
        
        # ---- case 1: λ = 0 (Drastic Sum) ----
        if p == 0.0:
            val_max, _ = torch.max(x, dim=dim)
            count_greater_than_zero = torch.sum((x > 0.0).float(), dim=dim)
            return torch.where(count_greater_than_zero <= 1, val_max, torch.ones_like(val_max))

        # ---- case 2: λ = ∞ (Maximum) ----
        elif p == float('inf'):
            return torch.max(x, dim=dim).values

        # ---- otherwise
        else:
            res = x.select(dim, 0)
            eps = 1e-6 
            
            for i in range(1, x.shape[dim]):
                a = torch.clamp(res, min=eps, max=1.0 - eps)
                b = torch.clamp(x.select(dim, i), min=eps, max=1.0 - eps)
                
                # ---> BLINDAGEM EXTRA: eps dentro do log impede divisões por zero na derivada <---
                # E o eps somado ao resultado do log protege a base da potência se p < 1
                term_a = torch.pow(-torch.log((1-a) + eps) + eps, p)
                term_b = torch.pow(-torch.log((1-b) + eps) + eps, p)
                
                soma_termos = term_a + term_b
                soma_termos_safe = torch.clamp(soma_termos, min=eps)
                
                # Consistência no uso do eps para o expoente
                den = 1.0 / (p + eps)
                
                # torch.exp é naturalmente estável para valores negativos grandes (vai para 0)
                res = 1.0 - torch.exp(-torch.pow(soma_termos_safe, den))
                
            return res

class Overlap:
    '''a^2 * b^2'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        res = x.select(dim, 0)
        for i in range(1, x.shape[dim]):
            a = res
            b = x.select(dim, i)
            
            res = a**2 * b**2
        return res

    
class Grouping:
    '''1 - (1-a)^2 * (1-b)^2'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        res = x.select(dim, 0)
        for i in range(1, x.shape[dim]):
            a = res
            b = x.select(dim, i)
            
            res = 1 - (1 - a)**2 * (1 - b)**2
        return res

#=======================================================
class NilpotentMin:
    '''
    Calculates the Nilpotent Minimum t-norm.
    Formula: T(a, b) = min(a, b) if a + b > 1, else 0.
    '''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        """
        Args:
            x: Input tensor [Batch, Attr]
            dim: Dimension for aggregation (along attributes)
        """
        # Começamos com o primeiro elemento da dimensão especificada
        # x.select(dim, 0) seleciona o primeiro índice na dimensão 'dim'
        res = x.select(dim, 0)
        
        # Percorremos os demais elementos para aplicar a regra par a par
        for i in range(1, x.shape[dim]):
            a = res
            b = x.select(dim, i)
            
            # Condição: a + b > 1
            condition = (a + b) > 1.0
            
            # Se a condição for verdadeira, o resultado é min(a, b)
            # Caso contrário, é 0 (ou um valor muito pequeno como 1e-6)
            res = torch.where(condition, 
                              torch.min(a, b), 
                              torch.zeros_like(a) + 1e-6)
            
        return res

#==================== Média ===================================

class ArithmeticMean:
    '''Calculates the Arithmetic Mean: f(x) = (1/n) * sum(xi)'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        return torch.mean(x, dim=dim)

class GeometricMean:
    '''Calculates the Geometric Mean: f(x) = root_n(prod(xi))'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        # Usamos log/exp para estabilidade numérica e evitar underflow
        return torch.exp(torch.mean(torch.log(x + 1e-6), dim=dim))

class HarmonicMean:
    '''Calculates the Harmonic Mean: f(x) = n / sum(1/xi)'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        n = x.shape[dim]
        return n / (torch.sum(1.0 / (x + 1e-6), dim=dim) + 1e-6)
    
class PowerMean:
    '''
    Calculates the Power Mean (Generalized Mean): 
    M_r(x) = ( (1/n) * sum(xi^r) )^(1/r)
    '''
    def __init__(self, p=2.0):
        self.p = p

    def __call__(self, x, dim):
        eps = 1e-6
        
        p_val = self.p
        if torch.is_tensor(p_val):
            p_safe = torch.where(
                torch.abs(p_val) < eps, 
                p_val + torch.where(p_val >= 0, eps, -eps), 
                p_val
            )
        else:
            p_safe = float(p_val)
            if abs(p_safe) < eps:
                p_safe = p_safe + (eps if p_safe >= 0 else -eps)

        x_safe = torch.clamp(x, min=eps, max=1e2)
        pow_x = torch.pow(x_safe, p_safe)
        
        mean_pow = torch.mean(pow_x, dim=dim)
        

        mean_pow_safe = torch.clamp(mean_pow, min=eps)
        
        den = 1.0 / p_safe
        
        return torch.pow(mean_pow_safe, den)

class MedianAggregation:
    '''
    Calculates the Median along a given dimension.
    If the number of elements is even, returns the average of the two middle values.
    '''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        # q=0.5 equivale à mediana estatística.
        # Por padrão, o PyTorch usa interpolação linear, 
        # o que resolve o caso de números pares de elementos.
        return torch.quantile(x, 0.5, dim=dim)

#==================== Mistos ===================================

class Prospector:
    '''
    Calculates the Prospector aggregation.
    Formula: (a + b) / (ab + (1-a)(1-b))
    '''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        # Começamos com o primeiro elemento na dimensão alvo
        res = x.select(dim, 0)
        
        # Percorremos os demais atributos para aplicar a regra em cadeia
        for i in range(1, x.shape[dim]):
            a = res
            b = x.select(dim, i)
            
            den = (1 + a * b)
            res = (a + b) / (den+1e-6)
            
        return res

class Example9Aggregation:
    '''Calculates: max(0, min(1, res + valor - param))'''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x, dim):
        p = float(self.p) if torch.is_tensor(self.p) else self.p
        res = x.select(dim, 0)
        for i in range(1, x.shape[dim]):
            val = x.select(dim, i)
            # Soma e subtrai o parâmetro
            sum_val = res + val - p
            # Mantém entre 0 e 1
            res = torch.clamp(sum_val, min=0.0, max=1.0)
        return res
    
class Example10Aggregation:
    '''Hybrid operator with thresholding at 0.5'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        res = x.select(dim, 0)
        for i in range(1, x.shape[dim]):
            val = x.select(dim, i)
            
            prod = res * val
            prob_sum = res + val - (res * val)
            
            # Condição 1: Produto > 0.5
            cond1 = prod > 0.5
            # Condição 2: (1-res)*(1-val) > 0.5
            cond2 = (1.0 - res) * (1.0 - val) > 0.5
            
            # Aplicando a lógica de decisão aninhada
            res = torch.where(cond1, prod, 
                              torch.where(cond2, prob_sum, 
                                          torch.full_like(prod, 0.5)))
        return res
    
class Example11Aggregation:
    '''Linear combination: 0.4*a + 0.4*b + 0.2*a*b'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        res = x.select(dim, 0)
        for i in range(1, x.shape[dim]):
            val = x.select(dim, i)
            # Aplicando os pesos fixos conforme sua função
            res = 0.4 * res + 0.4 * val + 0.2 * res * val
        return res
    
class Example12Aggregation:
    '''Rational operator: 2*a*b * (a + b - ab) / (a + b)'''
    def __init__(self):
        pass

    def __call__(self, x, dim):
        res = x.select(dim, 0)
        for i in range(1, x.shape[dim]):
            a = res
            b = x.select(dim, i)
            
            # Numerador: 2 * a * b * (soma_probabilística)
            num = 2 * a * b * (a + b - a * b)
            # Denominador: a + b
            den = a + b
            res = torch.where(den > 0, num / (den+1e-6), torch.zeros_like(num))
            
        return res