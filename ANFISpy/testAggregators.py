from aggregators import *
import torch
#==================== Testar ===================================

def test_operator(name, operator_instance, inputs):
    # Converte inputs para tensor [1, N] (1 exemplo com N atributos)
    x = torch.tensor([inputs], dtype=torch.float32)
    
    # Calcula usando sua classe PyTorch
    try:
        result_torch = operator_instance(x, dim=1)
        print(f"[{name}] Entradas: {inputs} -> Saída: {result_torch.item()}")
    except Exception as e:
        print(f"[{name}] Erro: {e}")

val1 = [0.6,0.4,0.5]
val2 = [0.2,0.4,0.6,0.8]
param = 3
operators = {
    0: ("Produto AND", lambda: ProdAND()),
    1: ("Produto OR", lambda: ProdOR()),
    
    2: ("Lukas AND", lambda: LukasiewiczAND()),
    3: ("Lukas OR", lambda: LukasiewiczOR()),
    
    4: ("Min AND", lambda: MinAND()),
    5: ("Max OR", lambda: MaxOR()),
    
    6: ("Frank AND", lambda: FrankAND(p=param)),
    7: ("Frank OR", lambda: FrankOR(p=param)),
    
    8: ("Drastic AND", lambda: DrasticAND()),
    9: ("Drastic OR", lambda: DrasticOR()),
    
    10: ("Yager AND", lambda: YagerAND(p=param)),
    11: ("Yager OR", lambda: YagerOR(p=param)),
    
    12: ("Hamacher AND", lambda: HamacherAND(p=param)),
    13: ("Hamacher OR", lambda: HamacherOR(p=param)),
    
    14: ("Dombi AND", lambda: DombiAND(p=param)),
    15: ("Dombi OR", lambda: DombiOR(p=param)),
    
    16: ("Schweizer AND", lambda: SchweizerSklarAND(p=param)),
    17: ("Schweizer OR", lambda: SchweizerSklarOR(p=param)),

    18: ("Aczel Alsina AND", lambda: AczelAlsinaAND(p=param)),
    19: ("Aczel Alsina OR", lambda: AczelAlsinaOR(p=param)),

    20: ("Nilpotent Minimum", lambda: NilpotentMin()),

    # Média
    21: ("Arithmetic Mean", lambda: ArithmeticMean()),
    22: ("Geometric Mean", lambda: GeometricMean()),
    23: ("Harmonic Mean", lambda: HarmonicMean()),
    24: ("Power Mean", lambda: PowerMean(p=param)),
    25: ("Median", lambda: MedianAggregation()),

    # Misto
    26: ("Prospector (Unorm)", lambda: Prospector()),
    27: ("Example 9 (Linear Clamp)", lambda: Example9Aggregation(param=0.5)),
    28: ("Example 10 (Hybrid Threshold)", lambda: Example10Aggregation()),
    29: ("Example 11 (Linear Combination)", lambda: Example11Aggregation()),
    30: ("Example 12 (Rational Hybrid)", lambda: Example12Aggregation()),

    31: ("Overlap AND", lambda: Overlap()),
    32: ("Grouping OR", lambda: Grouping()),
}

selected_ops = [31,32]

for idx in selected_ops:
    name, op_fn = operators[idx]
    op = op_fn()
    
    test_operator(f"{name} (3 val)", op, val1)
    test_operator(f"{name} (4 val)", op, val2)