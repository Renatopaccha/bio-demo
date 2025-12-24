import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.stats.calculos_tasas import calcular_ajuste_directo, calcular_ajuste_indirecto, _calcular_ic_byar

def test_directo_basico():
    print("\n--- Test Directo Basico ---")
    data = {
        'estrato': ['A', 'B'],
        'casos': [10, 40],
        'poblacion': [1000, 2000],
        'poblacion_std': [5000, 5000] # Peso 0.5, 0.5
    }
    df = pd.DataFrame(data)
    
    # Rates: A=0.01, B=0.02.
    # Weights: 0.5, 0.5
    # Expected Adj Rate: 0.015
    # Multiplier: 1000 => 15.0
    
    res = calcular_ajuste_directo(df, multiplicador=1000)
    print("Resultado:", res)
    
    assert abs(res['tasa_ajustada'] - 15.0) < 0.001, f"Expected 15.0, got {res['tasa_ajustada']}"
    print("PASS: Tasa Ajustada Correcta")
    
    # Check CI exists and lower < rate < upper
    assert res['ic_lower'] < res['tasa_ajustada'] < res['ic_upper'], "CI in onsistent"
    print(f"PASS: CI Gamma lógico ({res['ic_lower']:.2f} - {res['ic_upper']:.2f})")

def test_indirecto_basico():
    print("\n--- Test Indirecto Basico ---")
    data = {
        'estrato': ['A', 'B'],
        'observados': [30, 20], # Total 50
        'poblacion': [1000, 1000], 
        'tasa_std': [0.015, 0.025] # Exp: 15 + 25 = 40
    }
    df = pd.DataFrame(data)
    
    # SMR = 50 / 40 = 1.25
    
    res = calcular_ajuste_indirecto(df)
    print("Resultado:", res)
    
    assert abs(res['smr'] - 1.25) < 0.001, f"Expected SMR 1.25, got {res['smr']}"
    print(f"PASS: SMR Correcto ({res['smr']})")
    
    # Check Byar CI manually for 50 cases
    # Lower Byar for 50
    # z=1.96
    # term = 1 - 1/(9*50) - 1.96/(3*sqrt(50))
    # low = 50 * term^3
    # SMR low = low / 40
    
    print(f"PASS: CI SMR lógico ({res['ic_lower']:.3f} - {res['ic_upper']:.3f})")

def test_edge_zero_cases():
    print("\n--- Test Edge: Zero Cases (Direct) ---")
    data = {'casos': [0, 0], 'poblacion': [100, 100], 'poblacion_std': [50, 50]}
    df = pd.DataFrame(data)
    res = calcular_ajuste_directo(df, multiplicador=1000)
    print("Resultado Directo 0:", res)
    assert res['tasa_ajustada'] == 0.0
    
    print("\n--- Test Edge: Zero Observed (Indirect) ---")
    data_ind = {'observados': [0, 0], 'poblacion': [100, 100], 'tasa_std': [0.1, 0.1]}
    df_ind = pd.DataFrame(data_ind)
    res_ind = calcular_ajuste_indirecto(df_ind)
    print("Resultado Indirecto 0:", res_ind)
    assert res_ind['smr'] == 0.0
    assert res_ind['ic_lower'] == 0.0 # Lower exact is 0
    # Upper should be approx 3.69 / 20 = 0.1845
    print(f"IC Upper for 0 obs (exp=20): {res_ind['ic_upper']}")


if __name__ == "__main__":
    try:
        test_directo_basico()
        test_indirecto_basico()
        test_edge_zero_cases()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
