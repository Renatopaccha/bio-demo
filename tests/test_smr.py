
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.stats.calculos_tasas import (
    calcular_ajuste_indirecto,
    grafico_forestplot_smr,
    exportar_resultados_indirecto
)

def run_tests():
    print("Running tests for Indirect Method (SMR)...")
    
    # Create dummy data
    df = pd.DataFrame({
        'Grupo': ['0-4', '5-9'],
        'Casos': [10, 20],
        'Poblacion': [1000, 2000]
    })
    
    # Test 1: Calculation
    print("\n--- Test 1: Calculation ---")
    res = calcular_ajuste_indirecto(df, 'Grupo', 'Casos', 'Poblacion')
    
    assert 'smr' in res
    assert 'ic_95_inf' in res
    assert 'ic_95_sup' in res
    assert 'es_significativo' in res
    assert isinstance(res['por_grupo'], pd.DataFrame)
    
    print(f"SMR calculated: {res['smr']:.3f} (Expected > 0)")
    assert res['smr'] > 0
    print("PASS")
    
    # Test 2: Graph
    print("\n--- Test 2: Graph ---")
    fig = grafico_forestplot_smr(res)
    assert fig is not None
    # Check if layout title is present (rudimentary check of figure object)
    assert "SMR" in fig.layout.title.text
    print("PASS")
    
    # Test 3: Export
    print("\n--- Test 3: Export ---")
    try:
        paths = exportar_resultados_indirecto(res, res['por_grupo'], "test_smr_export")
        print(f"Files created: {paths}")
        
        # Cleanup
        for p in paths.values():
            if os.path.exists(p):
                os.remove(p)
                print(f"Removed {p}")
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        raise e

    print("\nALL SMR TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
