#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🧪 TESTS DEL MÓDULO DATA_NORMALIZATION

⚠️ IMPORTANTE:
- Este archivo SOLO hace tests, no modifica nada
- Puede ejecutarse múltiples veces sin efectos secundarios
- No toca archivos del proyecto
- Retorna 0 si pasa, 1 si falla

Ejecución:
    python test_normalization.py
"""

import sys
from pathlib import Path

# Agregar ruta del proyecto
sys.path.insert(0, str(Path(__file__).parent))

def test_normalizar_grupo():
    """Test de normalizar_grupo"""
    from modules.data_normalization import normalizar_grupo
    
    print("TEST 1: Normalizar grupos")
    
    # Test 1.1: Espacios
    resultado = normalizar_grupo("  Grupo A  ")
    assert resultado == "Grupo A", f"Esperado 'Grupo A', obtenido '{resultado}'"
    
    # Test 1.2: Número
    resultado = normalizar_grupo(1)
    assert resultado == "1", f"Esperado '1', obtenido '{resultado}'"
    
    # Test 1.3: None
    resultado = normalizar_grupo(None)
    assert resultado == "NA", f"Esperado 'NA', obtenido '{resultado}'"
    
    # Test 1.4: Minúsculas
    resultado = normalizar_grupo("  TRATAMIENTO B  ", convertir_a_minuscula=True)
    assert resultado == "tratamiento b", f"Esperado 'tratamiento b', obtenido '{resultado}'"
    
    # Test 1.5: Sin modificar original
    original = "  Grupo X  "
    _ = normalizar_grupo(original)
    assert original == "  Grupo X  ", "¡ERROR! Se modificó el valor original"
    
    print("✅ Todos los tests de normalizar_grupo pasaron\n")
    return True


def test_normalizar_valor_numerico():
    """Test de normalizar_valor_numerico"""
    from modules.data_normalization import normalizar_valor_numerico
    
    print("TEST 2: Normalizar números")
    
    # Test 2.1: String con decimales
    resultado = normalizar_valor_numerico("3.14159", decimales=2)
    assert resultado == 3.14, f"Esperado 3.14, obtenido {resultado}"
    
    # Test 2.2: String con espacios
    resultado = normalizar_valor_numerico("  0.5  ")
    assert resultado == 0.5, f"Esperado 0.5, obtenido {resultado}"
    
    # Test 2.3: None con defecto
    resultado = normalizar_valor_numerico(None, valor_defecto=0.0)
    assert resultado == 0.0, f"Esperado 0.0, obtenido {resultado}"
    
    # Test 2.4: String inválido
    resultado = normalizar_valor_numerico("abc", valor_defecto=-1.0)
    assert resultado == -1.0, f"Esperado -1.0, obtenido {resultado}"
    
    # Test 2.5: Sin modificar original
    original = "3.14159"
    _ = normalizar_valor_numerico(original)
    assert original == "3.14159", "¡ERROR! Se modificó el valor original"
    
    print("✅ Todos los tests de normalizar_valor_numerico pasaron\n")
    return True


def test_normalizar_par_tukey():
    """Test de normalizar_par_tukey"""
    from modules.data_normalization import normalizar_par_tukey
    
    print("TEST 3: Normalizar pares Tukey")
    
    # Test 3.1: Pares normales
    resultado = normalizar_par_tukey("  Grupo A  ", "Grupo B")
    assert resultado == ("Grupo A", "Grupo B"), f"Resultado inesperado: {resultado}"
    
    # Test 3.2: Ordenamiento
    resultado = normalizar_par_tukey("Grupo B", "Grupo A")
    assert resultado == ("Grupo A", "Grupo B"), f"Resultado inesperado (debe estar ordenado): {resultado}"
    
    # Test 3.3: Sin modificar originales
    g1, g2 = "  Grupo A  ", "Grupo B"
    _ = normalizar_par_tukey(g1, g2)
    assert g1 == "  Grupo A  " and g2 == "Grupo B", "¡ERROR! Se modificaron valores originales"
    
    print("✅ Todos los tests de normalizar_par_tukey pasaron\n")
    return True


def test_normalizar_bool():
    """Test de normalizar_bool"""
    from modules.data_normalization import normalizar_bool
    
    print("TEST 4: Normalizar booleanos")
    
    # Test 4.1: Bool
    assert normalizar_bool(True) == True
    assert normalizar_bool(False) == False
    
    # Test 4.2: Strings
    assert normalizar_bool("yes") == True
    assert normalizar_bool("sí") == True
    assert normalizar_bool("false") == False
    
    # Test 4.3: Números
    assert normalizar_bool(1) == True
    assert normalizar_bool(0) == False
    
    # Test 4.4: None con fallback
    assert normalizar_bool(None, fallback_pval=0.03) == True  # p < 0.05
    assert normalizar_bool(None, fallback_pval=0.10) == False  # p >= 0.05
    
    print("✅ Todos los tests de normalizar_bool pasaron\n")
    return True


def test_extraer_fila_tukey():
    """Test de extraer_fila_tukey"""
    from modules.data_normalization import extraer_fila_tukey
    
    print("TEST 5: Extraer fila Tukey")
    
    # Test 5.1: Fila válida
    row_datos = ["  Grupo A  ", "Grupo B", "2.5", "0.034", "-0.1", "5.1", True]
    resultado = extraer_fila_tukey(row_datos, indice_fila=0)
    
    assert resultado is not None, "Resultado es None"
    assert resultado['grupo1'] == "Grupo A"
    assert resultado['grupo2'] == "Grupo B"
    assert resultado['diferencia'] == 2.5
    assert resultado['es_significativo'] == True
    
    # Test 5.2: Sin modificar original
    original_row = ["  Grupo A  ", "Grupo B", "2.5", "0.034", "-0.1", "5.1", True]
    copia_row = list(original_row)
    _ = extraer_fila_tukey(original_row, indice_fila=0)
    assert original_row == copia_row, "¡ERROR! Se modificó la fila original"
    
    print("✅ Test de extraer_fila_tukey pasó\n")
    return True


def main():
    """Ejecutar todos los tests"""
    print("=" * 60)
    print("🧪 EJECUTANDO TESTS DEL MÓDULO DATA_NORMALIZATION")
    print("=" * 60)
    print()
    
    tests = [
        test_normalizar_grupo,
        test_normalizar_valor_numerico,
        test_normalizar_par_tukey,
        test_normalizar_bool,
        test_extraer_fila_tukey,
    ]
    
    resultados = []
    for test_func in tests:
        try:
            resultado = test_func()
            resultados.append(resultado)
        except Exception as e:
            print(f"❌ ERROR en {test_func.__name__}: {str(e)}\n")
            resultados.append(False)
    
    print("=" * 60)
    if all(resultados):
        print("🎉 TODOS LOS TESTS PASARON")
        print("=" * 60)
        return 0  # Éxito
    else:
        print(f"❌ {len([x for x in resultados if not x])} tests fallaron")
        print("=" * 60)
        return 1  # Fallo


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
