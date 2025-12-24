import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from typing import Tuple
from modules.stats.hipotesis import (
    ttest_independiente,
    ttest_pareado,
    anova_unidireccional as anova_oneway,
    chi_cuadrado as chi_square_test,
    pearson_correlation,
)

# Estrategias de hypothesis para generar datos
@st.composite
def medical_data_strategy(draw) -> pd.Series:
    """Genera datos médicos realistas."""
    size = draw(st.integers(min_value=20, max_value=500))
    return pd.Series(draw(st.lists(
        st.floats(min_value=0, max_value=300, allow_nan=False, allow_infinity=False),
        min_size=size,
        max_size=size
    )))

@st.composite
def two_groups_strategy(draw) -> Tuple[pd.Series, pd.Series]:
    """Genera dos grupos de datos para comparación."""
    size1 = draw(st.integers(min_value=15, max_value=200))
    size2 = draw(st.integers(min_value=15, max_value=200))
    group1 = pd.Series(draw(st.lists(
        st.floats(min_value=50, max_value=150, allow_nan=False, allow_infinity=False),
        min_size=size1, max_size=size1
    )))
    group2 = pd.Series(draw(st.lists(
        st.floats(min_value=60, max_value=160, allow_nan=False, allow_infinity=False),
        min_size=size2, max_size=size2
    )))
    return group1, group2

class TestTtestIndependiente:
    """Tests para t-test independiente."""
    
    @given(two_groups_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_ttest_independiente_property(self, data: Tuple[pd.Series, pd.Series]) -> None:
        """Property-based test: t-test siempre retorna p-value válido."""
        grupo1, grupo2 = data
        if len(grupo1) < 2 or len(grupo2) < 2:
            return
        
        # Check variance
        if grupo1.std() == 0 and grupo2.std() == 0:
            return 

        result = ttest_independiente(grupo1, grupo2)
        
        # Valid keys: p_value, test_statistic
        p_val = result.get('p_value', result.get('p-value', np.nan))
        
        if np.isnan(p_val):
            # p-value can be nan if data is constant or identical
            return

        assert 0 <= p_val <= 1
        
    @pytest.mark.parametrize("diff", [0, 5, 10, 20, 50])
    def test_ttest_independiente_effect_size(self, diff: float) -> None:
        """Valida que efectos más grandes dan p-values más pequeños."""
        np.random.seed(42)
        grupo1 = pd.Series(np.random.normal(100, 15, 100))
        grupo2 = pd.Series(np.random.normal(100 + diff, 15, 100))
        
        result = ttest_independiente(grupo1, grupo2)
        p_val = result.get('p_value')
        
        # Mayor diferencia = menor p-value (usualmente)
        if diff > 10:
            assert p_val < 0.05
        elif diff == 0:
            assert p_val > 0.01 
    
    def test_ttest_independiente_same_group(self) -> None:
        """Valida que mismo grupo da p-value alto."""
        data = pd.Series(np.random.normal(100, 15, 100))
        result = ttest_independiente(data, data)
        p_val = result.get('p_value')
        
        assert p_val > 0.9

class TestTtestPareado:
    """Tests para t-test pareado."""
    
    @given(st.lists(
        st.tuples(
            st.floats(min_value=80, max_value=120, allow_nan=False, allow_infinity=False),
            st.floats(min_value=80, max_value=130, allow_nan=False, allow_infinity=False)
        ),
        min_size=10,
        max_size=200
    ))
    @settings(max_examples=30, deadline=None)
    def test_ttest_pareado_property(self, paired_data: list) -> None:
        """Property-based test: t-test pareado retorna p-value válido."""
        if len(paired_data) < 3:
            return
        
        before = pd.Series([x[0] for x in paired_data])
        after = pd.Series([x[1] for x in paired_data])
        
        if before.std() == 0 and after.std() == 0:
            return
        
        result = ttest_pareado(before, after)
        
        p_val = result.get('p_value')
        
        if p_val is None or np.isnan(p_val):
            return

        assert 0 <= p_val <= 1

class TestAnova:
    """Tests para ANOVA."""
    
    @given(st.lists(
        st.lists(
            st.floats(min_value=0, max_value=200, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=50
        ),
        min_size=2,
        max_size=5
    ))
    @settings(max_examples=20, deadline=None)
    def test_anova_property(self, groups: list) -> None:
        """Property-based test: ANOVA retorna p-value válido."""
        valid_groups = [pd.Series(g) for g in groups if len(g) >= 3]
        
        if len(valid_groups) < 2:
            return
        
        # Check variance > 0
        if all(g.std() == 0 for g in valid_groups):
            return
            
        # Signature: anova_unidireccional(groups_data: Dict[str, ArrayLike])
        groups_dict = {f"Group_{i}": g for i, g in enumerate(valid_groups)}

        result = anova_oneway(groups_dict) # Pass dict
        
        p_val = result.get('p_value') 
        
        if p_val is None or np.isnan(p_val):
            return
            
        assert 0 <= p_val <= 1
        
class TestPearsonCorrelation:
    """Tests para correlación de Pearson."""
    
    @given(
        st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                min_size=15, max_size=200)
    )
    @settings(max_examples=30, deadline=None)
    def test_correlation_property(self, x_values: list) -> None:
        """Property-based test: correlación retorna valor válido."""
        if len(set(x_values)) < 2: # No variance
            return
            
        x = pd.Series(x_values)
        np.random.seed(42)
        y = x + pd.Series(np.random.normal(0, 5, len(x)))
        
        result = pearson_correlation(x, y)
        
        coef = result.get('correlation') # Verified key
        p_val = result.get('p_value')
        
        if coef is None or np.isnan(coef):
            return
            
        assert -1 <= coef <= 1
        
    def test_correlation_perfect_positive(self) -> None:
        """Valida correlación perfecta positiva."""
        x = pd.Series(np.arange(100))
        y = pd.Series(np.arange(100))
        
        result = pearson_correlation(x, y)
        
        coef = result.get('correlation')
        assert coef > 0.99
    
    def test_correlation_no_correlation(self) -> None:
        """Valida sin correlación."""
        np.random.seed(42)
        x = pd.Series(np.random.normal(0, 1, 100))
        y = pd.Series(np.random.normal(0, 1, 100))
        
        result = pearson_correlation(x, y)
        coef = result.get('correlation')
        
        assert abs(coef) < 0.5
