import pytest
from modules import types
from typing import Any, get_type_hints
import pandas as pd
import numpy as np

class TestTypesImport:
    """Tests de importación y disponibilidad de tipos."""
    
    def test_types_module_exists(self) -> None:
        """Verifica que el módulo types existe y es accesible."""
        assert types is not None
        assert hasattr(types, '__doc__')
    
    def test_serieslike_type_available(self) -> None:
        """Verifica que SeriesLike está disponible."""
        assert hasattr(types, 'SeriesLike')
    
    def test_dataframelike_type_available(self) -> None:
        """Verifica que DataFrameLike está disponible."""
        assert hasattr(types, 'DataFrameLike')
    
    def test_statresult_type_available(self) -> None:
        """Verifica que StatResult está disponible."""
        assert hasattr(types, 'StatResult')

class TestTypesDefinitions:
    """Tests de definición correcta de tipos."""
    
    def test_statresult_is_dict_like(self) -> None:
        """Verifica que StatResult es compatible con dict."""
        # StatResult debe ser Dict[str, Any]
        test_result: types.StatResult = {'value': 1.5, 'p_value': 0.05}
        assert isinstance(test_result, dict)
    
    def test_serieslike_accepts_series(self, simple_medical_data: pd.DataFrame) -> None:
        """Verifica que SeriesLike acepta pd.Series."""
        series = simple_medical_data['age']
        assert isinstance(series, pd.Series)
    
    def test_dataframelike_accepts_dataframe(self, simple_medical_data: pd.DataFrame) -> None:
        """Verifica que DataFrameLike acepta pd.DataFrame."""
        assert isinstance(simple_medical_data, pd.DataFrame)

    def test_groupdata_type_definition(self) -> None:
        """Verifica que GroupData está definido como alias."""
        # GroupData = Dict[str, ArrayLike]
        data: types.GroupData = {"group1": [1, 2], "group2": [3, 4]}
        assert isinstance(data, dict)
