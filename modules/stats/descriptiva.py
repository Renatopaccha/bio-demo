"""
Módulo de Renderizado: Estadística Descriptiva (Formato Médico/Investigación)
-----------------------------------------------------------------------------
Genera tablas de estadísticas descriptivas con formato profesional (Tabla 1)
y análisis detallado de distribución.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Dict, Any, Tuple

import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats

# Importamos la utilidad de guardado
from modules.utils import boton_guardar_tabla, boton_guardar_grafico, card_container

# Importar Copiloto IA para interpretación de tablas (con fallback robusto)
try:
    from modules.ai_chat import render_ai_actions_for_result as ai_actions_for_result
except ImportError:
    try:
        from ai_chat import render_ai_actions_for_result as ai_actions_for_result
    except ImportError:
        # Si no está disponible, creamos función dummy
        def ai_actions_for_result(*args, **kwargs):
            pass

from modules.stats.core import (
    calculate_descriptive_stats,
    detect_outliers_advanced,
    check_normality,
    check_homoscedasticity,
    check_symmetry_kurtosis,
    get_normal_curve_data,
    get_qq_coordinates,
    analyze_outlier_details,
    calculate_group_comparison,
    generate_table_one_structure,
    calculate_frequency_table,
    generate_crosstab_analysis,
    interpret_crosstab
)
from modules.stats.validators import validate_data_for_analysis

# =============================================================================
# OVERRIDE: Corrected Contingency Table Calculations
# =============================================================================
import html

def _safe_div(a, b):
    # Divide evitando división por cero
    return np.where(b == 0, np.nan, a / b)

def _fmt_int(x):
    if pd.isna(x):
        return "—"
    try:
        return f"{int(x)}"
    except Exception:
        return "—"

def _fmt_pct(x, decimals=2):
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):.{decimals}f}%"
    except Exception:
        return "—"

def _build_crosstab_matrices(sub_df: pd.DataFrame, row_var: str, col_var: str):
    """
    Devuelve:
      n_df: frecuencias con TOTAL fila/columna
      row_pct_df: % fila con TOTAL
      col_pct_df: % columna con TOTAL
      total_pct_df: % total con TOTAL
    """
    # Mantengo el comportamiento actual: excluir NA (como tu análisis actual de contingencia)
    d = sub_df[[row_var, col_var]].dropna()

    ct = pd.crosstab(d[row_var], d[col_var], dropna=True)

    if ct.empty:
        return None, None, None, None

    # Frecuencias + totales
    n_df = ct.copy()
    n_df["TOTAL"] = n_df.sum(axis=1)
    total_row = n_df.sum(axis=0)
    total_row.name = "TOTAL"
    n_df = pd.concat([n_df, total_row.to_frame().T], axis=0)

    grand_total = n_df.loc["TOTAL", "TOTAL"]

    # Totales por fila / columna (incluyen TOTAL)
    row_totals = n_df["TOTAL"].to_numpy().reshape(-1, 1)          # (n_rows, 1)
    col_totals = n_df.loc["TOTAL", :].to_numpy().reshape(1, -1)   # (1, n_cols)

    # % Fila
    row_pct_df = pd.DataFrame(
        _safe_div(n_df.to_numpy(), row_totals) * 100,
        index=n_df.index, columns=n_df.columns
    )

    # % Columna
    col_pct_df = pd.DataFrame(
        _safe_div(n_df.to_numpy(), col_totals) * 100,
        index=n_df.index, columns=n_df.columns
    )

    # % Total
    total_pct_df = pd.DataFrame(
        _safe_div(n_df.to_numpy(), grand_total) * 100,
        index=n_df.index, columns=n_df.columns
    )

    return n_df, row_pct_df, col_pct_df, total_pct_df

def _crosstab_to_html(n_df, row_pct_df, col_pct_df, total_pct_df, metrics, row_label_name="VALORES"):
    # Orden bonito y estable
    metric_order = ["n", "row_pct", "col_pct", "total_pct"]
    metrics = [m for m in metric_order if m in metrics]

    metric_names = {
        "n": "N",
        "row_pct": "% Fila",
        "col_pct": "% Columna",
        "total_pct": "% Total",
    }

    cols = list(n_df.columns)

    # Construcción HTML con "bloques" (rowspan) por cada categoría
    rows_html = []
    body_index = list(n_df.index)

    for i, r in enumerate(body_index):
        # zebra por bloque
        block_class = "ct-block-even" if i % 2 == 0 else "ct-block-odd"
        rowspan = max(1, len(metrics))

        first_metric = True
        for m in metrics:
            if m == "n":
                mat = n_df
                formatter = _fmt_int
                decimals = None
            elif m == "row_pct":
                mat = row_pct_df
                formatter = lambda x: _fmt_pct(x, 2)
            elif m == "col_pct":
                mat = col_pct_df
                formatter = lambda x: _fmt_pct(x, 2)
            else:
                mat = total_pct_df
                formatter = lambda x: _fmt_pct(x, 2)

            tds = []

            # Col 1: valor (solo en la primera fila del bloque, con rowspan)
            if first_metric:
                r_txt = html.escape(str(r))
                tds.append(f'<td class="ct-rowlabel {block_class}" rowspan="{rowspan}"><b>{r_txt}</b></td>')
                first_metric = False

            # Col 2: etiqueta de métrica
            tds.append(f'<td class="ct-metric {block_class}">{metric_names[m]}</td>')

            # Celdas por columnas
            for c in cols:
                v = mat.loc[r, c]
                # Alineación derecha; TOTAL con énfasis
                extra = " ct-totalcell" if (str(r) == "TOTAL" or str(c) == "TOTAL") else ""
                tds.append(f'<td class="ct-cell {block_class}{extra}">{formatter(v)}</td>')

            rows_html.append("<tr>" + "".join(tds) + "</tr>")

        # Separador entre bloques
        rows_html.append('<tr class="ct-sep"><td colspan="999"></td></tr>')

    # Header
    ths = [f"<th>{html.escape(row_label_name)}</th>", "<th>Métrica</th>"] + [f"<th>{html.escape(str(c))}</th>" for c in cols]
    thead = "<thead><tr>" + "".join(ths) + "</tr></thead>"

    tbody = "<tbody>" + "\n".join(rows_html) + "</tbody>"

    return f'<table class="ct-table">{thead}{tbody}</table>'

def generate_crosstab_analysis(sub_df: pd.DataFrame, row_var: str, col_var: str, metrics: list):
    """
    Override local: genera matrices (N, %fila, %columna, %total)
    y tabla HTML estilo Epidat.
    """
    n_df, row_pct_df, col_pct_df, total_pct_df = _build_crosstab_matrices(sub_df, row_var, col_var)

    if n_df is None or n_df.empty:
        return {
            "formatted_df": pd.DataFrame(),
            "raw_n": pd.DataFrame(),
            "analysis_text": "No se pudo construir la tabla cruzada (sin datos suficientes).",
            "table_html": "",
            "table_css": "",
        }

    # CSS académico (parecido a tu "paper" pero adaptado)
    table_css = """
    <style>
      .ct-table{
        width:100%;
        border-collapse:collapse;
        font-family: 'Source Sans Pro', Helvetica, Arial, sans-serif;
        font-size: 15px;
        border-radius: 14px;
        overflow: hidden;
      }
      .ct-table thead th{
        text-align:left;
        font-weight:700;
        color:#0B2E6B;
        background: #F6F8FC;
        padding: 0.85rem;
        border-bottom: 2px solid rgba(11,46,107,0.18);
        position: sticky;
        top: 0;
        z-index: 1;
      }
      .ct-table td{
        padding: 0.75rem 0.85rem;
        border-bottom: 1px solid rgba(0,0,0,0.08);
        vertical-align: middle;
      }
      .ct-rowlabel{ width: 28%; }
      .ct-metric{ width: 14%; color:#5B6472; font-weight:600; }
      .ct-cell{ text-align:right; font-variant-numeric: tabular-nums; }
      .ct-block-even{ background:#FFFFFF; }
      .ct-block-odd{ background:#FAFBFE; }
      .ct-totalcell{
        font-weight: 700;
        color:#0B2E6B;
      }
      .ct-sep td{
        padding:0;
        height: 10px;
        border: none;
        background: transparent;
      }
    </style>
    """

    table_html = _crosstab_to_html(
        n_df=n_df,
        row_pct_df=row_pct_df,
        col_pct_df=col_pct_df,
        total_pct_df=total_pct_df,
        metrics=metrics,
        row_label_name=row_var
    )

    # Reutiliza tu análisis textual existente si está disponible
    try:
        analysis_text = generar_analisis_contingencia(sub_df, row_var, col_var)
    except Exception:
        analysis_text = "Interpretación no disponible."

    return {
        "formatted_df": pd.DataFrame({"_": [1]}),  # marcador no vacío para no romper tu validación actual
        "raw_n": n_df,
        "row_pct": row_pct_df,
        "col_pct": col_pct_df,
        "total_pct": total_pct_df,
        "analysis_text": analysis_text,
        "table_html": table_html,
        "table_css": table_css,
    }


# =============================================================================
# INTERPRETACIÓN AUTOMÁTICA (TEXTO ESTILO TESIS / SALUD) - SOLO UI, NO CÁLCULOS
# =============================================================================
from math import sqrt
try:
    from scipy.stats import chi2_contingency
except Exception:
    chi2_contingency = None

def _fmt_num(x, d=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)

def _fmt_p(p):
    try:
        p = float(p)
        if np.isnan(p): 
            return ""
        return "<0.001" if p < 0.001 else f"{p:.3f}"
    except Exception:
        return str(p)

def _var_contexto_salud(var_name: str) -> str:
    """Pequeña heurística para dar contexto sanitario sin inventar causalidad."""
    v = (var_name or "").lower()
    if any(k in v for k in ["imc", "bmi"]):
        return " (variable antropométrica: IMC)"
    if "peso" in v:
        return " (variable antropométrica: peso)"
    if "talla" in v or "altura" in v or "estatura" in v:
        return " (variable antropométrica: talla/altura)"
    if any(k in v for k in ["edad", "años"]):
        return " (variable demográfica: edad)"
    if any(k in v for k in ["sexo", "género", "genero"]):
        return " (variable demográfica: sexo/género)"
    return ""

def generar_analisis_numerico_descriptivo(df_resumen: pd.DataFrame, titulo: str = "Interpretación automática") -> str:
    """
    Interpreta una tabla tipo resumen (1 fila por variable).
    Espera columnas típicas: Variable, N, Media, Mediana, IC 95%, D.E., CV %, Mín, Máx, IQR, P-Normalidad, Asimetría, Curtosis.
    Robusto a faltantes.
    """
    if df_resumen is None or df_resumen.empty:
        return "No hay resultados para interpretar."

    out = [f"### 🧠 {titulo}", 
           "Texto sugerido (borrador): útil para tesis/informe. Ajusta redacción según tu estudio (población, contexto clínico).",
           ""]
    
    for _, r in df_resumen.iterrows():
        var = str(r.get("Variable", "")).strip()
        if not var:
            continue
        
        n = r.get("N", None)
        p_norm = r.get("P-Normalidad", None)
        de = r.get("D.E.", r.get("DE", None))
        cv = r.get("CV %", r.get("CV (%)", None))
        media = r.get("Media", None)
        mediana = r.get("Mediana", None)
        ic = r.get("IC 95%", r.get("IC 95%", r.get("IC 95% (Media)", None)))
        minimo = r.get("Mín", r.get("Min", None))
        maximo = r.get("Máx", r.get("Max", None))
        iqr = r.get("IQR", None)
        asim = r.get("Asimetría", None)
        curt = r.get("Curtosis", None)

        # Normalidad (si existe)
        normal_txt = ""
        if p_norm is not None and str(p_norm) != "":
            try:
                pn = float(p_norm)
                normal_txt = f"Distribución {'compatible con normalidad' if pn > 0.05 else 'no compatible con normalidad'} (p={_fmt_p(pn)}). "
            except Exception:
                normal_txt = f"Normalidad: p={_fmt_p(p_norm)}. "

        # Dispersión por CV (si existe)
        disp_txt = ""
        try:
            cvv = float(str(cv).replace("%", "").strip())
            if cvv < 15:
                disp_txt = f"Variabilidad baja (CV≈{_fmt_num(cvv,2)}%). "
            elif cvv < 30:
                disp_txt = f"Variabilidad moderada (CV≈{_fmt_num(cvv,2)}%). "
            else:
                disp_txt = f"Variabilidad alta (CV≈{_fmt_num(cvv,2)}%). "
        except Exception:
            pass
        
        # Centro recomendado
        centro_txt = ""
        # Si hay p de normalidad y es normal -> media (IC) + DE
        try:
            pn = float(p_norm)
            if pn > 0.05:
                centro_txt = f"Se recomienda reportar Media±DE: {_fmt_num(media,2)}±{_fmt_num(de,2)}"
                if ic not in [None, "", np.nan]:
                    centro_txt += f", con IC95% {ic}"
                centro_txt += ". "
            else:
                # no normal -> mediana (IQR)
                if mediana not in [None, "", np.nan]:
                    centro_txt = f"Se recomienda reportar Mediana: {_fmt_num(mediana,2)}"
                    if iqr not in [None, "", np.nan]:
                        centro_txt += f" (IQR={_fmt_num(iqr,2)})"
                    centro_txt += ". "
        except Exception:
            # fallback si no hay normalidad
            if mediana not in [None, "", np.nan] and iqr not in [None, "", np.nan]:
                centro_txt = f"Centro: mediana {_fmt_num(mediana,2)} (IQR={_fmt_num(iqr,2)}). "
            elif media not in [None, "", np.nan] and de not in [None, "", np.nan]:
                centro_txt = f"Centro: media±DE {_fmt_num(media,2)}±{_fmt_num(de,2)}. "

        rango_txt = ""
        if minimo not in [None, "", np.nan] or maximo not in [None, "", np.nan]:
            rango_txt = f"Rango observado: {_fmt_num(minimo,2)} a {_fmt_num(maximo,2)}. "

        forma_txt = ""
        if asim not in [None, "", np.nan] or curt not in [None, "", np.nan]:
            forma_txt = f"Forma: asimetría={_fmt_num(asim,2)}; curtosis={_fmt_num(curt,2)}. "

        ctx = _var_contexto_salud(var)
        out.append(f"**{var}**{ctx}: n={_fmt_num(n,0)}. {normal_txt}{centro_txt}{disp_txt}{rango_txt}{forma_txt}".strip())
    
    out.append("")
    out.append("**Nota clínica/epidemiológica:** evita lenguaje causal; describe patrones. Si hay subgrupos, complementa con comparaciones (Tabla 1) o modelos.")
    return "\n\n".join(out)

def generar_analisis_frecuencias(freq_df: pd.DataFrame, var_name: str, n_total_registros: int = None) -> str:
    """
    Interpreta tabla de frecuencias (Categoría, Frecuencia (n), Porcentaje (%), Acumulado (%)).
    """
    if freq_df is None or freq_df.empty:
        return "No hay tabla de frecuencias para interpretar."

    df = freq_df.copy()
    # excluir TOTAL si existe
    if "Categoría" in df.columns:
        df_no_total = df[df["Categoría"].astype(str).str.upper() != "TOTAL"].copy()
    else:
        df_no_total = df.copy()

    # total válido
    n_valid = None
    try:
        if "Categoría" in df.columns and "Frecuencia (n)" in df.columns:
            total_row = df[df["Categoría"].astype(str).str.upper() == "TOTAL"]
            if not total_row.empty:
                n_valid = int(float(total_row["Frecuencia (n)"].iloc[0]))
    except Exception:
        pass

    # top categorías
    top_lines = []
    try:
        if "Frecuencia (n)" in df_no_total.columns:
            df_no_total["__n"] = pd.to_numeric(df_no_total["Frecuencia (n)"], errors="coerce")
            df_no_total = df_no_total.sort_values("__n", ascending=False)
        top = df_no_total.head(3)
        for _, r in top.iterrows():
            cat = str(r.get("Categoría", "")).strip()
            nn = r.get("Frecuencia (n)", "")
            pp = r.get("Porcentaje (%)", "")
            top_lines.append(f"{cat}: {nn} ({_fmt_num(pp,2)}%)")
    except Exception:
        pass

    missing_txt = ""
    try:
        if n_total_registros is not None and n_valid is not None and n_total_registros > n_valid:
            miss = n_total_registros - n_valid
            missing_txt = f"Se identifican **{miss} valores ausentes** (registros totales={n_total_registros}, válidos={n_valid}). "
    except Exception:
        pass

    acumulado_txt = ""
    if "Acumulado (%)" in df.columns:
        acumulado_txt = ("El **porcentaje acumulado** es interpretable principalmente cuando la variable tiene "
                         "un **orden natural** (p.ej., categorías ordinales). Si se ordena por frecuencia, el acumulado puede perder significado.")

    ctx = _var_contexto_salud(var_name)
    texto = [f"### 🧠 Interpretación automática: {var_name}{ctx}",
             missing_txt.strip(),
             ("Categorías más frecuentes (Top 3): " + "; ".join(top_lines) + ".") if top_lines else "",
             acumulado_txt,
             "Sugerencia para redacción: reporta n y % (y si aplica, comenta acumulados/jerarquía de categorías)."]
    return "\n\n".join([t for t in texto if t])

def generar_analisis_contingencia(sub_df: pd.DataFrame, row_var: str, col_var: str) -> str:
    """
    Interpreta relación entre dos variables categóricas (tabla de contingencia).
    Calcula Chi2 y V de Cramér si es posible.
    """
    if sub_df is None or sub_df.empty:
        return "No hay datos para interpretar."
    if row_var not in sub_df.columns or col_var not in sub_df.columns:
        return "No se encontraron las variables seleccionadas en el dataset."

    ct = pd.crosstab(sub_df[row_var], sub_df[col_var], dropna=True)
    if ct.empty:
        return "No se pudo construir la tabla cruzada (sin datos suficientes)."

    chi_txt = ""
    v_txt = ""
    if chi2_contingency is not None and ct.shape[0] >= 2 and ct.shape[1] >= 2:
        try:
            chi2, p, dof, exp = chi2_contingency(ct)
            chi_txt = f"Prueba ji-cuadrado: χ²={_fmt_num(chi2,2)}, gl={dof}, p={_fmt_p(p)}."
            # Cramér's V
            n = ct.to_numpy().sum()
            k = min(ct.shape[0]-1, ct.shape[1]-1)
            if n > 0 and k > 0:
                v = sqrt(chi2 / (n * k))
                # guía simple de magnitud (no rígida)
                if v < 0.10:
                    mag = "muy pequeña"
                elif v < 0.30:
                    mag = "pequeña a moderada"
                elif v < 0.50:
                    mag = "moderada"
                else:
                    mag = "alta"
                v_txt = f"Tamaño de efecto (V de Cramér)={_fmt_num(v,2)} → asociación {mag}."
        except Exception:
            pass

    # patrón de diferencias (porcentajes por fila)
    patron_txt = ""
    try:
        row_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        # detectar la celda más "dominante"
        max_cell = row_pct.stack().idxmax()
        max_val = row_pct.loc[max_cell[0], max_cell[1]]
        patron_txt = f"El patrón más marcado es **{row_var}={max_cell[0]}** con mayor proporción en **{col_var}={max_cell[1]}** ({_fmt_num(max_val,1)}% dentro de esa fila)."
    except Exception:
        pass

    texto = [
        f"### 🧠 Interpretación automática: {row_var} vs {col_var}",
        "Una tabla de contingencia permite evaluar la relación entre dos variables cualitativas y contrastar asociación con ji-cuadrado (si procede).",
        chi_txt,
        v_txt,
        patron_txt,
        "Recomendación: si p<0.05, reporta asociación (no causalidad) y acompaña con tamaños de efecto y porcentajes por fila/columna."
    ]
    return "\n\n".join([t for t in texto if t])

def mostrar_interpretacion_ui(md_text: str, key: str):
    """Wrapper UI consistente."""
    if not md_text:
        return
    with st.expander("🧠 Interpretación automática (borrador para tesis)", expanded=False):
        st.markdown(md_text)


# =============================================================================
# CORRELACIONES (PEARSON / SPEARMAN) - HELPERS
# =============================================================================
def _calcular_matrices_correlacion(df: pd.DataFrame, variables: List[str], metodo: str = "pearson"):
    """Calcula matrices de r, p-valor y N (observaciones válidas por par).

    - metodo: 'pearson' o 'spearman'
    - Usa eliminación por pares (pairwise complete observations), como SPSS/EpiDat.
    """
    metodo = (metodo or "pearson").lower().strip()
    if metodo not in ("pearson", "spearman"):
        metodo = "pearson"

    data = df[variables].copy()
    k = len(variables)

    r = np.full((k, k), np.nan, dtype=float)
    p = np.full((k, k), np.nan, dtype=float)
    n = np.zeros((k, k), dtype=int)

    for i, vi in enumerate(variables):
        # Diagonal: N de la variable consigo misma
        n_ii = int(data[vi].notna().sum())
        r[i, i] = 1.0
        n[i, i] = n_ii
        p[i, i] = np.nan

        for j in range(i + 1, k):
            vj = variables[j]
            par = data[[vi, vj]].dropna()
            nij = int(len(par))
            n[i, j] = nij
            n[j, i] = nij

            if nij < 3:
                rij, pij = (np.nan, np.nan)
            else:
                try:
                    if metodo == "pearson":
                        rij, pij = stats.pearsonr(par[vi].astype(float), par[vj].astype(float))
                    else:
                        rij, pij = stats.spearmanr(par[vi].astype(float), par[vj].astype(float))
                except Exception:
                    rij, pij = (np.nan, np.nan)

            r[i, j] = rij
            r[j, i] = rij
            p[i, j] = pij
            p[j, i] = pij

    r_df = pd.DataFrame(r, index=variables, columns=variables)
    p_df = pd.DataFrame(p, index=variables, columns=variables)
    n_df = pd.DataFrame(n, index=variables, columns=variables)
    return r_df, p_df, n_df


def _construir_tabla_correlacion_export(r_df: pd.DataFrame, p_df: pd.DataFrame, n_df: pd.DataFrame, include_n: bool = True):
    """Tabla (formato SPSS) para exportación/Reporte: filas por variable (r, p y opcionalmente N)."""
    vars_ = list(r_df.columns)
    filas = []
    metricas = [("r", r_df), ("p", p_df)]
    if include_n:
        metricas.append(("N", n_df))

    for vi in vars_:
        for etiqueta, mat in metricas:
            row = {"Variable": vi, "Métrica": etiqueta}
            for vj in vars_:
                row[vj] = mat.loc[vi, vj]
            filas.append(row)
    return pd.DataFrame(filas)



def _render_tabla_correlacion_html(r_df: pd.DataFrame, p_df: pd.DataFrame, n_df: pd.DataFrame, metodo: str, alpha: float = 0.05, include_n: bool = True):
    """Render HTML con estilo 'paper' y lectura clara (r / p / N opcional)."""
    import html as _html

    vars_ = list(r_df.columns)

    def fmt_r(x):
        return "" if pd.isna(x) else f"{x:.3f}"

    def fmt_p(x):
        if pd.isna(x):
            return ""
        if x < 0.001:
            return "<0.001"
        return f"{x:.3f}"

    def fmt_n(x):
        return "" if pd.isna(x) else f"{int(x)}"

    css = """
    <style>
    .corr-wrap { width: 100%; overflow: auto; border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; }
    table.corr { width: 100%; border-collapse: collapse; font-family: 'Source Sans Pro', Helvetica, Arial, sans-serif; }
    table.corr thead th {
        position: sticky; top: 0; z-index: 2;
        background: #f7f9fc;
        text-align: center;
        font-weight: 700;
        color: #0b2d6b;
        padding: 12px 10px;
        border-bottom: 2px solid rgba(11,45,107,0.18);
        white-space: nowrap;
    }
    table.corr thead th.left { text-align: left; }
    table.corr td {
        padding: 10px 10px;
        border-bottom: 1px solid rgba(0,0,0,0.08);
        vertical-align: middle;
        white-space: nowrap;
    }
    table.corr td.var {
        font-weight: 700;
        color: #1f2937;
        background: rgba(11,45,107,0.04);
        border-right: 1px solid rgba(0,0,0,0.06);
    }
    table.corr td.metric {
        font-weight: 600;
        color: #6b7280;
        width: 120px;
    }
    table.corr td.num { text-align: right; font-variant-numeric: tabular-nums; }
    table.corr tr.group-sep td { border-bottom: 2px solid rgba(0,0,0,0.12); }
    .sig { font-weight: 800; color: #0b2d6b; }
    .pval { color: #6b7280; }
    .note { font-size: 0.9rem; color: rgba(0,0,0,0.65); margin-top: 8px; }
    </style>
    """

    thead = "<thead><tr><th class='left'>Variable</th><th class='left'>Métrica</th>" + "".join(
        [f"<th>{_html.escape(v)}</th>" for v in vars_]
    ) + "</tr></thead>"

    body_rows = []
    rowspan = 3 if include_n else 2

    for vi in vars_:
        vi_esc = _html.escape(str(vi))

        # r row
        tds_r = []
        for vj in vars_:
            val = r_df.loc[vi, vj]
            pval = p_df.loc[vi, vj]
            txt = fmt_r(val)
            if (not pd.isna(pval)) and (pval < alpha) and (not pd.isna(val)) and (abs(val) >= 0.30) and (vi != vj):
                txt = f"<span class='sig'>{txt}</span>"
            tds_r.append(f"<td class='num'>{txt}</td>")

        tds_p = [f"<td class='num pval'>{fmt_p(p_df.loc[vi, vj])}</td>" for vj in vars_]

        body_rows.append(
            "<tr>"
            f"<td class='var' rowspan='{rowspan}'>{vi_esc}</td>"
            "<td class='metric'>Coef. (r)</td>" + "".join(tds_r) +
            "</tr>"
        )
        body_rows.append("<tr><td class='metric'>p-valor</td>" + "".join(tds_p) + "</tr>")

        if include_n:
            tds_n = [f"<td class='num'>{fmt_n(n_df.loc[vi, vj])}</td>" for vj in vars_]
            body_rows.append("<tr class='group-sep'><td class='metric'>N</td>" + "".join(tds_n) + "</tr>")
        else:
            body_rows.append("<tr class='group-sep'><td class='metric'></td>" + "".join([f"<td class='num'></td>" for _ in vars_]) + "</tr>")

    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"

    titulo = "Pearson" if (metodo or "").lower().startswith("p") else "Spearman"
    nota_n = "Incluye N por pares." if include_n else "N oculto."
    nota = f"<div class='note'>Método: <b>{titulo}</b>. {nota_n} Se resalta <b>r</b> cuando p&lt;{alpha:g} y |r|≥0.30.</div>"
    html = css + f"<div class='corr-wrap'><table class='corr'>{thead}{tbody}</table></div>" + nota
    return html


def generar_interpretacion_correlaciones(r_df: pd.DataFrame, p_df: pd.DataFrame, n_df: pd.DataFrame, metodo: str, alpha: float = 0.05, top_k: int = 6):
    """Interpretación amigable para tesis/investigación (área de salud)."""
    metodo = (metodo or "pearson").lower().strip()
    nombre_metodo = "Pearson" if metodo == "pearson" else "Spearman"

    vars_ = list(r_df.columns)
    pares = []
    for i in range(len(vars_)):
        for j in range(i + 1, len(vars_)):
            vi, vj = vars_[i], vars_[j]
            r = r_df.loc[vi, vj]
            p = p_df.loc[vi, vj]
            nij = n_df.loc[vi, vj]
            if pd.isna(r) or pd.isna(p) or nij == 0:
                continue
            pares.append((abs(r), r, p, nij, vi, vj))
    pares.sort(reverse=True, key=lambda x: x[0])

    if not pares:
        return (
            "### 🧠 Interpretación automática\n\n"
            "No se pudo calcular ninguna correlación válida con las variables seleccionadas.\n\n"
            "**Sugerencias:** verifica que existan al menos 2 variables numéricas con datos, y que no estén casi vacías."
        )

    n_vals = [x[3] for x in pares]
    n_min, n_max = int(min(n_vals)), int(max(n_vals))
    sig = [x for x in pares if x[2] < alpha]
    prop_sig = (len(sig) / len(pares)) * 100 if pares else 0

    def fuerza(absr):
        if absr < 0.20:
            return "muy débil"
        if absr < 0.40:
            return "débil"
        if absr < 0.60:
            return "moderada"
        if absr < 0.80:
            return "fuerte"
        return "muy fuerte"

    top = pares[:min(top_k, len(pares))]

    lineas = []
    lineas.append("### 🧠 Interpretación automática (borrador para tesis)")
    lineas.append(f"Se calcularon correlaciones **{nombre_metodo}** entre las variables seleccionadas.")
    lineas.append(f"El tamaño muestral por par (N) varió entre **{n_min}** y **{n_max}** debido a valores perdidos (análisis por pares).")
    lineas.append("")
    lineas.append(f"**Significancia estadística (α={alpha:g}):** {len(sig)} de {len(pares)} pares ({prop_sig:.1f}%) mostraron asociación estadísticamente significativa.")
    lineas.append("")
    lineas.append("**Asociaciones más destacadas (por magnitud |r|):**")
    for absr, r, pval, nij, vi, vj in top:
        signo = "positiva" if r > 0 else "negativa"
        pv_txt = "<0.001" if pval < 0.001 else f"{pval:.3f}"
        lineas.append(f"- **{vi}** vs **{vj}**: r = **{r:.3f}** ({fuerza(absr)}, {signo}), p = **{pv_txt}**, N = **{nij}**.")
    lineas.append("")
    lineas.append("**Cómo leer esto (en salud):**")
    lineas.append("- Correlación **positiva**: cuando una variable aumenta, la otra tiende a aumentar.")
    lineas.append("- Correlación **negativa**: cuando una variable aumenta, la otra tiende a disminuir.")
    lineas.append("- La correlación cuantifica **asociación**, no implica causalidad (posibles confusores: edad, comorbilidades, tratamiento, etc.).")
    lineas.append("")
    if metodo == "pearson":
        lineas.append("**Nota metodológica (Pearson):** evalúa relación **lineal** y es más sensible a outliers. Ideal con variables continuas y distribución aproximadamente normal para inferencia.")
    else:
        lineas.append("**Nota metodológica (Spearman):** evalúa asociación **monótona** (por rangos), útil si hay asimetría marcada, outliers o variables ordinales.")
    lineas.append("\n**Recomendación práctica:** si una correlación es importante para tu hipótesis clínica, revisa un **diagrama de dispersión** y considera análisis multivariado (p.ej., regresión) para ajustar confusores.")
    return "\n".join(lineas)


# =============================================================================
# FILTRO ESTILO EPIDATA (REGLAS + COMBINACIÓN Y/O)
# =============================================================================
def _epidata_get_ops_for_series(s: pd.Series) -> List[str]:
    if pd.api.types.is_numeric_dtype(s):
        return ["=", "≠", ">", ">=", "<", "<=", "entre", "en lista", "es NA", "no es NA"]
    else:
        return ["=", "≠", "contiene", "no contiene", "empieza con", "termina con", "en lista", "es NA", "no es NA"]


def _epidata_parse_list(value: str) -> List[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _epidata_apply_rule(df: pd.DataFrame, var: str, op: str, value: str) -> pd.Series:
    s = df[var]

    # NA checks
    if op == "es NA":
        return s.isna()
    if op == "no es NA":
        return ~s.isna()

    # Numeric
    if pd.api.types.is_numeric_dtype(s):
        def to_float(x: str) -> float:
            x = str(x).strip().replace(" ", "")
            if x.count(",") == 1 and x.count(".") == 0:
                x = x.replace(",", ".")
            return float(x)

        if op in ("=", "≠", ">", ">=", "<", "<="):
            try:
                v = to_float(value)
            except Exception:
                return pd.Series(False, index=df.index)

            s_num = pd.to_numeric(s, errors="coerce")
            if op == "=":
                return s_num == v
            if op == "≠":
                return s_num != v
            if op == ">":
                return s_num > v
            if op == ">=":
                return s_num >= v
            if op == "<":
                return s_num < v
            if op == "<=":
                return s_num <= v

        if op == "entre":
            vals = _epidata_parse_list(value)
            if len(vals) != 2:
                return pd.Series(False, index=df.index)
            try:
                a, b = to_float(vals[0]), to_float(vals[1])
            except Exception:
                return pd.Series(False, index=df.index)
            lo, hi = (a, b) if a <= b else (b, a)
            s_num = pd.to_numeric(s, errors="coerce")
            return (s_num >= lo) & (s_num <= hi)

        if op == "en lista":
            vals = _epidata_parse_list(value)
            if not vals:
                return pd.Series(False, index=df.index)
            s_num = pd.to_numeric(s, errors="coerce")
            ok = []
            for t in vals:
                try:
                    ok.append(float(t.replace(",", ".")))
                except Exception:
                    pass
            if not ok:
                return pd.Series(False, index=df.index)
            return s_num.isin(ok)

        return pd.Series(False, index=df.index)

    # Categorical / text
    s_txt = s.astype(str)

    if op == "=":
        return s_txt == str(value)
    if op == "≠":
        return s_txt != str(value)
    if op == "contiene":
        return s_txt.str.contains(str(value), case=False, na=False)
    if op == "no contiene":
        return ~s_txt.str.contains(str(value), case=False, na=False)
    if op == "empieza con":
        return s_txt.str.lower().str.startswith(str(value).lower(), na=False)
    if op == "termina con":
        return s_txt.str.lower().str.endswith(str(value).lower(), na=False)
    if op == "en lista":
        vals = _epidata_parse_list(value)
        if not vals:
            return pd.Series(False, index=df.index)
        vals_low = set(v.lower() for v in vals)
        return s_txt.str.lower().isin(vals_low)

    return pd.Series(False, index=df.index)


def _epidata_apply_filter(df: pd.DataFrame, rules: List[Dict[str, Any]], comb: str) -> Tuple[pd.DataFrame, str]:
    """Aplica reglas usando comb: 'Y' (AND) o 'O' (OR)."""
    if not rules:
        return df, "Sin filtro aplicado."

    comb = (comb or "Y").strip().upper()
    if comb not in ("Y", "O"):
        comb = "Y"

    masks = []
    desc_parts = []

    for r in rules:
        var = r.get("var")
        op = r.get("op")
        val = r.get("val", "")
        if not var or not op:
            continue
        if var not in df.columns:
            continue
        mask = _epidata_apply_rule(df, var, op, val)
        masks.append(mask)
        if op in ("es NA", "no es NA"):
            desc_parts.append(f"{var} {op}")
        else:
            desc_parts.append(f"{var} {op} {val}")

    if not masks:
        return df, "Filtro no válido (sin reglas aplicables)."

    if comb == "Y":
        final_mask = masks[0]
        for m in masks[1:]:
            final_mask = final_mask & m
        txt = " Y ".join(desc_parts)
    else:
        final_mask = masks[0]
        for m in masks[1:]:
            final_mask = final_mask | m
        txt = " O ".join(desc_parts)

    df2 = df[final_mask].copy()
    return df2, f"Filtro aplicado: {txt}"


# --- RENDER PRINCIPAL ---
def render_descriptiva(df: Optional[pd.DataFrame] = None, selected_vars: Optional[List[str]] = None):
    """
    Renderiza la sección de estadísticas descriptivas con formato de publicación médica.
    Organizada en 4 pestañas: Univariado, Comparativa, Gráficos y Tabla Inteligente.
    """
    st.subheader("📊 Estadística Descriptiva")
    
    # 1. Recuperación de Datos
    if df is None:
        if 'df_principal' in st.session_state:
            df = st.session_state.df_principal
        else:
            st.error("⚠️ No hay datos cargados en la sesión.")
            return
    
    # 2. Validación Básica
    valid, msg = validate_data_for_analysis(df)
    if not valid:
        st.error(f"⚠️ {msg}")
        return
    
    # 3. Selección de Variables (Global)
    if selected_vars is None:
        numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numericas:
            st.warning("El dataset no contiene variables numéricas.")
            return
            
        selected_vars = st.multiselect(
            "Seleccione las variables a analizar:",
            options=numericas,
            default=numericas[:5] if len(numericas) > 5 else numericas,
            key="desc_selected_vars_global"
        )
    
    if not selected_vars:
        st.info("Seleccione al menos una variable para continuar.")
        return
    
    st.divider()
    
    # 4. Estructura de Pestañas (AHORA SON 7)
    tab_univariado, tab_categoricas, tab_contingencia, tab_correlaciones, tab_comparativa, tab_graficos, tab_inteligente = st.tabs([
        "📋 Univariado (Global)",
        "🧩 Variables Categóricas",
        "🔀 Tablas de Contingencia",
        "🔗 Correlaciones",
        "⚔️ Comparativa (Tabla 1)", 
        "📊 Gráficos Diagnósticos",
        "📰 Tabla Inteligente (Paper)"
    ])
    
    # Columnas categóricas (se usa en Variables Categóricas y Tablas de Contingencia)
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Helper para render Paper en Univariado
    import streamlit.components.v1 as components
    
    def _render_df_paper_univariado(
        df_in: pd.DataFrame,
        max_height_px: int = 420,
        title: str = None,
        fmt: dict = None,
        highlight_rules: dict = None
    ) -> str:
        """
        Render genérico para tablas en Univariado con estilo Paper.
        - fmt: dict(col -> callable formatter(value)->str)
        - highlight_rules: dict(col -> callable value->css_inline_style)
          Ej: {"Acción": lambda v: "color:#B91C1C;font-weight:800" if "Eliminar" in str(v) else ...}
        """
        if df_in is None or df_in.empty:
            return "<div style='font-family:Arial;padding:8px;'>Sin datos para mostrar.</div>"

        df = df_in.copy().reset_index(drop=True)

        # formateo
        def _default_fmt(v):
            if pd.isna(v):
                return ""
            if isinstance(v, (int, np.integer)):
                return f"{int(v)}"
            if isinstance(v, (float, np.floating)):
                return f"{float(v):.2f}"
            return str(v)

        fmt = fmt or {}
        highlight_rules = highlight_rules or {}

        headers = "".join([f"<th>{c}</th>" for c in df.columns])

        rows = []
        for _, r in df.iterrows():
            tds = []
            for j, c in enumerate(df.columns):
                v = r[c]
                # aplicar formatter
                if c in fmt:
                    try:
                        txt = fmt[c](v)
                    except Exception:
                        txt = _default_fmt(v)
                else:
                    txt = _default_fmt(v)

                # estilos por celda
                inline = ""
                if c in highlight_rules:
                    try:
                        inline = highlight_rules[c](v) or ""
                    except Exception:
                        inline = ""

                td_class = "text" if j == 0 else "num"
                tds.append(f"<td class='{td_class}' style='{inline}'>{txt}</td>")
            rows.append(f"<tr>{''.join(tds)}</tr>")

        caption_html = f"<div class='cap'>{title}</div>" if title else ""

        html = f"""
        <html>
        <head>
          <style>
            :root {{
              --accent: #0B3A82;
              --ink: #111827;
              --muted: #6B7280;
              --line: rgba(17,24,39,.12);
              --line2: rgba(11,58,130,.22);
              --bg: #ffffff;
              --zebra: rgba(17,24,39,.02);
            }}

            body {{
              margin: 0;
              padding: 0;
              background: transparent;
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
              color: var(--ink);
            }}

            .wrap {{
              width: 100%;
              max-height: {max_height_px}px;
              overflow: auto;
              padding: 2px;
            }}

            .cap {{
              font-size: 0.95rem;
              font-weight: 700;
              color: var(--accent);
              margin: 0 0 10px 2px;
            }}

            table.paper {{
              width: 100%;
              border-collapse: separate;
              border-spacing: 0;
              background: var(--bg);
              font-size: 0.95rem;
              border: 1px solid var(--line);
              border-radius: 12px;
              overflow: hidden;
              box-shadow: 0 1px 10px rgba(17,24,39,.05);
            }}

            th, td {{
              padding: 0.70rem 0.85rem;
              border-bottom: 1px solid var(--line);
              vertical-align: middle;
              white-space: nowrap;
            }}

            thead th {{
              text-align: left;
              font-weight: 800;
              color: var(--accent);
              background: linear-gradient(to bottom, #ffffff, rgba(11,58,130,.02));
              border-bottom: 2px solid var(--line2);
              position: sticky;
              top: 0;
              z-index: 2;
            }}

            td.text {{ text-align: left; }}
            td.num  {{ text-align: right; font-variant-numeric: tabular-nums; }}

            tbody tr:nth-child(even) {{ background: var(--zebra); }}
            tbody tr:hover {{ background: rgba(11,58,130,.04); }}

            /* permitir que la primera columna (Variable) pueda quebrar línea */
            tbody td:first-child {{
              white-space: normal;
              max-width: 320px;
              word-break: break-word;
              font-weight: 600;
            }}
          </style>
        </head>
        <body>
          <div class="wrap">
            {caption_html}
            <table class="paper">
              <thead><tr>{headers}</tr></thead>
              <tbody>
                {''.join(rows)}
              </tbody>
            </table>
          </div>
        </body>
        </html>
        """
        return html
    
    # ==============================================================================
    # PESTAÑA 1: UNIVARIADO (GLOBAL) - SOLO NUMÉRICAS
    # ==============================================================================
    with tab_univariado:
        data_summary = []
        data_percentiles = []
        
        for var in selected_vars:
            stats = calculate_descriptive_stats(df[var])
            
            mean_val = stats.get('mean')
            ci_low = stats.get('ci95_lower')
            ci_high = stats.get('ci95_upper')
            
            if pd.notna(mean_val) and pd.notna(ci_low) and pd.notna(ci_high):
                mean_str = f"{mean_val:.2f} ({ci_low:.2f} - {ci_high:.2f})"
            else:
                mean_str = f"{mean_val:.2f}" if pd.notna(mean_val) else "-"
            
            median_val = stats.get('median')
            p25 = stats.get('p25')
            p75 = stats.get('p75')
            
            if pd.notna(median_val) and pd.notna(p25) and pd.notna(p75):
                median_str = f"{median_val:.2f} ({p25:.2f} - {p75:.2f})"
            else:
                median_str = f"{median_val:.2f}" if pd.notna(median_val) else "-"
                
            min_val = stats.get('min')
            max_val = stats.get('max')
            if pd.notna(min_val) and pd.notna(max_val):
                range_str = f"{min_val:.2f} - {max_val:.2f}"
            else:
                range_str = "-"
                
            cv_val = stats.get('cv')
            cv_str = f"{cv_val:.2f}%" if pd.notna(cv_val) else "-"
            
            row_summary = {
                'Variable': var,
                'N': stats.get('n', 0),
                'Media (IC 95%)': mean_str,
                'Mediana (P25 - P75)': median_str,
                'D.E.': stats.get('std'), 
                'Mín - Máx': range_str,
                'CV %': cv_str
            }
            data_summary.append(row_summary)
            
            row_perc = {
                'Variable': var,
                'P5': stats.get('p5'),
                'P10': stats.get('p10'),
                'P25 (Q1)': stats.get('p25'),
                'P50 (Mediana)': stats.get('p50'),
                'P75 (Q3)': stats.get('p75'),
                'P90': stats.get('p90'),
                'P95': stats.get('p95')
            }
            data_percentiles.append(row_perc)
            
        df_resumen = pd.DataFrame(data_summary)
        df_percentiles = pd.DataFrame(data_percentiles)
        
        st.markdown("### 📋 Resumen Descriptivo (Global)")
        st.caption("Reporte estándar con Intervalos de Confianza (IC 95%) y Rangos Intercuartílicos.")
        
        
        components.html(
            _render_df_paper_univariado(
                df_resumen,
                max_height_px=420,
                fmt={
                    "N": lambda v: "" if pd.isna(v) else f"{int(float(v))}",
                    "D.E.": lambda v: "" if pd.isna(v) else f"{float(v):.2f}",
                    "CV %": lambda v: "" if pd.isna(v) else f"{float(v):.2f}",
                }
            ),
            height=470,
            scrolling=True
        )
        
        boton_guardar_tabla(df_resumen, "Resumen_Descriptivo_Global", "btn_resumen_global", orientacion="Horizontal (como SPSS)")

        # Copiloto IA: Interpretación académica de la tabla + Chat
        ai_actions_for_result(
            df_resultado=df_resumen,
            titulo="Tabla 1: Resumen Descriptivo Global",
            notas="Estadísticas descriptivas principales del dataset incluyendo medidas de tendencia central, dispersión e intervalos de confianza al 95%.",
            key="univariado_global"
        )

        mostrar_interpretacion_ui(
            generar_analisis_numerico_descriptivo(df_resumen, titulo="Interpretación del Resumen Descriptivo"),
            key="interp_univariado_resumen"
        )
        
        st.divider()
        
        with st.expander("📍 Ver Distribución Detallada (Percentiles)", expanded=False):
            cols_perc = ['P5', 'P10', 'P25 (Q1)', 'P50 (Mediana)', 'P75 (Q3)', 'P90', 'P95']
            fmt_perc = {c: (lambda v: "" if pd.isna(v) else f"{float(v):.4f}") for c in cols_perc}
            components.html(
                _render_df_paper_univariado(
                    df_percentiles,
                    max_height_px=420,
                    fmt=fmt_perc
                ),
                height=470,
                scrolling=True
            )
            boton_guardar_tabla(df_percentiles, "Percentiles_Global", "btn_percentiles_global", orientacion="Horizontal (como SPSS)")
        
        st.divider()
        
        with st.expander("🚨 Diagnóstico de Outliers (Detección Avanzada)", expanded=True):
            reporte_outliers = []
            for var in selected_vars:
                out_res = detect_outliers_advanced(df[var])
                for method in ['iqr', 'zscore', 'mad']:
                    indices = out_res.get(f'{method}_outliers', [])
                    if indices:
                        analysis = analyze_outlier_details(df[var], indices, method)
                        reporte_outliers.append({
                            'Variable': var,
                            'Método': method.upper(),
                            'Número': analysis['count'],
                            'Valores': analysis['values_str'],
                            'Acción': analysis['action']
                        })
            if reporte_outliers:
                df_reporte = pd.DataFrame(reporte_outliers)
                
                def _hl_accion(v):
                    s = str(v)
                    if "Eliminar" in s:
                        return "color:#B91C1C;font-weight:800;"
                    if "Revisar" in s:
                        return "color:#B45309;font-weight:800;"
                    if "OK" in s:
                        return "color:#047857;font-weight:800;"
                    return ""

                components.html(
                    _render_df_paper_univariado(
                        df_reporte,
                        max_height_px=420,
                        highlight_rules={"Acción": _hl_accion}
                    ),
                    height=470,
                    scrolling=True
                )
                boton_guardar_tabla(df_reporte, "Reporte_Outliers_Global", "btn_outliers_global", orientacion="Horizontal (como SPSS)")
            else:
                st.success("✅ Tus datos están PERFECTAMENTE LIMPIOS (No hay outliers).")
        
        with st.expander("✅ Validación de Supuestos (Normality Checks)", expanded=False):
            assumptions_data = []
            for var in selected_vars:
                norm_res = check_normality(df[var])
                stats_basic = calculate_descriptive_stats(df[var])
                shape_res = check_symmetry_kurtosis(stats_basic.get('skewness'), stats_basic.get('kurtosis'))
                is_normal = norm_res.get('conclusion') == "Normal"
                is_symmetric = shape_res.get('symmetry_eval') == "Simétrica"
                recomendacion = "✅ Paramétrico (T-Test/ANOVA)" if is_normal else ("✅ Paramétrico (Robustez TCL)" if is_symmetric and stats_basic.get('n', 0) > 30 else "⚠️ No Paramétrico (Mann-Whitney/Kruskal)")
                assumptions_data.append({
                    'Variable': var,
                    'Shapiro-Wilk p-val': norm_res.get('shapiro_p'),
                    'Skewness (Simetría)': stats_basic.get('skewness'),
                    'Kurtosis': stats_basic.get('kurtosis'),
                    'Normality Conclusion': norm_res.get('conclusion'),
                    'Recomendación': recomendacion
                })
            df_assumptions = pd.DataFrame(assumptions_data)
            
            def _hl_p(v):
                try:
                    p = float(v)
                    if p >= 0.05:
                        return "color:#047857;font-weight:800;"
                    if p < 0.01:
                        return "color:#B91C1C;font-weight:800;"
                    return "color:#B45309;font-weight:800;"
                except Exception:
                    return ""

            components.html(
                _render_df_paper_univariado(
                    df_assumptions,
                    max_height_px=420,
                    fmt={
                        "Shapiro-Wilk W": lambda v: "" if pd.isna(v) else f"{float(v):.4f}",
                        "Shapiro-Wilk p-val": lambda v: "" if pd.isna(v) else (f"{float(v):.4f}" if float(v) >= 0.001 else "<0.001"),
                    },
                    highlight_rules={"Shapiro-Wilk p-val": _hl_p}
                ),
                height=470,
                scrolling=True
            )
            boton_guardar_tabla(df_assumptions, "Validacion_Supuestos_Global", "btn_supuestos_global", orientacion="Horizontal (como SPSS)")
    
    # ==============================================================================
    # PESTAÑA 2: VARIABLES CATEGÓRICAS
    # ==============================================================================
    with tab_categoricas:
        st.markdown("### 📊 Variables Categóricas (Tablas de Frecuencia)")
        st.caption("Tablas de frecuencia y gráficos para variables categóricas.")
        
        import streamlit.components.v1 as components

        def _render_tabla_freq_paper(df_in: pd.DataFrame, max_height_px: int = 420) -> str:
            df = df_in.copy()

            # Formateo tipo tesis (solo visual)
            for col in df.columns:
                col_low = str(col).lower()
                if "porcentaje" in col_low or "acumulado" in col_low or "%" in col_low:
                    df[col] = df[col].apply(lambda x: "" if pd.isna(x) else f"{float(x):.2f}%")
                elif "frecuencia" in col_low or col_low in ("n", "count"):
                    df[col] = df[col].apply(lambda x: "" if pd.isna(x) else f"{int(float(x))}")
                else:
                    df[col] = df[col].astype(str)

            # Construcción HTML con clase especial para TOTAL
            headers = "".join([f"<th>{c}</th>" for c in df.columns])

            rows_html = []
            first_col = df.columns[0]
            for _, row in df.iterrows():
                is_total = str(row[first_col]).strip().upper() == "TOTAL"
                tr_class = "total" if is_total else ""
                tds = []
                for j, c in enumerate(df.columns):
                    cell = row[c]
                    td_class = "text" if j == 0 else "num"
                    tds.append(f"<td class='{td_class}'>{cell}</td>")
                rows_html.append(f"<tr class='{tr_class}'>{''.join(tds)}</tr>")

            html = f"""
            <html>
            <head>
              <style>
                :root {{
                  --accent: #0B3A82;
                  --accent-soft: rgba(11,58,130,.08);
                  --ink: #111827;
                  --muted: #6B7280;
                  --line: rgba(17,24,39,.12);
                  --line2: rgba(11,58,130,.22);
                  --bg: #ffffff;
                  --zebra: rgba(17,24,39,.02);
                }}

                body {{
                  margin: 0;
                  padding: 0;
                  background: transparent;
                  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                  color: var(--ink);
                }}

                .paper-wrap {{
                  width: 100%;
                  max-height: {max_height_px}px;
                  overflow: auto;
                  padding: 2px;
                }}

                table.tabla-paper {{
                  width: 100%;
                  border-collapse: separate;
                  border-spacing: 0;
                  background: var(--bg);
                  font-size: 0.95rem;
                  border: 1px solid var(--line);
                  border-radius: 12px;
                  overflow: hidden;
                  box-shadow: 0 1px 10px rgba(17,24,39,.05);
                }}

                table.tabla-paper th,
                table.tabla-paper td {{
                  padding: 0.70rem 0.85rem;
                  border-bottom: 1px solid var(--line);
                  vertical-align: middle;
                  white-space: nowrap;
                }}

                table.tabla-paper thead th {{
                  text-align: left;
                  font-weight: 700;
                  color: var(--accent);
                  background: linear-gradient(to bottom, #ffffff, rgba(11,58,130,.02));
                  border-bottom: 2px solid var(--line2);
                  position: sticky;
                  top: 0;
                  z-index: 2;
                }}

                td.text {{ text-align: left; }}
                td.num  {{ text-align: right; font-variant-numeric: tabular-nums; }}

                tbody tr:nth-child(even) {{
                  background: var(--zebra);
                }}

                tbody tr:hover {{
                  background: rgba(11,58,130,.04);
                }}

                tr.total td {{
                  background: var(--accent-soft);
                  font-weight: 700;
                }}
              </style>
            </head>

            <body>
              <div class="paper-wrap">
                <table class="tabla-paper">
                  <thead><tr>{headers}</tr></thead>
                  <tbody>
                    {''.join(rows_html)}
                  </tbody>
                </table>
              </div>
            </body>
            </html>
            """
            return html
        
        if not cat_cols:
            st.info("No se detectaron variables categóricas (texto/categoría) en el dataset.")
        else:
            sel_cats = st.multiselect(
                "Seleccione variables categóricas para analizar:",
                options=cat_cols,
                default=[cat_cols[0]] if len(cat_cols) > 0 else None,
                key='multiselect_freq_tables'
            )
            
            if sel_cats:
                col_seg, _ = st.columns([1, 2])
                with col_seg:
                    segment_var = st.selectbox(
                        "🔹 Segmentar por (opcional):",
                        options=["(Ninguno)"] + [c for c in cat_cols if c not in sel_cats],
                        help="Divide el análisis en subgrupos (ej. por Sexo)."
                    )
                
                for i, var_cat in enumerate(sel_cats):
                    st.markdown(f"### 📌 Variable: {var_cat}")
                    
                    if segment_var == "(Ninguno)":
                        freq_df = calculate_frequency_table(df[var_cat])
                        if not freq_df.empty:
                            components.html(_render_tabla_freq_paper(freq_df, max_height_px=420), height=460, scrolling=True)
                            boton_guardar_tabla(freq_df, f"Frecuencia_{var_cat}", f"btn_freq_{i}", orientacion="Horizontal (como SPSS)")
                            
                            ai_actions_for_result(
                                freq_df, 
                                f"Frecuencia: {var_cat}", 
                                notas="Tabla de frecuencias absoluta y relativa.",
                                key=f"ai_freq_{i}"
                            )

                            n_total = int(df[var_cat].shape[0])
                            mostrar_interpretacion_ui(
                                generar_analisis_frecuencias(freq_df, var_cat, n_total_registros=n_total),
                                key=f"interp_freq_{i}"
                            )
                            df_plot = freq_df[freq_df['Categoría'] != 'TOTAL']
                            fig = px.bar(df_plot, x='Categoría', y='Frecuencia (n)', text='Porcentaje (%)', color='Frecuencia (n)', title=f"Distribución Global: {var_cat}", color_continuous_scale='Teal')
                            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            st.plotly_chart(fig, use_container_width=True)
                            boton_guardar_grafico(fig, f"Grafico_Frecuencia_{var_cat}", f"btn_fig_freq_{i}")
                    else:
                        grupos = df[segment_var].dropna().unique()
                        grupos.sort()
                        tabs = st.tabs([f"📂 {g}" for g in grupos])
                        for idx, grupo in enumerate(grupos):
                            with tabs[idx]:
                                st.caption(f"Subgrupo: **{segment_var} = {grupo}**")
                                df_sub = df[df[segment_var] == grupo]
                                freq_df_sub = calculate_frequency_table(df_sub[var_cat])
                                if not freq_df_sub.empty:
                                    components.html(_render_tabla_freq_paper(freq_df_sub, max_height_px=420), height=460, scrolling=True)
                                    boton_guardar_tabla(freq_df_sub, f"Frecuencia_{var_cat}_{segment_var}_{grupo}", f"btn_freq_{i}_{idx}", orientacion="Horizontal (como SPSS)")
                                    
                                    ai_actions_for_result(
                                        freq_df_sub, 
                                        f"Frecuencia: {var_cat} ({segment_var}={grupo})", 
                                        notas="Tabla de frecuencias para el subgrupo seleccionado.",
                                        key=f"ai_freq_{i}_{idx}"
                                    )

                                    n_total_g = int(df_sub[var_cat].shape[0])
                                    mostrar_interpretacion_ui(
                                        generar_analisis_frecuencias(freq_df_sub, f"{var_cat} ({segment_var}={grupo})", n_total_registros=n_total_g),
                                        key=f"interp_freq_{i}_{idx}"
                                    )
                                    df_plot_sub = freq_df_sub[freq_df_sub['Categoría'] != 'TOTAL']
                                    fig_sub = px.bar(df_plot_sub, x='Categoría', y='Frecuencia (n)', text='Porcentaje (%)', color='Frecuencia (n)', title=f"{var_cat} ({segment_var}={grupo})", color_continuous_scale='Blues')
                                    fig_sub.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                                    st.plotly_chart(fig_sub, use_container_width=True)
                                    boton_guardar_grafico(fig_sub, f"Grafico_Frecuencia_{var_cat}_{segment_var}_{grupo}", f"btn_fig_freq_{i}_{idx}")
                                else:
                                    st.warning("Sin datos para este grupo.")
                    st.divider()
            else:
                st.info("Seleccione al menos una variable categórica.")
    
    # ==============================================================================
    # PESTAÑA 3: TABLAS DE CONTINGENCIA
    # ==============================================================================
    with tab_contingencia:
        import streamlit.components.v1 as components
        
        st.markdown("### 🔀 Tablas de Contingencia (Bivariado)")
        st.caption("Análisis de relación entre dos variables categóricas (Chi-Cuadrado).")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            row_var = st.selectbox("Variable Fila:", options=cat_cols, index=0, key="ct_row")
        with c2:
            col_opts = [c for c in cat_cols if c != row_var]
            col_var = st.selectbox("Variable Columna:", options=col_opts, index=0, key="ct_col")
        with c3:
            seg_opts = ["(Ninguno)"] + [c for c in cat_cols if c not in [row_var, col_var]]
            seg_var = st.selectbox("🔹 Segmentar por:", options=seg_opts, key="ct_seg")
        
        metrics = st.multiselect(
            "Métricas a visualizar:",
            options=['n', 'row_pct', 'col_pct', 'total_pct'],
            default=['n', 'row_pct'],
            format_func=lambda x: {'n': 'Frecuencia (N)', 'row_pct': '% Fila', 'col_pct': '% Columna', 'total_pct': '% Total'}[x],
            key="ct_metrics"
        )
        
        st.divider()
        
        def _pct100(series_or_df):
            # Si el % viene en escala 0-1, lo convierte a 0-100 SOLO para visualización
            try:
                mx = pd.to_numeric(series_or_df.stack() if hasattr(series_or_df, "stack") else series_or_df, errors="coerce").max()
                if mx is not None and mx <= 1.01:
                    return series_or_df * 100
            except Exception:
                pass
            return series_or_df

        def _build_crosstab_stacked_df(res: dict, mets: list) -> pd.DataFrame:
            """
            Construye una tabla estilo EPIDAT:
            - 1ra columna: categoría de fila (solo en la primera subfila de cada grupo)
            - 2da columna: etiqueta de métrica (N, % Fila, % Columna, % Total)
            - resto: columnas de la variable columna + TOTAL
            """
            raw_n = res.get("raw_n")
            if raw_n is None or raw_n.empty:
                return pd.DataFrame()

            row_pct = _pct100(res.get("row_pct")) if res.get("row_pct") is not None else None
            col_pct = _pct100(res.get("col_pct")) if res.get("col_pct") is not None else None
            tot_pct = _pct100(res.get("total_pct")) if res.get("total_pct") is not None else None

            colnames = list(raw_n.columns)

            met_map = {
                "n": ("N", raw_n, "int"),
                "row_pct": ("% Fila", row_pct, "pct"),
                "col_pct": ("% Columna", col_pct, "pct"),
                "total_pct": ("% Total", tot_pct, "pct"),
            }

            # Orden fijo de presentación tipo EPIDAT (respeta selección del usuario)
            order = [m for m in ["n", "row_pct", "col_pct", "total_pct"] if m in mets]
            if not order:
                order = ["n"]

            rows_out = []
            for idx in raw_n.index:
                first = True
                for met in order:
                    label, mat, kind = met_map[met]
                    r = {"VALORES": str(idx) if first else "", "Métrica": label}
                    if mat is None or mat.empty:
                        # si no hay matriz, deja vacío
                        for c in colnames:
                            r[c] = ""
                    else:
                        for c in colnames:
                            v = mat.loc[idx, c] if (idx in mat.index and c in mat.columns) else ""
                            if kind == "int":
                                try:
                                    r[c] = "" if pd.isna(v) else f"{int(float(v))}"
                                except Exception:
                                    r[c] = str(v)
                            else:
                                try:
                                    r[c] = "" if pd.isna(v) else f"{float(v):.2f}%"
                                except Exception:
                                    r[c] = str(v)
                    rows_out.append(r)
                    first = False

            df_out = pd.DataFrame(rows_out, columns=["VALORES", "Métrica"] + colnames)
            return df_out

        def _render_crosstab_paper_stacked(df_stacked: pd.DataFrame, max_height_px: int = 520) -> str:
            """
            Render HTML estilo Paper para la tabla apilada (EPIDAT-like).
            """
            if df_stacked is None or df_stacked.empty:
                return "<div style='font-family:Arial; padding:8px;'>Sin datos para mostrar.</div>"

            headers = "".join([f"<th>{c}</th>" for c in df_stacked.columns])

            body = []
            for _, row in df_stacked.iterrows():
                is_group = str(row["VALORES"]).strip() != ""
                tr_class = "group-start" if is_group else ""
                tds = []
                for j, c in enumerate(df_stacked.columns):
                    val = row[c]
                    if j <= 1:
                        td_class = "left"
                    else:
                        td_class = "num"
                    tds.append(f"<td class='{td_class}'>{val}</td>")
                body.append(f"<tr class='{tr_class}'>{''.join(tds)}</tr>")

            html = f"""
            <html>
            <head>
              <style>
                :root {{
                  --accent: #0B3A82;
                  --ink: #111827;
                  --muted: #6B7280;
                  --line: rgba(17,24,39,.12);
                  --line2: rgba(11,58,130,.22);
                  --bg: #ffffff;
                  --zebra: rgba(17,24,39,.02);
                  --groupbg: rgba(11,58,130,.06);
                  --leftbg: rgba(0,0,0,.04);
                }}

                body {{
                  margin:0; padding:0; background:transparent;
                  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                  color: var(--ink);
                }}

                .wrap {{
                  width: 100%;
                  max-height: {max_height_px}px;
                  overflow: auto;
                  padding: 2px;
                }}

                table.paper {{
                  width: 100%;
                  border-collapse: separate;
                  border-spacing: 0;
                  background: var(--bg);
                  font-size: 0.95rem;
                  border: 1px solid var(--line);
                  border-radius: 12px;
                  overflow: hidden;
                  box-shadow: 0 1px 10px rgba(17,24,39,.05);
                }}

                th, td {{
                  padding: 0.70rem 0.85rem;
                  border-bottom: 1px solid var(--line);
                  white-space: nowrap;
                  vertical-align: middle;
                }}

                thead th {{
                  position: sticky;
                  top: 0;
                  z-index: 2;
                  text-align: left;
                  font-weight: 800;
                  color: var(--accent);
                  background: linear-gradient(to bottom, #ffffff, rgba(11,58,130,.02));
                  border-bottom: 2px solid var(--line2);
                }}

                td.left {{ text-align: left; }}
                td.num  {{ text-align: right; font-variant-numeric: tabular-nums; }}

                /* Zebra */
                tbody tr:nth-child(even) {{ background: var(--zebra); }}

                /* Inicio de grupo (cuando VALORES no está vacío) */
                tr.group-start td {{
                  border-top: 2px solid var(--line2);
                }}

                /* Estilo tipo EPIDAT: columna VALORES sombreada */
                tbody td:first-child {{
                  background: var(--leftbg);
                  font-weight: 700;
                  color: #374151;
                }}

                /* Segunda columna (Métrica) ligeramente resaltada */
                tbody td:nth-child(2) {{
                  color: var(--muted);
                  font-weight: 700;
                }}

              </style>
            </head>
            <body>
              <div class="wrap">
                <table class="paper">
                  <thead><tr>{headers}</tr></thead>
                  <tbody>{''.join(body)}</tbody>
                </table>
              </div>
            </body>
            </html>
            """
            return html
        
        def render_crosstab_view(sub_df, r_var, c_var, mets, context_key):
            if sub_df.empty:
                st.warning("No hay datos para este grupo.")
                return
            res = generate_crosstab_analysis(sub_df, r_var, c_var, mets)
            if res['formatted_df'].empty:
                st.warning("No se pudo generar la tabla (datos insuficientes).")
                return
            st.write("📋 **Tabla Cruzada**")
            
            html_page = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
{res.get("table_css","")}
</head>
<body>
{res.get("table_html","")}
</body>
</html>
"""
            
            components.html(
                html_page,
                height=620,
                scrolling=True
            )
            
            # Construir tabla apilada para exportación (matching visual display)
            def _build_export_stacked_df(res, mets):
                """Build stacked DataFrame for export matching the visual table."""
                n_df = res.get('raw_n')
                row_pct = res.get('row_pct')
                col_pct = res.get('col_pct')
                total_pct = res.get('total_pct')
                
                if n_df is None or n_df.empty:
                    return pd.DataFrame()
                
                metric_order = ["n", "row_pct", "col_pct", "total_pct"]
                mets_sorted = [m for m in metric_order if m in mets]
                if not mets_sorted:
                    mets_sorted = ["n"]
                
                metric_labels = {
                    "n": "N",
                    "row_pct": "% Fila",
                    "col_pct": "% Columna",
                    "total_pct": "% Total"
                }
                
                rows = []
                for idx in n_df.index:
                    for i, m in enumerate(mets_sorted):
                        row_data = {"VALORES": str(idx) if i == 0 else "", "Métrica": metric_labels[m]}
                        
                        if m == "n":
                            mat = n_df
                        elif m == "row_pct":
                            mat = row_pct
                        elif m == "col_pct":
                            mat = col_pct
                        else:
                            mat = total_pct
                        
                        if mat is not None:
                            for col in n_df.columns:
                                val = mat.loc[idx, col]
                                if m == "n":
                                    row_data[str(col)] = "" if pd.isna(val) else int(val)
                                else:
                                    row_data[str(col)] = "" if pd.isna(val) else f"{val:.2f}"
                        
                        rows.append(row_data)
                
                cols_order = ["VALORES", "Métrica"] + [str(c) for c in n_df.columns]
                return pd.DataFrame(rows, columns=cols_order)
            
            df_stacked = _build_export_stacked_df(res, mets)
            
            # Exportar tabla apilada completa (matching visual display)
            boton_guardar_tabla(
                df_stacked,
                f"Contingencia_{r_var}_vs_{c_var}_{context_key}",
                f"btn_ct_{context_key}",
                orientacion="Horizontal (como SPSS)"
            )

            ai_actions_for_result(
                res.get('raw_n'), 
                f"Contingencia: {r_var} vs {c_var}", 
                notas=f"Tabla cruzada que analiza la relación entre {r_var} y {c_var}. {res.get('analysis_text','')}",
                key=f"ai_ct_{context_key}"
            )

            st.info(f"💡 **Análisis Inteligente (resumen):**\n\n{res.get('analysis_text','')}")
            mostrar_interpretacion_ui(
                generar_analisis_contingencia(sub_df, r_var, c_var),
                key=f"interp_ct_{context_key}"
            )
            with st.expander("🎨 Ver Mapa de Calor", expanded=False):
                raw_matrix = res['raw_n'].drop(index='TOTAL', columns='TOTAL', errors='ignore')
                fig = px.imshow(raw_matrix, text_auto=True, aspect="auto", color_continuous_scale="Viridis", title=f"Mapa de Calor: {r_var} vs {c_var}")
                st.plotly_chart(fig, use_container_width=True, key=f"heatmap_{context_key}")
                boton_guardar_grafico(fig, f"Heatmap_{r_var}_vs_{c_var}_{context_key}", f"btn_hm_{context_key}")
        
        
        if row_var and col_var:
            if seg_var == "(Ninguno)":
                render_crosstab_view(df, row_var, col_var, metrics, "global")
            else:
                grupos = sorted(df[seg_var].dropna().unique())
                tabs = st.tabs([f"📂 {g}" for g in grupos])
                for i, grupo in enumerate(grupos):
                    with tabs[i]:
                        st.caption(f"Análisis para subgrupo: **{seg_var} = {grupo}**")
                        df_filtered = df[df[seg_var] == grupo]
                        render_crosstab_view(df_filtered, row_var, col_var, metrics, f"seg_{i}")
        else:
            st.info("Seleccione variables de Fila y Columna para comenzar.")
    
    # -------------------------------------------------------------------------
    # PESTAÑA 4: CORRELACIONES (PEARSON / SPEARMAN)
    # -------------------------------------------------------------------------
    with tab_correlaciones:
        st.markdown("## 🔗 Correlaciones")
        st.caption("Analiza asociación entre variables numéricas (Pearson y/o Spearman). Incluye filtro (subconjuntos).")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No se encontraron variables numéricas en el dataset.")
        else:
            # -------------------------------
            # UI principal
            # -------------------------------
            colA, colB, colC, colD = st.columns([2.4, 1.2, 1.2, 1.2])

            with colA:
                vars_corr = st.multiselect(
                    "Variables numéricas para analizar:",
                    options=numeric_cols,
                    default=[v for v in (selected_vars or []) if v in numeric_cols][:8] or numeric_cols[:min(6, len(numeric_cols))],
                    help="Recomendación: 2–10 variables para lectura clara."
                )

            cat_cols_corr = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            with colB:
                segmento_corr = st.selectbox(
                    "🔷 Segmentar resultados (opcional):",
                    options=["(General - Sin Segmentar)"] + cat_cols_corr,
                    index=0
                )

            with colC:
                use_pearson = st.checkbox("Pearson", value=True)
                use_spearman = st.checkbox("Spearman", value=False)

            with colD:
                include_n = st.checkbox("Mostrar N", value=True, help="Presenta N (observaciones válidas por par).")
                alpha = st.slider("α", 0.01, 0.10, 0.05, 0.01)

            st.divider()

            # -------------------------------
            # FILTRO
            # -------------------------------
            with st.expander("🧩 Filtro", expanded=False):
                st.markdown(
                    "**¿Para qué sirve?** Un *filtro* restringe el análisis a un subconjunto de registros. "
                    "Ejemplos en salud: `Edad >= 18` **Y** `Sexo = Mujer` (solo adultas), o `IMC > 30` **O** `HTA = Sí`."
                )

                key = "corr_filter"
                if f"{key}_rules" not in st.session_state:
                    st.session_state[f"{key}_rules"] = []
                if f"{key}_comb" not in st.session_state:
                    st.session_state[f"{key}_comb"] = "Y"
                if f"{key}_enabled" not in st.session_state:
                    st.session_state[f"{key}_enabled"] = False

                st.session_state[f"{key}_enabled"] = st.checkbox("Activar filtro", value=st.session_state[f"{key}_enabled"])
                st.session_state[f"{key}_comb"] = st.radio("Combinar reglas con:", ["Y", "O"], horizontal=True, index=0 if st.session_state[f"{key}_comb"]=="Y" else 1)

                fcol1, fcol2, fcol3, fcol4 = st.columns([1.6, 1.0, 1.4, 1.0])

                with fcol1:
                    f_var = st.selectbox("Variable", options=list(df.columns), key="corr_f_var")
                ops = _epidata_get_ops_for_series(df[f_var])

                with fcol2:
                    f_op = st.selectbox("Operador", options=ops, key="corr_f_op")


                # --- Reset del valor cuando cambia Variable u Operador (evita que quede "pegado") ---
                _prev_var_key = "corr_filter_prev_var"
                _prev_op_key = "corr_filter_prev_op"

                if _prev_var_key not in st.session_state:
                    st.session_state[_prev_var_key] = None
                if _prev_op_key not in st.session_state:
                    st.session_state[_prev_op_key] = None

                if st.session_state[_prev_var_key] != f_var or st.session_state[_prev_op_key] != f_op:
                    # limpiar posibles widgets previos
                    for k in [
                        "corr_val_text",
                        "corr_val_num",
                        "corr_val_cat",
                        "corr_val_list",
                        "corr_val_between_lo",
                        "corr_val_between_hi",
                        "corr_val_list_csv",
                    ]:
                        st.session_state.pop(k, None)

                    st.session_state[_prev_var_key] = f_var
                    st.session_state[_prev_op_key] = f_op

                # --- Widget inteligente para "Valor" ---
                s_sel = df[f_var]
                is_num = pd.api.types.is_numeric_dtype(s_sel)

                # preparar lista de opciones (para categóricas o listas)
                # (para no explotar memoria, limitamos unique si es enorme)
                uniq_raw = s_sel.dropna().unique()
                uniq_count = len(uniq_raw)

                # convertimos a valores "display" estables (strings) pero preservando números
                # si son numéricos, mejor conservar como float/int en el widget numérico
                uniq_as_str = sorted([str(x) for x in uniq_raw])[:2000]

                f_val_str = ""  # <- ESTE es el string final que se guardará en la regla

                with fcol3:
                    if f_op in ("es NA", "no es NA"):
                        st.text_input("Valor (no aplica)", value="(no requerido)", disabled=True, key="corr_val_text")
                        f_val_str = ""
                    else:
                        if is_num:
                            # NUMÉRICO
                            if f_op == "entre":
                                c1, c2 = st.columns(2)
                                with c1:
                                    lo = st.number_input("Mín", value=0.0, key="corr_val_between_lo")
                                with c2:
                                    hi = st.number_input("Máx", value=0.0, key="corr_val_between_hi")
                                f_val_str = f"{lo},{hi}"

                            elif f_op == "en lista":
                                # si pocos valores únicos, multiselect; si demasiados, CSV
                                if uniq_count <= 200:
                                    # usar valores reales (numéricos) para seleccionar
                                    opts = sorted([float(x) for x in pd.to_numeric(s_sel.dropna(), errors="coerce").dropna().unique()])
                                    picked = st.multiselect("Valores (lista)", options=opts, default=[], key="corr_val_list")
                                    f_val_str = ",".join([str(v) for v in picked])
                                else:
                                    csv = st.text_input("Valores (CSV)", value="", key="corr_val_list_csv", help="Ej: 1,2,3")
                                    f_val_str = csv.strip()

                            else:
                                # operadores simples (=, ≠, >, >=, <, <=)
                                numv = st.number_input("Valor", value=0.0, key="corr_val_num")
                                f_val_str = str(numv)

                        else:
                            # CATEGÓRICO / TEXTO
                            if f_op in ("=", "≠"):
                                if uniq_count == 0:
                                    st.text_input("Valor", value="", key="corr_val_text")
                                    f_val_str = ""
                                else:
                                    val = st.selectbox(
                                        "Valor",
                                        options=uniq_as_str,
                                        index=0,
                                        key="corr_val_cat",
                                        help=f"Categorías detectadas: {uniq_count}"
                                    )
                                    f_val_str = str(val)

                            elif f_op == "en lista":
                                picked = st.multiselect(
                                    "Valores (lista)",
                                    options=uniq_as_str,
                                    default=[],
                                    key="corr_val_list",
                                    help=f"Categorías detectadas: {uniq_count} (elige varias)"
                                )
                                f_val_str = ",".join([str(v) for v in picked])

                            else:
                                # contiene / no contiene / empieza con / termina con
                                txt = st.text_input("Texto", value="", key="corr_val_text")
                                f_val_str = txt.strip()

                with fcol4:
                    if st.button("Agregar regla", use_container_width=True):
                        st.session_state[f"{key}_rules"].append({"var": f_var, "op": f_op, "val": f_val_str})

                b1, b2, b3 = st.columns([1.2, 1.2, 1.2])
                with b1:
                    if st.button("Eliminar última", use_container_width=True) and st.session_state[f"{key}_rules"]:
                        st.session_state[f"{key}_rules"].pop()
                with b2:
                    if st.button("Limpiar reglas", use_container_width=True):
                        st.session_state[f"{key}_rules"] = []
                with b3:
                    st.write("")

                if st.session_state[f"{key}_rules"]:
                    st.markdown("**Reglas actuales:**")
                    st.dataframe(pd.DataFrame(st.session_state[f"{key}_rules"]), use_container_width=True, hide_index=True)
                else:
                    st.info("No hay reglas. Si activas el filtro sin reglas, no se aplicará nada.")

            # Aplicar filtro SOLO para correlaciones
            df_base = df
            filtro_txt = "Sin filtro."
            if st.session_state.get("corr_filter_enabled", False) and st.session_state.get("corr_filter_rules"):
                df_base, filtro_txt = _epidata_apply_filter(df, st.session_state["corr_filter_rules"], st.session_state.get("corr_filter_comb", "Y"))

            st.caption(f"📌 {filtro_txt}  |  Registros: {len(df)} → {len(df_base)}")

            st.divider()

            # -------------------------------
            # Validación método
            # -------------------------------
            metodos = []
            if use_pearson:
                metodos.append(("Pearson", "pearson"))
            if use_spearman:
                metodos.append(("Spearman", "spearman"))

            if len(vars_corr) < 2:
                st.info("Selecciona al menos **2 variables numéricas**.")
            elif not metodos:
                st.warning("Activa al menos un método: **Pearson** y/o **Spearman**.")
            elif len(df_base) < 3:
                st.warning("Con el filtro actual hay muy pocos registros para calcular correlaciones (se recomienda N ≥ 3).")
            else:
                import streamlit.components.v1 as components

                def _render_correlacion_bloque(df_in: pd.DataFrame, titulo: str, method_label: str, method_key: str, key_prefix: str):
                    r_df, p_df, n_df = _calcular_matrices_correlacion(df_in, vars_corr, metodo=method_key)

                    st.markdown(f"### {titulo} — {method_label}")
                    st.caption("Formato SPSS: por cada variable se muestra coeficiente (r), p-valor y opcionalmente N (por pares).")

                    html_tbl = _render_tabla_correlacion_html(r_df, p_df, n_df, metodo=method_key, alpha=alpha, include_n=include_n)
                    altura = min(200 + (len(vars_corr) * (3 if include_n else 2) * 44), 820)
                    components.html(html_tbl, height=altura, scrolling=True)

                    df_export_raw = _construir_tabla_correlacion_export(r_df, p_df, n_df, include_n=include_n)
                    
                    # -----------------------------------------------------
                    # Formato para que Excel se vea bien (3 decimales, p<0.001)
                    # -----------------------------------------------------
                    df_export = df_export_raw.copy()
                    vars_cols = [c for c in df_export.columns if c not in ["Variable","Métrica"]]

                    def _fmt_r(x):
                        return "" if pd.isna(x) else f"{float(x):.3f}"
                    def _fmt_p(x):
                        if pd.isna(x): return ""
                        x = float(x)
                        return "<0.001" if x < 0.001 else f"{x:.3f}"
                    def _fmt_n(x):
                        return "" if pd.isna(x) else str(int(x))

                    for i in df_export.index:
                        m = str(df_export.loc[i, "Métrica"])
                        if m == "r":
                            for c in vars_cols: df_export.loc[i, c] = _fmt_r(df_export.loc[i, c])
                        elif m == "p":
                            for c in vars_cols: df_export.loc[i, c] = _fmt_p(df_export.loc[i, c])
                        elif m == "N":
                            for c in vars_cols: df_export.loc[i, c] = _fmt_n(df_export.loc[i, c])

                    # =========================
                    # ACCIONES (Excel / Reporte / IA)
                    # =========================

                    # Nombre “bonito” y consistente
                    titulo_resultado = f"Correlaciones — {titulo} ({method_label})"

                    # 1) Exportación Excel + Añadir al reporte (usamos la versión formateada)
                    boton_guardar_tabla(
                        df_export,
                        titulo_resultado.replace(" ", "_"),
                        f"corr_export_{key_prefix}_{method_key}",
                        orientacion="Matriz (coeficiente / p-valor / N)"
                    )

                    # 2) Botones de IA
                    #    - La IA debe usar df_export_raw para tener precisión numérica
                    filtro_activo = st.session_state.get("corr_filter_enabled", False) and bool(st.session_state.get("corr_filter_rules"))
                    texto_filtro = "Sin filtro." if not filtro_activo else "Con filtro activo (subpoblación)."

                    ai_actions_for_result(
                        df_resultado=df_export_raw,
                        titulo=titulo_resultado,
                        notas=(
                            f"{texto_filtro} "
                            f"α={alpha}. include_n={include_n}. "
                            f"Variables: {', '.join(vars_corr)}."
                        ),
                        key=f"corr_ai_{key_prefix}_{method_key}"
                    )

                    with st.expander("🧠 Interpretación automática (borrador para tesis)", expanded=True):
                        texto = generar_interpretacion_correlaciones(r_df, p_df, n_df, metodo=method_key, alpha=alpha, top_k=6)
                        # si filtro activo, añadimos una línea explicativa
                        if st.session_state.get("corr_filter_enabled", False) and st.session_state.get("corr_filter_rules"):
                            texto = texto + "\n\n**Filtro aplicado:** Este análisis se calculó solo con el subconjunto filtrado (útil para analizar subpoblaciones)."
                        st.markdown(texto)

                # -------------------------------
                # Render general / segmentado
                # -------------------------------
                def _render_for_df(df_in: pd.DataFrame, titulo_base: str, prefix: str):
                    if len(metodos) == 2:
                        t1, t2 = st.tabs([metodos[0][0], metodos[1][0]])
                        with t1:
                            _render_correlacion_bloque(df_in, titulo_base, metodos[0][0], metodos[0][1], f"{prefix}_m0")
                        with t2:
                            _render_correlacion_bloque(df_in, titulo_base, metodos[1][0], metodos[1][1], f"{prefix}_m1")
                    else:
                        _render_correlacion_bloque(df_in, titulo_base, metodos[0][0], metodos[0][1], f"{prefix}_m0")

                if segmento_corr == "(General - Sin Segmentar)":
                    _render_for_df(df_base, "Correlación Global", "global")
                else:
                    grupos = sorted(df_base[segmento_corr].dropna().unique())
                    if not grupos:
                        st.warning("No hay grupos válidos para segmentación (después del filtro).")
                    else:
                        st.info(f"📂 Segmentado por: **{segmento_corr}**")
                        tabs_g = st.tabs([f"{g}" for g in grupos])
                        for i, g in enumerate(grupos):
                            with tabs_g[i]:
                                df_sub = df_base[df_base[segmento_corr] == g]
                                _render_for_df(df_sub, f"Correlación — {g}", f"seg_{i}")

    
    with tab_comparativa:
        st.markdown("### ⚔️ Comparativa de Grupos (Tabla 1)")
        cat_vars = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        group_col_comp = st.selectbox(
            "🔀 Agrupar por (Variable Categórica):",
            options=["(Ninguno)"] + cat_vars,
            index=0,
            key="group_selector_tab2",
            help="Seleccione grupos (ej: Tratamiento vs Control) para generar p-values y comparaciones."
        )
        
        if group_col_comp == "(Ninguno)":
            st.info("ℹ️ Seleccione una variable de agrupación arriba para generar la Tabla 1 con P-values.")
        else:
            n_groups = df[group_col_comp].nunique()
            if n_groups > 10:
                st.warning(f"⚠️ La variable '{group_col_comp}' tiene {n_groups} categorías. La tabla será muy ancha.")
            
            st.markdown(f"**Comparando grupos según:** `{group_col_comp}`")
            st.divider()
            
            available_vars = [c for c in df.columns if c != group_col_comp]
            default_mix = [v for v in selected_vars if v in available_vars]
            
            table1_vars = st.multiselect(
                "Seleccione variables para la Tabla 1:",
                options=available_vars,
                default=default_mix,
                key="table1_vars_selector"
            )
            
            if table1_vars:
                with st.spinner("Calculando estadísticas y P-Values..."):
                    df_table1 = generate_table_one_structure(df, table1_vars, group_col_comp)
                
                if not df_table1.empty:
                    def highlight_p(val):
                        return 'font-weight: bold; color: #2c3e50; background-color: #d5dbdb' if isinstance(val, str) and (val.startswith("<") or (val.replace('.','',1).isdigit() and float(val) < 0.05)) else ''
                    
                    st.dataframe(
                        df_table1.style.applymap(highlight_p, subset=['P-Value']).set_properties(**{'text-align': 'left'}),
                        use_container_width=True,
                        hide_index=True
                    )
                    st.caption("Nota: Para variables numéricas se muestra Mean ± SD (Paramétrico) o Median (IQR). Para categóricas n (%).")
                    
                    boton_guardar_tabla(df_table1, f"Tabla1_{group_col_comp}", "btn_tabla1", orientacion="Horizontal (como SPSS)")
                    
                    ai_actions_for_result(
                        df_table1, 
                        f"Comparativa: Grupos por {group_col_comp}", 
                        notas="Tabla 1 de comparación de características basales entre grupos con valores P.",
                        key="ai_table1"
                    )
                    
                else:
                    st.warning("No se pudieron generar resultados. Verifique que las variables tengan datos.")
            else:
                st.info("Seleccione al menos una variable para construir la tabla.")
    
    # ==============================================================================
    # PESTAÑA 3: GRÁFICOS DIAGNÓSTICOS
    # ==============================================================================
    with tab_graficos:
        st.markdown("### 📊 Inspección Visual")
        subtab_panel, subtab_comparar = st.tabs(["🔬 Panel Diagnóstico (Univariado)", "🆚 Comparación de Grupos"])
        
        with subtab_panel:
            st.caption("Diagnóstico de normalidad y evaluación de transformaciones.")
            c_sel, c_chk = st.columns([2, 1])
            with c_sel:
                var_plot = st.selectbox("Seleccione variable a diagnosticar:", options=selected_vars, key="selector_plot_var_panel")
            
            if var_plot:
                original_data = df[var_plot].dropna()
                data_plot = original_data
                is_transformed = False
                
                with c_chk:
                    st.write("")
                    apply_log = st.checkbox("🛠️ Simular Log10", help="Aplica Log10(x+1) para intentar normalizar.")
                
                if apply_log:
                    if (original_data < 0).any():
                        st.warning("⚠️ No se puede aplicar Log10: Hay valores negativos.")
                    else:
                        data_plot = np.log10(original_data + 1)
                        is_transformed = True
                        st.toast("Transformación Logarítmica Aplicada (Datos temporales)", icon="🛠️")
                
                n = len(data_plot)
                norm_res = check_normality(data_plot)
                p_val = norm_res.get('shapiro_p', 0.0)
                status_color = "#2ecc71" if p_val > 0.05 else "#e74c3c"
                status_text = "NORMAL" if p_val > 0.05 else "NO NORMAL"
                msg_trans = " (Transformado Log10)" if is_transformed else ""
                
                st.markdown(
                    f"<div style='padding: 10px; border-radius: 5px; background-color: rgba(240,242,246,0.5); border: 1px solid #ddd; margin-bottom: 15px;'>"
                    f"<b>Diagnóstico{msg_trans}:</b> <span style='color:{status_color}; font-weight:bold'>{status_text}</span> "
                    f"(Shapiro-Wilk p = <b>{p_val:.4f}</b>, n={n})</div>",
                    unsafe_allow_html=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    title_hist = f"Distribución: {var_plot}{msg_trans} | SW p={p_val:.3f}"
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(x=data_plot, histnorm='probability density', name='Histograma', marker_color='#3498db', opacity=0.6))
                    x_norm, y_norm = get_normal_curve_data(data_plot)
                    if x_norm is not None:
                        fig_hist.add_trace(go.Scatter(x=x_norm, y=y_norm, mode='lines', name='Normal Teórica', line=dict(color='orange', width=2)))
                    mean_val = data_plot.mean()
                    median_val = data_plot.median()
                    fig_hist.add_vline(x=mean_val, line_width=2, line_dash="dash", line_color="red", annotation_text="Media")
                    fig_hist.add_vline(x=median_val, line_width=2, line_dash="dot", line_color="green", annotation_text="Med")
                    fig_hist.update_layout(title=title_hist, margin=dict(l=20, r=20, t=40, b=20), height=350)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    boton_guardar_grafico(fig_hist, f"Histograma_{var_plot}", "btn_hist_panel")
                
                with col2:
                    qq_data = get_qq_coordinates(data_plot)
                    if qq_data:
                        fig_qq = go.Figure()
                        fig_qq.add_trace(go.Scatter(x=qq_data['theoretical'], y=qq_data['sample'], mode='markers', name='Datos', marker=dict(color='#8e44ad')))
                        min_x, max_x = min(qq_data['theoretical']), max(qq_data['theoretical'])
                        fig_qq.add_trace(go.Scatter(x=[min_x, max_x], y=[qq_data['slope']*min_x + qq_data['intercept'], qq_data['slope']*max_x + qq_data['intercept']], mode='lines', name='Ideal', line=dict(color='red')))
                        fig_qq.update_layout(title="Q-Q Plot (Validación Normalidad)", xaxis_title="Teórico", yaxis_title="Muestral", margin=dict(l=20, r=20, t=40, b=20), height=350)
                        st.plotly_chart(fig_qq, use_container_width=True)
                        boton_guardar_grafico(fig_qq, f"QQ_{var_plot}", "btn_qq_panel")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(x=data_plot, boxpoints='outliers', name=f"{var_plot}", marker_color='#f39c12', orientation='h'))
                    fig_box.update_layout(title="Box Plot (Outliers)", margin=dict(l=20, r=20, t=40, b=20), height=350)
                    st.plotly_chart(fig_box, use_container_width=True)
                    boton_guardar_grafico(fig_box, f"Boxplot_{var_plot}", "btn_box_panel")
                
                with col4:
                    fig_viol = go.Figure()
                    fig_viol.add_trace(go.Violin(y=data_plot, box_visible=True, line_color='black', meanline_visible=True, fillcolor='#1abc9c', opacity=0.6, x0=f"{var_plot}", points='all', jitter=0.05, pointpos=0))
                    fig_viol.update_layout(title="Violin Híbrido (Densidad + Puntos)", margin=dict(l=20, r=20, t=40, b=20), height=350)
                    st.plotly_chart(fig_viol, use_container_width=True)
                    boton_guardar_grafico(fig_viol, f"Violin_{var_plot}", "btn_violin_panel")
        
        with subtab_comparar:
            st.caption("Visualización comparativa con pruebas de hipótesis automáticas.")
            c1, c2 = st.columns(2)
            with c1:
                var_comp = st.selectbox("Variable Numérica:", options=selected_vars, key="sel_var_comp_tab3")
            with c2:
                cat_options = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                group_comp = st.selectbox("Variable de Agrupación:", options=["(Seleccione)"] + cat_options, key="sel_group_comp_tab3")
            
            if var_comp and group_comp != "(Seleccione)":
                df_c = df[[var_comp, group_comp]].dropna()
                stats_res = calculate_group_comparison(df, var_comp, group_comp)
                test_name = stats_res.get('test_used', 'N/A')
                p_val_str = stats_res.get('p_value_str', '-')
                stats_title = f"{var_comp} por {group_comp} | {test_name}: p={p_val_str}"
                
                col_a, col_b = st.columns(2)
                with col_a:
                    fig_box_c = px.box(df_c, x=group_comp, y=var_comp, color=group_comp, points="all", title=f"Dispersión: {stats_title}", color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_box_c.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_box_c, use_container_width=True)
                    boton_guardar_grafico(fig_box_c, f"Comparacion_Box_{var_comp}_{group_comp}", "btn_box_comp")
                
                with col_b:
                    fig_viol_c = px.violin(df_c, x=group_comp, y=var_comp, color=group_comp, box=True, points=False, title=f"Densidad: {stats_title}", color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_viol_c.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_viol_c, use_container_width=True)
                    boton_guardar_grafico(fig_viol_c, f"Comparacion_Violin_{var_comp}_{group_comp}", "btn_violin_comp")
                
                st.info(f"ℹ️ **Interpretación:** El gráfico izquierdo muestra cada paciente individual (puntos) y la mediana. El derecho muestra la 'forma' de los datos (densidad). El valor p (**{p_val_str}**) indica inferencia estadística.")
            elif group_comp == "(Seleccione)":
                st.info("👋 Seleccione una variable de agrupación para generar los gráficos comparativos.")
    
    # ==============================================================================
    # PESTAÑA 4: TABLA INTELIGENTE (PERSONALIZABLE PRO)
    # ==============================================================================
    with tab_inteligente:
        st.markdown("### 📑 Tabla de Estadísticos Personalizada")
        st.markdown("Configura exactamente qué métricas deseas calcular, al estilo de software estadístico profesional.")
        
        # --- CONFIGURACIÓN AVANZADA ---
        with st.expander("⚙️ Configuración de Variables y Métricas", expanded=True):
            # 1. Selección de Variables y Segmentación
            c_sel_1, c_sel_2 = st.columns(2)
            with c_sel_1:
                vars_inteligentes = st.multiselect(
                    "Variables a Analizar:", 
                    options=selected_vars, 
                    default=selected_vars[:5] if len(selected_vars) > 5 else selected_vars
                )
            with c_sel_2:
                cat_opts = ["(General - Sin Segmentar)"] + df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                segmento = st.selectbox("🔀 Segmentar Resultados por:", options=cat_opts)
            
            st.divider()
            st.markdown("**Seleccione los estadísticos a calcular:**")
            
            # 2. Panel de Métricas (4 Columnas)
            col_tend, col_disp, col_perc, col_form = st.columns(4)
            
            with col_tend:
                st.markdown("###### 🎯 Tendencia Central")
                check_n = st.checkbox("N (Conteo)", value=True)
                check_media = st.checkbox("Media", value=True)
                check_mediana = st.checkbox("Mediana", value=True)
                check_moda = st.checkbox("Moda")
                check_suma = st.checkbox("Suma")
                check_gmean = st.checkbox("Media Geométrica")
                check_ic = st.checkbox("IC 95% (Media)", value=True)
            
            with col_disp:
                st.markdown("###### 📏 Dispersión")
                check_std = st.checkbox("Desviación Típica", value=True)
                check_var = st.checkbox("Varianza")
                check_cv = st.checkbox("Coef. Variación (%)", value=True)
                check_min = st.checkbox("Mínimo", value=True)
                check_max = st.checkbox("Máximo", value=True)
                check_rango = st.checkbox("Recorrido (Rango)")
                check_iqr = st.checkbox("Rango Intercuartílico", value=True)
                check_se = st.checkbox("Error Est. Media")
            
            with col_perc:
                st.markdown("###### 📍 Percentiles")
                check_cuartiles = st.checkbox("Cuartiles (25, 50, 75)")
                check_deciles = st.checkbox("Deciles (10, 20...90)")
                custom_perc_str = st.text_input("Personalizados (ej: 5, 95, 99)", help="Separa los valores por comas")
            
            with col_form:
                st.markdown("###### 🔔 Forma y Dist.")
                check_asimetria = st.checkbox("Asimetría (Skewness)")
                check_curtosis = st.checkbox("Curtosis")
                check_norm = st.checkbox("Prueba Normalidad (p)", value=True)
        
        # --- FUNCIÓN HELPER LOCAL (Con alias seguro 'ss') ---
        import scipy.stats as ss  # IMPORTACIÓN SEGURA AQUÍ
        
        def calcular_metricas_fila(serie_datos):
            """Calcula solo lo seleccionado para una serie de datos"""
            res = {}
            datos = serie_datos.dropna()
            n = len(datos)
            
            if n == 0:
                return {}
            
            # Tendencia Central
            if check_n:
                res['N'] = n
            if check_media:
                res['Media'] = np.mean(datos)
            if check_mediana:
                res['Mediana'] = np.median(datos)
            if check_moda:
                try:
                    moda_res = ss.mode(datos, keepdims=True)  # Usamos ss
                    res['Moda'] = moda_res.mode[0]
                except:
                    res['Moda'] = np.nan
            if check_suma:
                res['Suma'] = np.sum(datos)
            if check_gmean:
                try:
                    if (datos <= 0).any():
                        res['M. Geom.'] = np.nan
                    else:
                        res['M. Geom.'] = ss.gmean(datos)  # Usamos ss
                except:
                    res['M. Geom.'] = np.nan
            if check_ic:
                try:
                    se = ss.sem(datos)  # Usamos ss (Aquí fallaba antes)
                    h = se * ss.t.ppf((1 + 0.95) / 2., n-1)
                    m = np.mean(datos)
                    res['IC 95%'] = f"[{m-h:.2f} - {m+h:.2f}]"
                except:
                    res['IC 95%'] = "-"
            
            # Dispersión
            if check_std:
                res['D.E.'] = np.std(datos, ddof=1)
            if check_var:
                res['Varianza'] = np.var(datos, ddof=1)
            if check_cv:
                mu = np.mean(datos)
                res['CV %'] = (np.std(datos, ddof=1) / mu * 100) if mu != 0 else 0
            if check_min:
                res['Mín'] = np.min(datos)
            if check_max:
                res['Máx'] = np.max(datos)
            if check_rango:
                res['Rango'] = np.max(datos) - np.min(datos)
            if check_iqr:
                res['IQR'] = np.percentile(datos, 75) - np.percentile(datos, 25)
            if check_se:
                res['E.E.M.'] = ss.sem(datos)  # Usamos ss
            
            # Forma
            if check_asimetria:
                res['Asimetría'] = ss.skew(datos)  # Usamos ss
            if check_curtosis:
                res['Curtosis'] = ss.kurtosis(datos)  # Usamos ss
            if check_norm:
                try:
                    if n < 3:
                        res['P-Normalidad'] = np.nan
                    elif n < 50:
                        _, p = ss.shapiro(datos)  # Usamos ss
                        res['P-Normalidad'] = p
                    else:
                        # KS contra normal estandarizada
                        _, p = ss.kstest((datos - np.mean(datos))/np.std(datos, ddof=1), 'norm')
                        res['P-Normalidad'] = p
                except:
                    res['P-Normalidad'] = np.nan
            
            # Percentiles
            if check_cuartiles:
                res['P25'] = np.percentile(datos, 25)
                res['P50'] = np.percentile(datos, 50)
                res['P75'] = np.percentile(datos, 75)
            
            if check_deciles:
                for d in range(10, 100, 10):
                    res[f'P{d}'] = np.percentile(datos, d)
            
            if custom_perc_str:
                try:
                    vals = [float(x.strip()) for x in custom_perc_str.split(',') if x.strip()]
                    for v in vals:
                        if 0 <= v <= 100:
                            res[f'P{int(v)}'] = np.percentile(datos, v)
                except:
                    pass
            
            return res
        
        # --- MOTOR DE RENDERIZADO AISLADO (IFRAME) ---
        def renderizar_tabla_tesis_aislada(df_plano):
            """
            Genera un documento HTML independiente con el CSS del usuario.
            Al usarse en un IFrame, Streamlit NO puede sobrescribir los estilos.
            """
            df_indexed = df_plano.set_index('Variable')
            # Re-estructurar MultiIndex
            grupos_map = {
                'Tendencia Central': ['N', 'Media', 'Mediana', 'Moda', 'Suma', 'M. Geom.', 'IC 95%'],
                'Dispersión': ['D.E.', 'Varianza', 'CV %', 'Mín', 'Máx', 'Rango', 'IQR', 'E.E.M.'],
                'Forma y Dist.': ['Asimetría', 'Curtosis', 'P-Normalidad'],
                'Percentiles': [c for c in df_indexed.columns if c.startswith('P') and c[1:].isdigit()]
            }
            new_cols = []
            for col in df_indexed.columns:
                grupo = next((g for g, cs in grupos_map.items() if col in cs), 'Otros')
                new_cols.append((grupo, col))
            df_indexed.columns = pd.MultiIndex.from_tuples(new_cols)
            # --- HTML + CSS EXACTO DEL USUARIO ---
            html_content = f"""
            <html>
            <head>
              <link rel="preconnect" href="https://fonts.googleapis.com">
              <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
              <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap" rel="stylesheet">

              <style>
                :root {{
                  --paper-blue: #002676;
                  --line: rgba(0, 0, 0, 0.10);
                  --line-strong: rgba(0, 38, 118, 0.20);
                  --zebra: rgba(0, 0, 0, 0.02);
                }}

                body {{
                  background-color: transparent;
                  margin: 0;
                  padding: 0;
                }}

                .tabla-contenedor {{
                  width: 100%;
                  overflow-x: auto;
                  font-family: 'Source Sans Pro', Helvetica, Arial, sans-serif;
                }}

                /* TABLA estilo "paper" */
                table.tabla-paper {{
                  width: 100%;
                  border-collapse: collapse;
                  margin-bottom: 1.5rem;
                  font-size: 0.92rem;
                  background: white;
                }}

                table.tabla-paper th,
                table.tabla-paper td {{
                  padding: 0.65rem 0.75rem;
                  border-bottom: 1px solid var(--line);   /* SOLO línea horizontal */
                  vertical-align: middle;
                  white-space: nowrap;                   /* evita saltos raros */
                }}

                /* Encabezados */
                table.tabla-paper thead th {{
                  text-align: left;
                  font-weight: 600;
                  color: var(--paper-blue);
                  border-bottom: 2px solid var(--line-strong);
                  background: #fff;
                }}

                /* MultiIndex: fila 1 (grupos) */
                table.tabla-paper thead tr:nth-child(1) th {{
                  background: #f7f9fc;
                  text-transform: uppercase;
                  font-size: 0.78rem;
                  letter-spacing: 0.04em;
                }}

                /* MultiIndex: fila 2 (métricas) */
                table.tabla-paper thead tr:nth-child(2) th {{
                  background: #fff;
                  font-size: 0.86rem;
                }}

                /* Primera columna (Variable) */
                table.tabla-paper tbody th {{
                  text-align: left;
                  font-weight: 600;
                  color: #333;
                  background: #fff;
                }}

                /* Números a la derecha */
                table.tabla-paper tbody td {{
                  text-align: right;
                }}

                /* Zebra sutil */
                table.tabla-paper tbody tr:nth-child(even) {{
                  background: var(--zebra);
                }}

                table.tabla-paper tbody tr:hover {{
                  background: rgba(0, 38, 118, 0.04);
                }}
              </style>
            </head>

            <body>
              <div class="tabla-contenedor">
                {df_indexed.to_html(float_format="%.2f", border=0, classes="tabla-paper")}
              </div>
            </body>
            </html>
            """
            return html_content

        def renderizar_tabla_tesis_vertical_aislada(df_plano: pd.DataFrame) -> str:
            """
            Tabla estilo tesis: estadísticos en filas (hacia abajo) y variables en columnas.
            Ideal para presentar 1–3 variables en una tesis.
            """
            df_indexed = df_plano.set_index("Variable")  # filas=variables, cols=métricas
            metrics_present = list(df_indexed.columns)
            variables = list(df_indexed.index)

            # Orden y agrupación (filtra solo lo que existe)
            grupos_map = {
                "Tendencia Central": ["N", "Media", "Mediana", "Moda", "Suma", "M. Geom.", "IC 95%"],
                "Dispersión": ["D.E.", "Varianza", "CV %", "Mín", "Máx", "Rango", "IQR", "E.E.M."],
                "Forma y Dist.": ["Asimetría", "Curtosis", "P-Normalidad"],
                "Percentiles": [c for c in metrics_present if c.startswith("P") and c[1:].isdigit()],
            }

            ordered_groups = []
            for g, cols in grupos_map.items():
                cols_ok = [c for c in cols if c in metrics_present]
                if cols_ok:
                    ordered_groups.append((g, cols_ok))

            def fmt(metric: str, val):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return ""
                if metric == "N":
                    try:
                        return f"{int(val)}"
                    except Exception:
                        return str(val)
                if metric == "P-Normalidad":
                    try:
                        v = float(val)
                        return "<0.001" if v < 0.001 else f"{v:.3f}"
                    except Exception:
                        return str(val)
                if metric == "IC 95%":
                    return str(val)  # ya viene como string tipo [a - b]
                # resto numéricos
                try:
                    return f"{float(val):.2f}"
                except Exception:
                    return str(val)

            ncols = 1 + len(variables)

            # Construcción HTML manual (para filas de grupo tipo "paper")
            header_th = "".join([f"<th>{v}</th>" for v in variables])

            body_rows = []
            for group_name, cols_in_group in ordered_groups:
                body_rows.append(
                    f"<tr class='grupo'><td colspan='{ncols}'>{group_name.upper()}</td></tr>"
                )
                for metric in cols_in_group:
                    tds = [f"<td class='metric'>{metric}</td>"]
                    for var in variables:
                        val = df_indexed.loc[var, metric]
                        tds.append(f"<td class='num'>{fmt(metric, val)}</td>")
                    body_rows.append("<tr>" + "".join(tds) + "</tr>")

            html_table = f"""
            <table class="tabla-vertical">
              <thead>
                <tr>
                  <th class="metric-head">Estadístico</th>
                  {header_th}
                </tr>
              </thead>
              <tbody>
                {''.join(body_rows)}
              </tbody>
            </table>
            """

            html_content = f"""
            <html>
            <head>
              <link rel="preconnect" href="https://fonts.googleapis.com">
              <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
              <link href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap" rel="stylesheet">

              <style>
                :root{{
                  --accent: #0B3A82;
                  --accent-soft: rgba(11,58,130,.08);
                  --ink: #111827;
                  --muted: #6B7280;
                  --line: rgba(17,24,39,.12);
                  --line2: rgba(11,58,130,.22);
                  --bg: #ffffff;
                  --zebra: rgba(17,24,39,.02);
                  --group: rgba(11,58,130,.06);
                }}

                body{{
                  margin:0;
                  padding:0;
                  background: transparent;
                  font-family: 'Source Sans Pro', Helvetica, Arial, sans-serif;
                  color: var(--ink);
                }}

                .wrap{{ width:100%; overflow-x:auto; padding: 0.25rem 0; }}

                table.tabla-vertical{{
                  width:100%;
                  border-collapse: separate;
                  border-spacing: 0;
                  background: var(--bg);
                  font-size: 0.95rem;
                  border: 1px solid var(--line);
                  border-radius: 12px;
                  overflow: hidden;
                  box-shadow: 0 1px 10px rgba(17,24,39,.05);
                }}

                table.tabla-vertical th,
                table.tabla-vertical td{{
                  padding: 0.70rem 0.85rem;
                  border-bottom: 1px solid var(--line);
                  vertical-align: middle;
                  white-space: nowrap;
                }}

                table.tabla-vertical thead th{{
                  text-align:left;
                  font-weight: 700;
                  color: var(--accent);
                  background: linear-gradient(to bottom, #ffffff, rgba(11,58,130,.02));
                  border-bottom: 2px solid var(--line2);
                }}

                table.tabla-vertical thead th:not(.metric-head){{
                  color: var(--ink);
                  font-weight: 700;
                }}

                table.tabla-vertical td.metric{{
                  font-weight: 700;
                  color: var(--ink);
                }}

                table.tabla-vertical td.num{{
                  text-align:right;
                  font-variant-numeric: tabular-nums;
                }}

                table.tabla-vertical tbody tr:nth-child(even){{
                  background: var(--zebra);
                }}

                table.tabla-vertical tbody tr:hover{{
                  background: rgba(11,58,130,.04);
                }}

                tr.grupo td{{
                  background: var(--group);
                  font-weight: 800;
                  letter-spacing: .08em;
                  text-transform: uppercase;
                  font-size: 0.78rem;
                  color: var(--muted);
                  border-top: 1px solid var(--line2);
                  border-bottom: 1px solid var(--line2);
                  position: relative;
                }}

                tr.grupo td::before{{
                  content:"";
                  position:absolute;
                  left:0; top:0; bottom:0;
                  width: 6px;
                  background: var(--accent);
                  opacity: .35;
                }}
              </style>
            </head>
            <body>
              <div class="wrap">
                {html_table}
              </div>
            </body>
            </html>
            """
            return html_content

        # --- LÓGICA DE VISUALIZACIÓN EN LA PESTAÑA ---
        if vars_inteligentes:
            st.divider()
            import scipy.stats as ss
            
            col_v, _ = st.columns([1, 3])
            vista_p = col_v.toggle("📰 Vista Académica (Tesis)", value=True)
            
            # Caso General
            if segmento == "(General - Sin Segmentar)":
                filas = []
                for v in vars_inteligentes:
                    m = calcular_metricas_fila(df[v])
                    m['Variable'] = v
                    filas.append(m)
                df_res = pd.DataFrame(filas)
                
                if not df_res.empty:
                    st.markdown(f"##### 📊 Resultados Globales (N={len(df)})")
                    if vista_p:
                        import streamlit.components.v1 as components
                        
                        # Selector de orientación
                        orientacion = st.radio(
                            "Orientación de la tabla (para tesis):",
                            ["Horizontal (como SPSS)", "Vertical (estadísticos hacia abajo)"],
                            horizontal=True
                        )
                        
                        # Renderizar según orientación elegida
                        if orientacion == "Vertical (estadísticos hacia abajo)":
                            html_completo = renderizar_tabla_tesis_vertical_aislada(df_res)
                        else:
                            html_completo = renderizar_tabla_tesis_aislada(df_res)
                        
                        components.html(html_completo, height=520, scrolling=True)
                    else:
                        st.dataframe(df_res, use_container_width=True)
                    
                    # Pasar orientación al botón de exportar (usa horizontal si vista académica está desactivada)
                    orientacion_export = orientacion if vista_p else "Horizontal (como SPSS)"
                    boton_guardar_tabla(df_res, "Descriptiva_Global", "btn_dg", orientacion=orientacion_export)
                    
                    ai_actions_for_result(
                        df_res, 
                        "Estadísticos Globales Personalizados", 
                        notas="Resumen estadístico personalizado con métricas seleccionadas por el usuario.",
                        key="ai_desc_custom_global"
                    )
            
            # Caso Segmentado
            else:
                grupos = sorted(df[segmento].dropna().unique())
                st.info(f"📂 Segmentado por: **{segmento}**")
                tbs = st.tabs([f"{g}" for g in grupos])
                for i, g in enumerate(grupos):
                    with tbs[i]:
                        df_sub = df[df[segmento] == g]
                        filas_g = []
                        for v in vars_inteligentes:
                            m = calcular_metricas_fila(df_sub[v])
                            m['Variable'] = v
                            filas_g.append(m)
                        df_res_g = pd.DataFrame(filas_g)
                        if not df_res_g.empty:
                            if vista_p:
                                import streamlit.components.v1 as components
                                
                                # Selector de orientación
                                orientacion_g = st.radio(
                                    "Orientación de la tabla (para tesis):",
                                    ["Horizontal (como SPSS)", "Vertical (estadísticos hacia abajo)"],
                                    horizontal=True,
                                    key=f"orientacion_seg_{i}"
                                )
                                
                                # Renderizar según orientación elegida
                                if orientacion_g == "Vertical (estadísticos hacia abajo)":
                                    html_completo = renderizar_tabla_tesis_vertical_aislada(df_res_g)
                                else:
                                    html_completo = renderizar_tabla_tesis_aislada(df_res_g)
                                
                                components.html(html_completo, height=520, scrolling=True)
                            else:
                                st.dataframe(df_res_g, use_container_width=True)
                            
                            # Pasar orientación al botón de exportar (usa horizontal si vista académica está desactivada)
                            orientacion_g_export = orientacion_g if vista_p else "Horizontal (como SPSS)"
                            boton_guardar_tabla(df_res_g, f"Desc_{g}", f"btn_{i}", orientacion=orientacion_g_export)
                            
                            ai_actions_for_result(
                                df_res_g, 
                                f"Estadísticos: {g}", 
                                notas=f"Resumen estadístico personalizado para el grupo {g}.",
                                key=f"ai_desc_custom_{i}"
                            )
        else:
            st.info("👈 Selecciona variables numéricas arriba para comenzar.")

# =============================================================================
# Helper entrypoint para `ejecutar_modulo`
# =============================================================================
def render():
    import streamlit as st
    import inspect
    df = st.session_state.get("df_principal")
    sig = inspect.signature(render_descriptiva)
    if len(sig.parameters) >= 1:
        return render_descriptiva(df)
    else:
        return render_descriptiva()
