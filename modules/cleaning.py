import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modules.utils import boton_guardar_grafico, boton_guardar_tabla

def render_limpieza():
    """
    Renderiza la interfaz de limpieza y carga de datos.
    Incluye manejo robusto de errores y optimización con st.form.
    """
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🛠️ Data Studio: Ingeniería de Datos")
    
    # 1. Carga de Archivo (ROBUSTO)
    with st.expander("📂 Cargar Archivo", expanded=True):
        uploaded_file = st.file_uploader("Sube tu Excel (.xlsx)", type=["xlsx", "xls"])
        
        if uploaded_file:
            try:
                # Intento de lectura seguro
                xls = pd.ExcelFile(uploaded_file)
                sheet = st.selectbox("Selecciona la Hoja", xls.sheet_names)
                
                if st.button("📥 Cargar Datos"):
                    try:
                        df = pd.read_excel(uploaded_file, sheet_name=sheet)
                        
                        # Limpieza Anti-KoboToolbox básica
                        k = ["start", "end", "device", "index", "uuid", "submission", "time", "date", "consentimiento", "latitude", "longitude", "gps", "geo", "altitude", "precision", "hola", "gracias", "bienvenido", "note", "intro"]
                        df_clean = df[[c for c in df.columns if not any(x in c.lower() for x in k)]].copy()
                        
                        st.session_state.df_principal = df_clean
                        st.session_state.historial_cambios = [] 
                        st.success(f"✅ Datos cargados: {df_clean.shape[0]} filas, {df_clean.shape[1]} columnas.")
                        st.rerun()
                        
                    except Exception as e_read:
                        st.error(f"Error al procesar la hoja seleccionada: {e_read}")
                        
            except ValueError as e_val:
                st.error("Error: Archivo inválido o corrupto. Por favor, verifica que sea un Excel válido.")
            except Exception as e:
                st.error(f"Error: Archivo inválido o corrupto. Detalle: {e}")
                
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Herramientas Avanzadas
    if st.session_state.df_principal is not None:
        df_work = st.session_state.df_principal
        
        # Tabs Principales
        tab1, tab2, tab3, tab4 = st.tabs(["🧹 Limpieza Inteligente", "⚙️ Transformación & Fórmulas", "🔍 Filtros Avanzados", "🕵️ Auditoría de Calidad"])
        
        # --- TAB 1: LIMPIEZA INTELIGENTE & EDICIÓN ---
        with tab1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # 1. EXCEL INTERACTIVO (EDICIÓN MANUAL) - OPTIMIZADO CON FORM
            st.subheader("📝 Editor Interactivo (Excel)")
            st.caption("Edita celdas, añade filas o bórralas. Los cambios se aplicarán solo al guardar.")
            
            with st.form("form_edicion"):
                # Editor con num_rows="dynamic" para añadir/borrar
                edited_df = st.data_editor(
                    df_work,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="data_editor_main"
                )
                
                submit_btn = st.form_submit_button("💾 Guardar Cambios Manuales")
                
                if submit_btn:
                    if not edited_df.equals(df_work):
                        st.session_state.historial_cambios.append(df_work.copy())
                        st.session_state.df_principal = edited_df
                        st.success("✅ Cambios guardados correctamente.")
                        st.rerun()
                    else:
                        st.info("No se detectaron cambios para guardar.")

            st.markdown("---")
            
            # 2. HERRAMIENTAS AUTOMÁTICAS (CON PREVIEW)
            c1, c2 = st.columns(2)
            
            # COLUMNA IZQUIERDA: VALORES FALTANTES
            with c1:
                st.subheader("Valores Faltantes (Nulos)")
                total_nulls = df_work.isnull().sum().sum()
                st.metric("Total Celdas Vacías", total_nulls)
                
                col_null = st.selectbox("Columna a corregir", [c for c in df_work.columns if df_work[c].isnull().sum() > 0], key="sel_null_col")
                
                if col_null:
                    n_nulos = df_work[col_null].isnull().sum()
                    st.info(f"Columna '{col_null}' tiene {n_nulos} nulos.")
                    method_null = st.selectbox("Acción", ["Eliminar filas vacías", "Rellenar con Media", "Rellenar con Mediana", "Rellenar con Anterior (FFill)"], key="sel_null_method")
                    
                    if method_null == "Eliminar filas vacías":
                        # Lógica de PREVIEW
                        rows_to_drop = df_work[df_work[col_null].isnull()]
                        
                        if st.button("🔍 Vista Previa Eliminación", key="btn_prev_null"):
                            st.session_state['preview_nulls'] = True
                        
                        if st.session_state.get('preview_nulls'):
                            st.warning(f"⚠️ Se eliminarán estas {len(rows_to_drop)} filas:")
                            st.dataframe(rows_to_drop.head())
                            
                            if st.button("🔴 Confirmar y Eliminar", key="btn_conf_null"):
                                st.session_state.historial_cambios.append(df_work.copy())
                                st.session_state.df_principal = df_work.dropna(subset=[col_null])
                                st.session_state['preview_nulls'] = False # Reset
                                st.success("Filas eliminadas.")
                                st.rerun()
                    else:
                        # Imputación directa (menos destructiva)
                        if st.button("Aplicar Imputación", key="btn_imp_null"):
                            st.session_state.historial_cambios.append(df_work.copy())
                            if "Media" in method_null:
                                val = df_work[col_null].mean() if pd.api.types.is_numeric_dtype(df_work[col_null]) else None
                                if val is not None: st.session_state.df_principal[col_null] = df_work[col_null].fillna(val)
                            elif "Mediana" in method_null:
                                val = df_work[col_null].median() if pd.api.types.is_numeric_dtype(df_work[col_null]) else None
                                if val is not None: st.session_state.df_principal[col_null] = df_work[col_null].fillna(val)
                            elif "FFill" in method_null:
                                st.session_state.df_principal[col_null] = df_work[col_null].ffill()
                            st.success("Imputación realizada.")
                            st.rerun()

            # COLUMNA DERECHA: OUTLIERS & COLUMNAS
            with c2:
                st.subheader("Outliers & Columnas")
                
                # Gestión de Columnas
                with st.expander("🗑️ Eliminar Columnas"):
                    col_to_drop = st.multiselect("Selecciona columnas a borrar", df_work.columns)
                    if col_to_drop and st.button("Borrar Columnas Seleccionadas"):
                        st.session_state.historial_cambios.append(df_work.copy())
                        st.session_state.df_principal = df_work.drop(columns=col_to_drop)
                        st.success("Columnas borradas.")
                        st.rerun()
                
                st.markdown("---")
                st.write("**Detectar Outliers**")
                num_cols = df_work.select_dtypes(include=np.number).columns
                col_out = st.selectbox("Variable Numérica", num_cols, key="sel_out_col")
                
                if col_out:
                    # Visualización
                    fig, ax = plt.subplots(figsize=(6, 2))
                    sns.boxplot(x=df_work[col_out], color="#f87171", ax=ax) # Red color
                    st.pyplot(fig)
                    
                    if st.button("🔍 Analizar Outliers (IQR)", key="btn_prev_out"):
                        st.session_state['preview_out'] = True
                    
                    if st.session_state.get('preview_out'):
                        Q1 = df_work[col_out].quantile(0.25)
                        Q3 = df_work[col_out].quantile(0.75)
                        IQR = Q3 - Q1
                        mask_out = ((df_work[col_out] < (Q1 - 1.5 * IQR)) | (df_work[col_out] > (Q3 + 1.5 * IQR)))
                        outliers = df_work[mask_out]
                        
                        st.warning(f"⚠️ Se detectaron {len(outliers)} valores atípicos:")
                        st.dataframe(outliers[[col_out]].head())
                        
                        if st.button("✂️ Recortar Outliers (Confirmar)", key="btn_conf_out"):
                            st.session_state.historial_cambios.append(df_work.copy())
                            st.session_state.df_principal = df_work[~mask_out]
                            st.session_state['preview_out'] = False
                            st.success("Outliers eliminados.")
                            st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        # --- TAB 2: TRANSFORMACIÓN & FÓRMULAS ---
        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # HERRAMIENTA A: CAMBIAR TIPOS
            with st.expander("🔄 Cambiar Tipos de Datos", expanded=False):
                c_t1, c_t2, c_t3 = st.columns([2, 2, 1])
                with c_t1: col_type = st.selectbox("Columna", df_work.columns, key="sel_col_type")
                with c_t2: new_type = st.selectbox("Nuevo Tipo", ["Texto (String)", "Número (Float)", "Número (Int)", "Fecha (DateTime)"])
                with c_t3: 
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Convertir"):
                        try:
                            st.session_state.historial_cambios.append(df_work.copy())
                            if "Texto" in new_type: st.session_state.df_principal[col_type] = df_work[col_type].astype(str)
                            elif "Float" in new_type: st.session_state.df_principal[col_type] = pd.to_numeric(df_work[col_type], errors='coerce')
                            elif "Int" in new_type: st.session_state.df_principal[col_type] = pd.to_numeric(df_work[col_type], errors='coerce').fillna(0).astype(int)
                            elif "Fecha" in new_type: st.session_state.df_principal[col_type] = pd.to_datetime(df_work[col_type], errors='coerce')
                            st.success("Conversión exitosa.")
                            st.rerun()
                        except Exception as e: st.error(f"Error: {e}")

            # HERRAMIENTA B: CALCULADORA
            with st.expander("🧮 Calculadora de Variables (Feature Engineering)", expanded=False):
                st.info("Usa sintaxis pandas. Ej: `df['Peso'] / (df['Talla']**2)`")
                c_calc1, c_calc2 = st.columns(2)
                with c_calc1: new_var_name = st.text_input("Nombre Nueva Variable", "IMC")
                with c_calc2: formula = st.text_input("Fórmula", "Peso / (Talla**2)")
                
                if st.button("➕ Crear Variable"):
                    try:
                        st.session_state.historial_cambios.append(df_work.copy())
                        st.session_state.df_principal.eval(f"{new_var_name} = {formula}", inplace=True)
                        st.success(f"Variable '{new_var_name}' creada.")
                        st.rerun()
                    except Exception as e: st.error(f"Error en fórmula: {e}")

            # HERRAMIENTA C: TEXTO AVANZADO
            with st.expander("🔤 Operaciones de Texto (Split/Merge)", expanded=False):
                op_text = st.radio("Operación", ["Fusionar Columnas", "Dividir Columna"], horizontal=True)
                if op_text == "Fusionar Columnas":
                    c_txt1, c_txt2 = st.columns(2)
                    with c_txt1: col_a = st.selectbox("Columna A", df_work.columns)
                    with c_txt2: col_b = st.selectbox("Columna B", df_work.columns)
                    sep = st.text_input("Separador", " ")
                    new_col_merge = st.text_input("Nombre Resultante", "Nombre_Completo")
                    if st.button("Fusionar"):
                        st.session_state.historial_cambios.append(df_work.copy())
                        st.session_state.df_principal[new_col_merge] = df_work[col_a].astype(str) + sep + df_work[col_b].astype(str)
                        st.rerun()
                else:
                    col_split = st.selectbox("Columna a Dividir", df_work.columns)
                    sep_split = st.text_input("Delimitador", " ")
                    if st.button("Dividir"):
                        st.session_state.historial_cambios.append(df_work.copy())
                        split_df = df_work[col_split].astype(str).str.split(sep_split, expand=True)
                        for i in range(split_df.shape[1]):
                            st.session_state.df_principal[f"{col_split}_{i+1}"] = split_df[i]
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # --- TAB 3: FILTROS AVANZADOS ---
        with tab3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # 1. BUSCADOR GLOBAL
            st.subheader("🔍 Buscador Global")
            search_term = st.text_input("Buscar en todo el dataset", placeholder="Ej. Paciente 001")
            if search_term:
                mask = df_work.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
                st.dataframe(df_work[mask], use_container_width=True)
                st.caption(f"Resultados: {mask.sum()} filas encontradas.")
                boton_guardar_tabla(df_work[mask], f"Resultados Búsqueda Global: {search_term}", "global_search_table_btn")
            
            st.markdown("---")
            
            # 2. FILTRO MULTICRITERIO
            st.subheader("🌪️ Filtro Avanzado")
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1: col_filter = st.selectbox("Columna", df_work.columns, key="filter_col")
            with c_f2: condition = st.selectbox("Condición", ["Igual a", "Mayor que", "Menor que", "Contiene"])
            with c_f3: val_filter = st.text_input("Valor")
            
            if st.button("Aplicar Filtro"):
                try:
                    st.session_state.historial_cambios.append(df_work.copy())
                    if condition == "Igual a":
                        if pd.api.types.is_numeric_dtype(df_work[col_filter]):
                            st.session_state.df_principal = df_work[df_work[col_filter] == float(val_filter)]
                        else:
                            st.session_state.df_principal = df_work[df_work[col_filter].astype(str) == val_filter]
                    elif condition == "Mayor que":
                        st.session_state.df_principal = df_work[df_work[col_filter] > float(val_filter)]
                    elif condition == "Menor que":
                        st.session_state.df_principal = df_work[df_work[col_filter] < float(val_filter)]
                    elif condition == "Contiene":
                        st.session_state.df_principal = df_work[df_work[col_filter].astype(str).str.contains(val_filter, case=False)]
                    st.success("Filtro aplicado.")
                    st.rerun()
                except Exception as e: st.error(f"Error al filtrar: {e}")

            if st.button("🔄 Resetear Filtros (Volver al original)"):
                if len(st.session_state.historial_cambios) > 0:
                     st.session_state.df_principal = st.session_state.historial_cambios[0] 
                     st.session_state.historial_cambios = []
                     st.rerun()
                else:
                    st.info("No hay cambios para deshacer.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # --- TAB 4: AUDITORÍA DE CALIDAD ---
        with tab4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🕵️ Auditoría de Calidad (Detección de Errores)")
            
            # 1. DETECTOR DE "RESPUESTAS ROBÓTICAS"
            with st.expander("🤖 Detector de Respuestas Robóticas (Patrones Repetidos)", expanded=True):
                st.caption("Detecta filas donde el usuario marcó la misma opción en múltiples preguntas.")
                cols_robot = st.multiselect("Selecciona las preguntas de escala (ítems)", df_work.select_dtypes(include=np.number).columns, key="robot_cols")
                
                if cols_robot:
                    if st.button("🚩 Marcar sospechosos por Patrón"):
                        st.session_state.historial_cambios.append(df_work.copy())
                        std_devs = df_work[cols_robot].std(axis=1)
                        mask_robot = std_devs == 0
                        
                        if 'Flag_Calidad' not in st.session_state.df_principal.columns:
                            st.session_state.df_principal['Flag_Calidad'] = None
                        
                        st.session_state.df_principal.loc[mask_robot, 'Flag_Calidad'] = 'Patron_Repetido'
                        st.success(f"Se marcaron {mask_robot.sum()} filas sospechosas.")
                        st.rerun()

            st.markdown("---")

            # 2. CONSTRUCTOR DE REGLAS LÓGICAS
            with st.expander("🧠 Constructor de Reglas Lógicas (Inconsistencias)", expanded=False):
                st.caption("Define reglas para detectar contradicciones (ej. Edad < 10 AND Estado Civil == 'Viudo').")
                
                c_l1, c_l2, c_l3, c_l4, c_l5, c_l6, c_l7 = st.columns([2, 1, 2, 1, 2, 1, 2])
                with c_l1: col_a_log = st.selectbox("Columna A", df_work.columns, key="log_col_a")
                with c_l2: op_a_log = st.selectbox("Op", [">", "<", "==", "!=", ">=", "<="], key="log_op_a")
                with c_l3: val_a_log = st.text_input("Valor A", key="log_val_a")
                with c_l4: connector = st.selectbox("Y/O", ["AND", "OR"], key="log_conn")
                with c_l5: col_b_log = st.selectbox("Columna B", df_work.columns, key="log_col_b")
                with c_l6: op_b_log = st.selectbox("Op", ["==", "!=", ">", "<", ">=", "<="], key="log_op_b")
                with c_l7: val_b_log = st.text_input("Valor B", key="log_val_b")
                
                if st.button("🔍 Buscar Inconsistencias"):
                    try:
                        def fmt_val(col, val):
                            if pd.api.types.is_numeric_dtype(df_work[col]):
                                return val
                            else:
                                return f"'{val}'"

                        q_a = f"`{col_a_log}` {op_a_log} {fmt_val(col_a_log, val_a_log)}"
                        q_b = f"`{col_b_log}` {op_b_log} {fmt_val(col_b_log, val_b_log)}"
                        
                        connector_pd = "&" if connector == "AND" else "|"
                        
                        mask_logic = df_work.eval(f"({q_a}) {connector_pd} ({q_b})")
                        
                        if mask_logic.any():
                            st.warning(f"⚠️ Se encontraron {mask_logic.sum()} inconsistencias.")
                            st.dataframe(df_work[mask_logic].head())
                            
                            if st.button("🚩 Marcar por Lógica", key="btn_flag_logic"):
                                st.session_state.historial_cambios.append(df_work.copy())
                                if 'Flag_Calidad' not in st.session_state.df_principal.columns:
                                    st.session_state.df_principal['Flag_Calidad'] = None
                                
                                mask_existing = st.session_state.df_principal['Flag_Calidad'].notna()
                                st.session_state.df_principal.loc[mask_logic & ~mask_existing, 'Flag_Calidad'] = 'Inconsistencia_Logica'
                                st.session_state.df_principal.loc[mask_logic & mask_existing, 'Flag_Calidad'] = st.session_state.df_principal.loc[mask_logic & mask_existing, 'Flag_Calidad'] + ', Inconsistencia_Logica'
                                
                                st.success("Filas marcadas.")
                                st.rerun()
                        else:
                            st.info("No se encontraron coincidencias con esta regla.")
                            
                    except Exception as e:
                        st.error(f"Error en la regla: {e}")

            st.markdown("---")

            # 3. VALIDACIÓN DE RANGOS BIOLÓGICOS
            with st.expander("🧬 Validación de Rangos Biológicos (Valores Imposibles)", expanded=False):
                st.caption("Detecta valores fuera de lo fisiológicamente posible (ej. Peso < 10kg o Glucosa > 1000).")
                col_bio = st.selectbox("Variable Numérica", df_work.select_dtypes(include=np.number).columns, key="bio_col")
                
                c_b1, c_b2 = st.columns(2)
                with c_b1: min_bio = st.number_input("Mínimo Aceptable", value=0.0)
                with c_b2: max_bio = st.number_input("Máximo Aceptable", value=100.0)
                
                if st.button("🚩 Marcar Fuera de Rango"):
                    st.session_state.historial_cambios.append(df_work.copy())
                    mask_bio = (df_work[col_bio] < min_bio) | (df_work[col_bio] > max_bio)
                    
                    if mask_bio.any():
                        if 'Flag_Calidad' not in st.session_state.df_principal.columns:
                            st.session_state.df_principal['Flag_Calidad'] = None
                            
                        mask_existing = st.session_state.df_principal['Flag_Calidad'].notna()
                        st.session_state.df_principal.loc[mask_bio & ~mask_existing, 'Flag_Calidad'] = 'Fuera_Rango'
                        st.session_state.df_principal.loc[mask_bio & mask_existing, 'Flag_Calidad'] = st.session_state.df_principal.loc[mask_bio & mask_existing, 'Flag_Calidad'] + ', Fuera_Rango'
                        
                        st.success(f"Se marcaron {mask_bio.sum()} valores fuera de rango.")
                        st.rerun()
                    else:
                        st.info("Todos los valores están dentro del rango.")

            st.markdown("---")

            # 4. GESTIÓN FINAL DE ERRORES
            st.subheader("🗑️ Gestión de Errores Detectados")
            if 'Flag_Calidad' in df_work.columns:
                df_flagged = df_work[df_work['Flag_Calidad'].notna() & (df_work['Flag_Calidad'] != '')]
                
                if not df_flagged.empty:
                    st.warning("⚠️ Revisa esta lista. Si un dato es REAL (Falso Positivo), borra el texto de la columna 'Flag_Calidad' manualmente o cámbialo a 'Validado'.")
                    
                    edited_flags = st.data_editor(
                        df_flagged, 
                        use_container_width=True, 
                        key="editor_errors_human",
                        disabled=[c for c in df_flagged.columns if c != 'Flag_Calidad'] 
                    )
                    
                    col_keep, col_del = st.columns(2)
                    
                    with col_keep:
                        if st.button("✅ Conservar Todo (Ignorar Alertas)", type="secondary"):
                            st.session_state.historial_cambios.append(df_work.copy())
                            st.session_state.df_principal['Flag_Calidad'] = None
                            st.success("Alertas ignoradas. Se conservaron todos los datos.")
                            st.rerun()

                    with col_del:
                        if st.button("🗑️ Eliminar SOLO los marcados como Error", type="primary"):
                            st.session_state.historial_cambios.append(df_work.copy())
                            
                            for idx, row in edited_flags.iterrows():
                                st.session_state.df_principal.loc[idx, 'Flag_Calidad'] = row['Flag_Calidad']
                            
                            df_final_check = st.session_state.df_principal
                            mask_to_delete = (
                                df_final_check['Flag_Calidad'].notna() & 
                                (df_final_check['Flag_Calidad'] != '') & 
                                (df_final_check['Flag_Calidad'] != 'Validado')
                            )
                            
                            rows_deleted = mask_to_delete.sum()
                            st.session_state.df_principal = df_final_check[~mask_to_delete].copy()
                            
                            st.success(f"Se eliminaron {rows_deleted} filas marcadas como error. Los datos validados se conservaron.")
                            st.rerun()
                else:
                    st.success("✅ Todo limpio. No hay errores detectados.")
            else:
                st.info("No se han detectado problemas de calidad aún.")

            st.markdown('</div>', unsafe_allow_html=True)

        # VISUALIZADOR DE DATOS ACTUAL
        st.markdown("### 📋 Vista Previa de Datos")
        
        ver_todas = st.checkbox("Mostrar todas las filas (Puede ser lento si hay muchos datos)")
        
        if ver_todas:
            st.dataframe(df_work, use_container_width=True)
            st.caption(f"Mostrando el total de {df_work.shape[0]} filas.")
            boton_guardar_tabla(df_work, "Vista Completa de Datos", "data_full_table_btn")
        else:
            st.dataframe(df_work.head(50), use_container_width=True)
            st.caption(f"Mostrando vista previa de 50 filas (Total: {df_work.shape[0]}). Activa la casilla arriba para ver todo.")
            boton_guardar_tabla(df_work.head(50), "Vista Previa de Datos", "data_preview_table_btn")
        
        if len(st.session_state.historial_cambios) > 0:
            if st.button("↩️ Deshacer último cambio global"):
                st.session_state.df_principal = st.session_state.historial_cambios.pop()
                st.success("Cambio revertido.")
                st.rerun()
