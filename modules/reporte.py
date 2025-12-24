import streamlit as st
import pandas as pd
import io

def render_reporte():
    st.header("📦 Mi Reporte Acumulado")
    st.markdown("Aquí se guardan las tablas y gráficos que seleccionaste para exportar.")
    
    # Verificar si hay items
    if 'reporte_items' not in st.session_state or len(st.session_state['reporte_items']) == 0:
        st.info("📭 Tu reporte está vacío.")
        st.markdown("""
            **¿Cómo añadir elementos?**
            
            Ve a cualquier sección (Ej: Estadística Descriptiva) y busca el botón **'➕ Añadir al Reporte'** 
            que aparece encima de las tablas o gráficos.
        """)
        return
    
    # Mostrar lista de items
    st.write(f"Tienes **{len(st.session_state['reporte_items'])} elementos** listos para exportar.")
    
    # Botón de acción principal
    col_main, col_trash = st.columns([3, 1])
    
    with col_main:
        # Lógica de Generación de Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            fmt_header = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            
            for i, item in enumerate(st.session_state['reporte_items']):
                # Limpiar nombre de hoja
                safe_title = "".join([c for c in item['titulo'] if c.isalnum() or c in (' ', '_')])[:25]
                sheet_name = f"{i+1}_{safe_title}"
                
                if item['tipo'] == 'df':
                    item['data'].to_excel(writer, sheet_name=sheet_name, index=False)
                    worksheet = writer.sheets[sheet_name]
                    for col_num, value in enumerate(item['data'].columns.values):
                        worksheet.write(0, col_num, value, fmt_header)
                        worksheet.set_column(col_num, col_num, 20)
                        
                elif item['tipo'] == 'img':
                    worksheet = workbook.add_worksheet(sheet_name)
                    worksheet.write(0, 0, f"Gráfico: {item['titulo']}", fmt_header)
                    worksheet.insert_image(2, 1, item['titulo'], {'image_data': item['data']})
        
        st.download_button(
            label="📥 Descargar Reporte Excel (.xlsx)",
            data=output.getvalue(),
            file_name="Reporte_Biometric_Final.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="btn_descarga_final_page",
            type="primary",
            use_container_width=True
        )
    
    with col_trash:
        if st.button("🗑️ Vaciar Todo", type="secondary", use_container_width=True):
            st.session_state['reporte_items'] = []
            st.rerun()
    
    st.divider()
    
    # Vista Previa de Elementos
    st.subheader("Vista Previa")
    for i, item in enumerate(st.session_state['reporte_items']):
        with st.expander(f"📄 {i+1}. {item['titulo']}"):
            if item['tipo'] == 'df':
                st.dataframe(item['data'], use_container_width=True, height=200)
            elif item['tipo'] == 'img':
                st.image(item['data'])
