import streamlit as st
import pandas as pd
import numpy as np
from functools import wraps
import os
import time
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import psutil for accurate memory monitoring, fallback to basic if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class PerformanceOptimizer:
    """
    Performance optimization helper for BioStat Easy.
    Implements 3-layer strategy: Chunking, Caching, Memory Management.
    """

    def __init__(self):
        pass

    def get_memory_usage(self):
        """Returns current process memory usage in MB."""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / 1024 / 1024
                return mem
            else:
                # Fallback for unix-based systems
                import resource
                mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                # On Mac, ru_maxrss is in bytes? No, usually KB on Linux, bytes on Mac? 
                # Actually commonly KB. Let's stick to simple if robust.
                # If we aren't sure, returning 0 is safer than crashing.
                return mem 
        except Exception:
            return 0.0

    def memory_profiler(self, func):
        """Decorator to monitor memory usage before and after function execution."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            mem_before = self.get_memory_usage()
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            mem_after = self.get_memory_usage()
            duration = time.time() - start_time
            
            diff = mem_after - mem_before
            logger.info(f"💾 MEMORY: {func.__name__} | Diff: {diff:.2f} MB | Total: {mem_after:.2f} MB | Time: {duration:.2f}s")
            
            # Simple heuristic: if diff > 100MB, suggest manual cleanup (future feature)
            return result
        return wrapper

    def create_session_cache(self, key, func, ttl=3600, *args, **kwargs):
        """
        Cache mechanism using st.session_state with TTL (Time To Live).
        """
        cache_key = f"cache_{key}"
        timestamp_key = f"ts_{key}"
        
        current_time = time.time()
        
        # Check if exists and valid
        if cache_key in st.session_state and timestamp_key in st.session_state:
            cached_ts = st.session_state[timestamp_key]
            if current_time - cached_ts < ttl:
                # Valid cache
                return st.session_state[cache_key]
        
        # Calculate fresh
        result = func(*args, **kwargs)
        
        # Store in cache
        st.session_state[cache_key] = result
        st.session_state[timestamp_key] = current_time
        
        return result

    def paginate_dataframe(self, df, page_size=100, key_prefix="table"):
        """
        Manages pagination state in Streamlit.
        Returns: slice of dataframe, current_page, total_pages
        """
        if df is None or df.empty:
            return df, 1, 1

        total_rows = len(df)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        
        # State key for this specific table
        state_key = f"page_{key_prefix}"
        
        if state_key not in st.session_state:
            st.session_state[state_key] = 1
            
        current_page = st.session_state[state_key]
        
        # Bound Check
        if current_page > total_pages: current_page = total_pages
        if current_page < 1: current_page = 1
        st.session_state[state_key] = current_page
        
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        
        return df.iloc[start_idx:end_idx], current_page, total_pages

    def safe_render_large_table(self, df, max_rows=100, key_prefix="large_df"):
        """
        Renders a dataframe safe from truncation limits. 
        Auto-paginates if larger than max_rows.
        Includes CSV download option.
        """
        if df is None:
            st.info("No hay datos para mostrar.")
            return

        if len(df) <= max_rows:
            st.dataframe(df, use_container_width=True)
            return

        # Pagination Logic
        st.markdown(f"**Vista previa ({len(df)} filas totales)**")
        
        cols = st.columns([2, 5, 1, 1])
        
        state_key = f"page_{key_prefix}"
        
        # Controls
        with cols[2]:
            if st.button("◀", key=f"prev_{key_prefix}"):
                if st.session_state.get(state_key, 1) > 1:
                    st.session_state[state_key] -= 1
                    st.rerun()
                    
        with cols[3]:
            # We need to compute total pages to know if we can go next
            total_p = (len(df) + max_rows - 1) // max_rows
            if st.button("▶", key=f"next_{key_prefix}"):
                if st.session_state.get(state_key, 1) < total_p:
                    st.session_state[state_key] += 1
                    st.rerun()

        # Get Slice
        df_slice, curr, tot = self.paginate_dataframe(df, max_rows, key_prefix)
        
        with cols[1]:
            st.caption(f"Página {curr} de {tot}")

        # Render Slice
        st.dataframe(df_slice, use_container_width=True)
        
        # Download Option using standard pandas
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Descargar Tabla Completa (CSV)",
            data=csv,
            file_name=f"{key_prefix}_full.csv",
            mime="text/csv",
            key=f"dl_{key_prefix}"
        )

    def safe_render_large_plot(self, fig, max_elements=5000):
        """
        Safe rendering for plots.
        Currently a wrapper for standard rendering, placeholders for more complex logic
        like downsampling if points > max_elements.
        """
        # For now, just robust render
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error renderizando gráfico: {e}")
            # Fallback to static if interactive fails?
            # st.pyplot(fig) # Only if fig is matplotlib compatible, but usually it's plotly here.
            pass

optimizer = PerformanceOptimizer()
