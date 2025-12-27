import sys
import os
import uuid
import io
import base64
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Prevent GUI crash on macOS
import matplotlib.pyplot as plt

# --- PATH SETUP ---
# Add parent directory to sys.path to allow importing 'modules'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing core modules
# We use try-except blocks to gracefully handle import errors if paths are wrong
from modules.stats import core
from modules import plots

# --- FASTAPI SETUP ---
app = FastAPI(title="Biometric API", version="1.0.0")

# Configure CORS (Allow All for Development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev (e.g., localhost:5173, localhost:3000)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SESSION MANAGER (In-Memory) ---
class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, pd.DataFrame] = {}

    def create_session(self, df: pd.DataFrame) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = df
        return session_id

    def get_session(self, session_id: str) -> Optional[pd.DataFrame]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

session_manager = SessionManager()

# --- PYDANTIC MODELS ---
class DescriptiveStatsRequest(BaseModel):
    session_id: str
    variable: str

class PlotRequest(BaseModel):
    session_id: str
    plot_type: str # 'hist', 'box', 'scatter', 'bar'
    variable: str
    variable_y: Optional[str] = None
    hue: Optional[str] = None

# --- HELPER FUNCTIONS ---
def fig_to_base64(fig) -> str:
    """Converts a Matplotlib Figure to a Base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig) # Close to free memory
    return img_str

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Biometric API is running"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads an Excel or CSV file, loads it into pandas, and returns a Session ID.
    Also returns metadata about columns and types.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file filename")

    try:
        contents = await file.read()
        
        # Determine loader based on extension
        if file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents))
        elif file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use .xlsx or .csv")

        # Create Session
        session_id = session_manager.create_session(df)

        # Generate Metadata
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            columns_info.append({
                "name": col,
                "type": dtype,
                "is_numeric": is_numeric,
                "missing": int(df[col].isna().sum())
            })

        return {
            "session_id": session_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": columns_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/api/stats/descriptive")
def get_descriptive_stats(request: DescriptiveStatsRequest):
    """
    Calculates detailed descriptive statistics for a specific variable.
    Uses modules.stats.core.calculate_descriptive_stats
    """
    df = session_manager.get_session(request.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if request.variable not in df.columns:
        raise HTTPException(status_code=404, detail=f"Variable '{request.variable}' not found in dataset")

    # Use existing core logic
    try:
        stats_result = core.calculate_descriptive_stats(df[request.variable])
        # Also run Normality Check
        normality_result = core.check_normality(df[request.variable])
        advanced_result = core.calculate_advanced_descriptive_stats(df[request.variable])
        
        # Merge results
        merged_results = {**stats_result, **normality_result, **advanced_result}
        
        # Clean and Convert Logic
        clean_results = {}
        for k, v in merged_results.items():
            # Handle Numpy Integers
            if isinstance(v, (np.integer, int)):
                 clean_results[k] = int(v)
            # Handle Numpy Floats
            elif isinstance(v, (np.floating, float)):
                if np.isnan(v) or np.isinf(v):
                    clean_results[k] = None
                else:
                    clean_results[k] = float(v)
            # Handle other types (str, list, None)
            else:
                clean_results[k] = v

        return clean_results

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Statistical calculation error: {str(e)}")

@app.post("/api/plots/generate")
def generate_plot(request: PlotRequest):
    """
    Generates a plot using modules.plots and returns it as a Base64 string.
    """
    df = session_manager.get_session(request.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        fig = None
        
        # Dispatch to modules.plots functions
        if request.plot_type == 'hist':
            fig = plots.plot_distribucion_numerica(df, request.variable, tipo='hist', show_normal=True)
            
        elif request.plot_type == 'box':
            fig = plots.plot_boxplot_univariado(df, request.variable)
            
        elif request.plot_type == 'bar':
            fig = plots.plot_barras_categorico(df, request.variable)
            
        elif request.plot_type == 'scatter':
            if not request.variable_y:
                 raise HTTPException(status_code=400, detail="Scatter plot requires variable_y")
            fig = plots.plot_scatter_con_regresion(df, request.variable, request.variable_y, color_var=request.hue, add_reg=True)
        
        else:
             raise HTTPException(status_code=400, detail=f"Unknown plot type: {request.plot_type}")

        if fig:
            img_b64 = fig_to_base64(fig)
            return {"image_base64": img_b64}
        else:
             raise HTTPException(status_code=500, detail="Failed to generate plot (fig is None)")

    except Exception as e:
        # Print stack trace in server logs (useful for debugging)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Plotting error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
