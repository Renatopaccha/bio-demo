const API_URL = 'http://localhost:8000/api';

export interface UploadResponse {
    session_id: string;
    filename: string;
    rows: number;
    columns: {
        name: string;
        type: string;
        is_numeric: boolean;
        missing: number;
    }[];
}

export const uploadFile = async (file: File): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`Error uploading file: ${response.statusText}`);
    }

    return response.json();
};

export const getDescriptiveStats = async (sessionId: string, variable: string) => {
    const response = await fetch(`${API_URL}/stats/descriptive`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            session_id: sessionId,
            variable,
        }),
    });

    if (!response.ok) {
        throw new Error(`Error fetching stats: ${response.statusText}`);
    }

    return response.json();
};

export const generatePlot = async (sessionId: string, plotType: string, variable: string, variableY?: string, hue?: string) => {
    const response = await fetch(`${API_URL}/plots/generate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            session_id: sessionId,
            plot_type: plotType,
            variable,
            variable_y: variableY,
            hue,
        }),
    });

    if (!response.ok) {
        throw new Error(`Error generating plot: ${response.statusText}`);
    }

    return response.json(); // { image_base64: "..." }
};
