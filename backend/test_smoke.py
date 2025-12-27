import requests
import pandas as pd
import io

BASE_URL = "http://localhost:8000"

def test_api():
    # 1. Health
    try:
        r = requests.get(f"{BASE_URL}/")
        print(f"Health: {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"Server not running? {e}")
        return

    # 2. Upload
    csv_content = "age,weight,height\n25,70,175\n30,80,180\n35,75,178\n40,85,182\n22,60,165"
    files = {'file': ('test.csv', csv_content, 'text/csv')}
    
    r_upload = requests.post(f"{BASE_URL}/api/upload", files=files)
    if r_upload.status_code != 200:
        print(f"Upload Failed: {r_upload.text}")
        return
    
    upload_data = r_upload.json()
    session_id = upload_data['session_id']
    print(f"Upload Success. Session ID: {session_id}")
    print(f"Metadata: {upload_data['columns']}")

    # 3. Stats
    payload = {"session_id": session_id, "variable": "age"}
    r_stats = requests.post(f"{BASE_URL}/api/stats/descriptive", json=payload)
    
    if r_stats.status_code == 200:
        print("Stats Success:")
        print(r_stats.json())
    else:
        print(f"Stats Failed: {r_stats.text}")

    # 4. Plot
    plot_payload = {
        "session_id": session_id,
        "plot_type": "hist",
        "variable": "weight"
    }
    r_plot = requests.post(f"{BASE_URL}/api/plots/generate", json=plot_payload)
    if r_plot.status_code == 200:
        print("Plot Success (Base64 received)")
    else:
        print(f"Plot Failed: {r_plot.text}")

if __name__ == "__main__":
    test_api()
