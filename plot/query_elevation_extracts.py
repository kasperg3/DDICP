import os
import requests
from pathlib import Path
import shutil
# Load API key from file.
with open("data/gpxz_api_key.txt", "r") as file:
    os.environ["GPXZ_API_KEY"] = file.read().strip()
# Build request
# Hilly 9.448553757616736, 54.95645626143309, 9.455719464994786, 54.96478160452244
# Water: (9.456748389817477, 55.11382229912007, 9.477989692140483, 55.12666669385226) 
# Flat:
    # "bbox_top": "54.96100438506525",
    # "bbox_bottom": "54.952615782731385",
    # "bbox_left": "9.416186034424953",
    # "bbox_right": "9.42699155445743",
    
# Hilly
    # "bbox_top": "54.96478160452244",
    # "bbox_bottom": "54.95645626143309",
    # "bbox_left": "9.448553757616736",
    # "bbox_right": "9.455719464994786",

# Water
    # "bbox_top": "55.12666669385226",
    # "bbox_bottom": "55.11382229912007",
    # "bbox_left": "9.456748389817477",
    # "bbox_right": "9.477989692140483",
    
query_params = {
    "bbox_top": "54.96478160452244",
    "bbox_bottom": "54.95645626143309",
    "bbox_left": "9.448553757616736",
    "bbox_right": "9.455719464994786",
    "res_m": 2,  # Metres.
    "api-key": os.environ["GPXZ_API_KEY"],
}

# Query data.
response = requests.get(
    "https://api.gpxz.io/v1/elevation/hires-raster",
    params=query_params,
    stream=True,
)
response.raise_for_status()

# Save to file.
dest_path = Path("data/raster/HillyTerrainNature.geotiff")
with open(dest_path, "wb") as f:
    shutil.copyfileobj(response.raw, f)