import json
import os
import requests
from pathlib import Path

# Create directories if they don't exist
os.makedirs('indigo-league/silhouettes', exist_ok=True)
os.makedirs('indigo-league/reveals', exist_ok=True)

# Read the JSON file
with open('pokemon-data-page-3.json', 'r') as f:
    pokemon_data = json.load(f)

# Download function
def download_image(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded: {save_path}")
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")

# Download images for each Pokemon
for pokemon in pokemon_data:
    dex_num = pokemon['pokemon']
    
    # Download silhouette
    silhouette_path = f'indigo-league/silhouettes/{dex_num}.png'
    download_image(pokemon['silhouette'], silhouette_path)
    
    # Download reveal
    reveal_path = f'indigo-league/reveals/{dex_num}.png'
    download_image(pokemon['reveal'], reveal_path)

print("Download complete!") 