import os
import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
from tqdm import tqdm
import re
import argparse
import aiohttp
import aiofiles

def create_directories():
    """Create directories for storing images."""
    base_dir = Path("indigo-league")
    silhouettes_dir = base_dir / "silhouettes"
    reveals_dir = base_dir / "reveals"
    
    silhouettes_dir.mkdir(parents=True, exist_ok=True)
    reveals_dir.mkdir(parents=True, exist_ok=True)
    
    return silhouettes_dir, reveals_dir

async def get_pokemon_info(page, url):
    """Scrape Pokemon information from the webpage using Playwright."""
    print(f"Fetching URL: {url}")
    await page.goto(url)
    
    # Wait for the content to load
    await page.wait_for_selector('.segment-entry', timeout=10000)
    
    # Get all Pokemon entries
    pokemon_entries = []
    
    # Find all Pokemon entries
    entries = await page.query_selector_all('.segment-entry')
    print(f"\nFound {len(entries)} segment entries")
    
    for entry in entries:
        try:
            # Extract Pokemon name
            name_elem = await entry.query_selector('h3')
            if not name_elem:
                print("No h3 element found in entry")
                continue
                
            name = await name_elem.text_content()
            name = name.strip()
            print(f"\nProcessing Pokemon: {name}")
            
            # Find all images
            images = await entry.query_selector_all('img')
            print(f"Found {len(images)} images")
            
            if len(images) >= 2:
                silhouette_url = await images[0].get_attribute('src')
                reveal_url = await images[1].get_attribute('src')
                
                # Extract episode number
                text_content = await entry.text_content()
                episode_match = re.search(r'SL\s*(\d+)', text_content)
                episode_num = episode_match.group(1) if episode_match else "000"
                
                print(f"Episode: {episode_num}")
                print(f"Silhouette URL: {silhouette_url}")
                print(f"Reveal URL: {reveal_url}")
                
                pokemon_entries.append({
                    'name': name.lower(),
                    'episode': episode_num,
                    'silhouette_url': silhouette_url,
                    'reveal_url': reveal_url
                })
            else:
                print(f"Not enough images found for {name}")
                
        except Exception as e:
            print(f"Error processing entry: {e}")
            continue
    
    return pokemon_entries

async def download_image(session, url, save_path):
    """Download an image from URL and save it to the specified path."""
    print(f"Downloading: {url}")
    async with session.get(url) as response:
        response.raise_for_status()
        async with aiofiles.open(save_path, 'wb') as f:
            await f.write(await response.read())
    print(f"Saved to: {save_path}")

async def main_async():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Scrape Pokemon images from Pocket Monsters website')
    parser.add_argument('url', help='URL of the page to scrape (e.g., https://pocketmonsters.net/segments/Who\'s%20that%20Pokémon/Indigo%20League)')
    parser.add_argument('--pages', type=int, default=4, help='Number of pages to scrape (default: 4)')
    args = parser.parse_args()

    # Create directories
    silhouettes_dir, reveals_dir = create_directories()
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Scrape all pages
        all_pokemon = []
        for page_num in range(1, args.pages + 1):
            url = f"{args.url}?page={page_num}"
            print(f"\nScraping page {page_num}...")
            pokemon_entries = await get_pokemon_info(page, url)
            all_pokemon.extend(pokemon_entries)
        
        await browser.close()
    
    print(f"\nFound {len(all_pokemon)} Pokemon entries")
    
    if len(all_pokemon) == 0:
        print("No Pokemon entries found. Please check the URL and page structure.")
        return
    
    # Download images
    async with aiohttp.ClientSession() as session:
        for entry in tqdm(all_pokemon, desc="Downloading images"):
            # Create filenames
            silhouette_filename = f"{entry['episode']}-{entry['name']}.png"
            reveal_filename = f"{entry['episode']}-{entry['name']}.png"
            
            # Download silhouette
            silhouette_path = silhouettes_dir / silhouette_filename
            try:
                await download_image(session, entry['silhouette_url'], silhouette_path)
            except Exception as e:
                print(f"Error downloading silhouette for {entry['name']}: {e}")
            
            # Download reveal
            reveal_path = reveals_dir / reveal_filename
            try:
                await download_image(session, entry['reveal_url'], reveal_path)
            except Exception as e:
                print(f"Error downloading reveal for {entry['name']}: {e}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 